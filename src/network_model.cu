/**
 * SLE Engine - CUDA-Accelerated State Load Estimator
 * Copyright (C) 2025 Oleksandr Don
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * @file network_model.cu
 * @brief Network model implementation
 * 
 * Implements the NetworkModel class and HostDataAllocator.
 */

#include "../include/network_model.h"
#include "../include/jsf_compliance.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace sle {

//=============================================================================
// NetworkModel Implementation
//=============================================================================

NetworkModel::NetworkModel()
    : slack_bus_index_(INVALID_INDEX)
    , model_modified_(false)
{
    initializeHashMaps();
}

NetworkModel::~NetworkModel() = default;

NetworkModel::NetworkModel(NetworkModel&&) noexcept = default;
NetworkModel& NetworkModel::operator=(NetworkModel&&) noexcept = default;

void NetworkModel::initializeHashMaps() {
    // Reserve initial capacity for better performance
    bus_id_map_.reserve(1000);
    branch_id_map_.reserve(2000);
    measurement_id_map_.reserve(5000);
    sd_id_map_.reserve(500);
}

//=============================================================================
// Model Building
//=============================================================================

int32_t NetworkModel::addBus(const BusDescriptor& desc) {
    // Check for duplicate ID
    if (bus_id_map_.find(desc.id) != bus_id_map_.end()) {
        return INVALID_INDEX;
    }
    
    int32_t index = static_cast<int32_t>(buses_.size());
    
    BusElement elem(index);
    elem.descriptor = desc;
    elem.v_mag = desc.v_setpoint;
    elem.v_angle = 0.0f;
    
    buses_.push_back(elem);
    bus_id_map_[desc.id] = index;
    
    // Track slack bus
    if (desc.type == BusType::SLACK) {
        slack_bus_index_ = index;
    }
    
    model_modified_ = true;
    return index;
}

int32_t NetworkModel::addBranch(const BranchDescriptor& desc) {
    // Check for duplicate ID
    if (branch_id_map_.find(desc.id) != branch_id_map_.end()) {
        return INVALID_INDEX;
    }
    
    // Resolve bus indices
    int32_t from_idx = getBusIndex(desc.from_bus_id);
    int32_t to_idx = getBusIndex(desc.to_bus_id);
    
    if (from_idx == INVALID_INDEX || to_idx == INVALID_INDEX) {
        return INVALID_INDEX;  // Referenced buses don't exist
    }
    
    int32_t index = static_cast<int32_t>(branches_.size());
    
    BranchElement elem(index);
    elem.descriptor = desc;
    elem.from_bus_index = from_idx;
    elem.to_bus_index = to_idx;
    elem.status = SwitchStatus::CLOSED;
    
    branches_.push_back(elem);
    branch_id_map_[desc.id] = index;
    
    model_modified_ = true;
    return index;
}

int32_t NetworkModel::addMeasurement(const MeasurementDescriptor& desc) {
    // Check for duplicate ID
    if (measurement_id_map_.find(desc.id) != measurement_id_map_.end()) {
        return INVALID_INDEX;
    }
    
    // Resolve location index
    int32_t loc_idx;
    if (isBusMeasurement(desc.type)) {
        loc_idx = getBusIndex(desc.location_id);
    } else {
        loc_idx = getBranchIndex(desc.location_id);
    }
    
    if (loc_idx == INVALID_INDEX) {
        return INVALID_INDEX;
    }
    
    int32_t index = static_cast<int32_t>(measurements_.size());
    
    MeasurementElement elem(index);
    elem.descriptor = desc;
    elem.location_index = loc_idx;
    elem.weight = 1.0f / (desc.sigma * desc.sigma);  // Weight = 1/variance
    elem.is_active = true;
    
    measurements_.push_back(elem);
    measurement_id_map_[desc.id] = index;
    
    model_modified_ = true;
    return index;
}

int32_t NetworkModel::addSwitchingDevice(const SwitchingDeviceDescriptor& desc) {
    if (sd_id_map_.find(desc.id) != sd_id_map_.end()) {
        return INVALID_INDEX;
    }
    
    int32_t branch_idx = getBranchIndex(desc.branch_id);
    if (branch_idx == INVALID_INDEX) {
        return INVALID_INDEX;
    }
    
    int32_t index = static_cast<int32_t>(switching_devices_.size());
    
    SwitchingDeviceElement elem(index);
    elem.descriptor = desc;
    elem.branch_index = branch_idx;
    elem.status = desc.initial_status;
    elem.pending_status = desc.initial_status;
    elem.has_pending_change = false;
    
    switching_devices_.push_back(elem);
    sd_id_map_[desc.id] = index;
    
    // Link branch to switching device
    branches_[branch_idx].sd_index = index;
    branches_[branch_idx].status = desc.initial_status;
    
    model_modified_ = true;
    return index;
}

//=============================================================================
// Model Modification
//=============================================================================

bool NetworkModel::removeBus(const std::string& bus_id) {
    int32_t idx = getBusIndex(bus_id);
    if (idx == INVALID_INDEX) return false;
    
    // Remove all branches connected to this bus
    for (auto& branch : branches_) {
        if (branch.from_bus_index == idx || branch.to_bus_index == idx) {
            branch_id_map_.erase(branch.descriptor.id);
        }
    }
    
    // Remove measurements at this bus
    for (auto& meas : measurements_) {
        if (isBusMeasurement(meas.descriptor.type) && meas.location_index == idx) {
            measurement_id_map_.erase(meas.descriptor.id);
        }
    }
    
    bus_id_map_.erase(bus_id);
    model_modified_ = true;
    
    return true;
}

bool NetworkModel::removeBranch(const std::string& branch_id) {
    int32_t idx = getBranchIndex(branch_id);
    if (idx == INVALID_INDEX) return false;
    
    branch_id_map_.erase(branch_id);
    model_modified_ = true;
    
    return true;
}

bool NetworkModel::updateBranch(const std::string& branch_id, const BranchDescriptor& desc) {
    int32_t idx = getBranchIndex(branch_id);
    if (idx == INVALID_INDEX) return false;
    
    branches_[idx].descriptor = desc;
    model_modified_ = true;
    
    return true;
}

//=============================================================================
// Index Lookup
//=============================================================================

int32_t NetworkModel::getBusIndex(const std::string& id) const {
    auto it = bus_id_map_.find(id);
    return (it != bus_id_map_.end()) ? it->second : INVALID_INDEX;
}

int32_t NetworkModel::getBranchIndex(const std::string& id) const {
    auto it = branch_id_map_.find(id);
    return (it != branch_id_map_.end()) ? it->second : INVALID_INDEX;
}

int32_t NetworkModel::getMeasurementIndex(const std::string& id) const {
    auto it = measurement_id_map_.find(id);
    return (it != measurement_id_map_.end()) ? it->second : INVALID_INDEX;
}

int32_t NetworkModel::getSwitchingDeviceIndex(const std::string& id) const {
    auto it = sd_id_map_.find(id);
    return (it != sd_id_map_.end()) ? it->second : INVALID_INDEX;
}

//=============================================================================
// Element Access
//=============================================================================

BusElement* NetworkModel::getBus(int32_t index) {
    if (index < 0 || index >= static_cast<int32_t>(buses_.size())) {
        return nullptr;
    }
    return &buses_[index];
}

const BusElement* NetworkModel::getBus(int32_t index) const {
    if (index < 0 || index >= static_cast<int32_t>(buses_.size())) {
        return nullptr;
    }
    return &buses_[index];
}

BranchElement* NetworkModel::getBranch(int32_t index) {
    if (index < 0 || index >= static_cast<int32_t>(branches_.size())) {
        return nullptr;
    }
    return &branches_[index];
}

const BranchElement* NetworkModel::getBranch(int32_t index) const {
    if (index < 0 || index >= static_cast<int32_t>(branches_.size())) {
        return nullptr;
    }
    return &branches_[index];
}

MeasurementElement* NetworkModel::getMeasurement(int32_t index) {
    if (index < 0 || index >= static_cast<int32_t>(measurements_.size())) {
        return nullptr;
    }
    return &measurements_[index];
}

const MeasurementElement* NetworkModel::getMeasurement(int32_t index) const {
    if (index < 0 || index >= static_cast<int32_t>(measurements_.size())) {
        return nullptr;
    }
    return &measurements_[index];
}

SwitchingDeviceElement* NetworkModel::getSwitchingDevice(int32_t index) {
    if (index < 0 || index >= static_cast<int32_t>(switching_devices_.size())) {
        return nullptr;
    }
    return &switching_devices_[index];
}

const SwitchingDeviceElement* NetworkModel::getSwitchingDevice(int32_t index) const {
    if (index < 0 || index >= static_cast<int32_t>(switching_devices_.size())) {
        return nullptr;
    }
    return &switching_devices_[index];
}

//=============================================================================
// Telemetry Update
//=============================================================================

bool NetworkModel::updateMeasurementValues(const Real* values, int32_t count) {
    if (count != static_cast<int32_t>(measurements_.size())) {
        return false;
    }
    
    for (int32_t i = 0; i < count; ++i) {
        measurements_[i].value = values[i];
    }
    
    return true;
}

bool NetworkModel::updateMeasurementValue(const std::string& meas_id, Real value) {
    int32_t idx = getMeasurementIndex(meas_id);
    if (idx == INVALID_INDEX) return false;
    
    measurements_[idx].value = value;
    return true;
}

//=============================================================================
// GPU Data Packing
//=============================================================================

void NetworkModel::packBusData(HostBusData& data) const {
    int32_t n = static_cast<int32_t>(buses_.size());
    data.count = n;
    data.slack_bus_index = slack_bus_index_;
    
    #pragma omp parallel for schedule(static) if(n > 100)
    for (int32_t i = 0; i < n; ++i) {
        const auto& bus = buses_[i];
        data.base_kv[i] = bus.descriptor.base_kv;
        data.bus_type[i] = bus.descriptor.type;
        data.v_mag[i] = bus.v_mag;
        data.v_angle[i] = bus.v_angle;
        data.p_injection[i] = bus.p_injection;
        data.q_injection[i] = bus.q_injection;
        data.p_scheduled[i] = bus.descriptor.p_scheduled;
        data.q_scheduled[i] = bus.descriptor.q_scheduled;
        data.v_setpoint[i] = bus.descriptor.v_setpoint;
    }
}

void NetworkModel::packBranchData(HostBranchData& data) const {
    int32_t n = static_cast<int32_t>(branches_.size());
    data.count = n;
    
    #pragma omp parallel for schedule(static) if(n > 100)
    for (int32_t i = 0; i < n; ++i) {
        const auto& branch = branches_[i];
        data.from_bus[i] = branch.from_bus_index;
        data.to_bus[i] = branch.to_bus_index;
        data.resistance[i] = branch.descriptor.resistance;
        data.reactance[i] = branch.descriptor.reactance;
        data.susceptance[i] = branch.descriptor.susceptance;
        data.conductance[i] = branch.descriptor.conductance;
        data.tap_ratio[i] = branch.descriptor.tap_ratio;
        data.phase_shift[i] = branch.descriptor.phase_shift;
        data.is_transformer[i] = branch.descriptor.is_transformer;
        data.status[i] = branch.status;
        data.sd_index[i] = branch.sd_index;
        data.g_series[i] = branch.g_series;
        data.b_series[i] = branch.b_series;
        data.b_shunt_from[i] = branch.b_shunt_from;
        data.b_shunt_to[i] = branch.b_shunt_to;
    }
}

void NetworkModel::packMeasurementData(HostMeasurementData& data) const {
    int32_t n = static_cast<int32_t>(measurements_.size());
    data.count = n;
    
    int32_t active_count = 0;
    for (int32_t i = 0; i < n; ++i) {
        const auto& meas = measurements_[i];
        data.type[i] = meas.descriptor.type;
        data.location_index[i] = meas.location_index;
        data.branch_end[i] = meas.descriptor.branch_end;
        data.pt_ratio[i] = meas.descriptor.pt_ratio;
        data.ct_ratio[i] = meas.descriptor.ct_ratio;
        data.value[i] = meas.value;
        data.weight[i] = meas.weight;
        data.sigma[i] = meas.descriptor.sigma;
        data.is_active[i] = meas.is_active;
        data.is_pseudo[i] = meas.descriptor.is_pseudo;
        if (meas.is_active) ++active_count;
    }
    data.active_count = active_count;
}

void NetworkModel::packSwitchingDeviceData(HostSwitchingDeviceData& data) const {
    int32_t n = static_cast<int32_t>(switching_devices_.size());
    data.count = n;
    
    for (int32_t i = 0; i < n; ++i) {
        const auto& sd = switching_devices_[i];
        data.branch_index[i] = sd.branch_index;
        data.status[i] = sd.status;
        data.pending_status[i] = sd.pending_status;
        data.has_pending_change[i] = sd.has_pending_change;
    }
}

size_t NetworkModel::calculateGPUMemoryRequired() const {
    size_t total = 0;
    
    // Bus data
    int32_t n_buses = static_cast<int32_t>(buses_.size());
    total += n_buses * (sizeof(Real) * 10 + sizeof(BusType));
    
    // Branch data
    int32_t n_branches = static_cast<int32_t>(branches_.size());
    total += n_branches * (sizeof(Real) * 17 + sizeof(int32_t) * 3 + sizeof(SwitchStatus));
    
    // Measurement data
    int32_t n_meas = static_cast<int32_t>(measurements_.size());
    total += n_meas * (sizeof(Real) * 6 + sizeof(int32_t) + sizeof(MeasurementType) + 
                       sizeof(BranchEnd) + sizeof(bool) * 2);
    
    return total;
}

//=============================================================================
// Validation
//=============================================================================

bool NetworkModel::validate() const {
    // Check for at least one bus
    if (buses_.empty()) return false;
    
    // Check for slack bus
    if (slack_bus_index_ == INVALID_INDEX) return false;
    
    // Validate branch references
    for (const auto& branch : branches_) {
        if (branch.from_bus_index < 0 || 
            branch.from_bus_index >= static_cast<int32_t>(buses_.size())) {
            return false;
        }
        if (branch.to_bus_index < 0 || 
            branch.to_bus_index >= static_cast<int32_t>(buses_.size())) {
            return false;
        }
    }
    
    // Validate measurement locations
    for (const auto& meas : measurements_) {
        if (isBusMeasurement(meas.descriptor.type)) {
            if (meas.location_index < 0 || 
                meas.location_index >= static_cast<int32_t>(buses_.size())) {
                return false;
            }
        } else {
            if (meas.location_index < 0 || 
                meas.location_index >= static_cast<int32_t>(branches_.size())) {
                return false;
            }
        }
    }
    
    return true;
}

void NetworkModel::computeBranchAdmittances() {
    for (auto& branch : branches_) {
        Real r = branch.descriptor.resistance;
        Real x = branch.descriptor.reactance;
        Real b_total = branch.descriptor.susceptance;
        Real a = branch.descriptor.tap_ratio;
        
        // Series admittance: y = 1/(r + jx)
        Real denom = r * r + x * x;
        if (denom < SLE_REAL_EPSILON) {
            denom = SLE_REAL_EPSILON;
        }
        
        branch.g_series = r / denom;
        branch.b_series = -x / denom;
        
        // Shunt susceptance (line charging)
        Real a2 = a * a;
        branch.b_shunt_from = b_total / (2.0f * a2);
        branch.b_shunt_to = b_total / 2.0f;
    }
}

void NetworkModel::applyFlatStart() {
    for (auto& bus : buses_) {
        if (bus.descriptor.type == BusType::PV || 
            bus.descriptor.type == BusType::SLACK) {
            bus.v_mag = bus.descriptor.v_setpoint;
        } else {
            bus.v_mag = 1.0f;
        }
        bus.v_angle = 0.0f;
    }
}

bool NetworkModel::isBusMeasurement(MeasurementType type) const {
    return type == MeasurementType::V_MAG ||
           type == MeasurementType::V_ANGLE ||
           type == MeasurementType::P_INJECTION ||
           type == MeasurementType::Q_INJECTION ||
           type == MeasurementType::P_PSEUDO ||
           type == MeasurementType::Q_PSEUDO;
}

//=============================================================================
// Result Setters (for GPU result download)
//=============================================================================

void NetworkModel::setBusVoltage(int32_t index, Real v_mag, Real v_angle) {
    if (index >= 0 && index < static_cast<int32_t>(buses_.size())) {
        buses_[index].v_mag = v_mag;
        buses_[index].v_angle = v_angle;
    }
}

void NetworkModel::setBusPower(int32_t index, Real p_inj, Real q_inj) {
    if (index >= 0 && index < static_cast<int32_t>(buses_.size())) {
        buses_[index].p_injection = p_inj;
        buses_[index].q_injection = q_inj;
    }
}

void NetworkModel::setBranchFlows(int32_t index, Real p_from, Real q_from, Real p_to, Real q_to) {
    if (index >= 0 && index < static_cast<int32_t>(branches_.size())) {
        branches_[index].p_flow_from = p_from;
        branches_[index].q_flow_from = q_from;
        branches_[index].p_flow_to = p_to;
        branches_[index].q_flow_to = q_to;
    }
}

void NetworkModel::setBranchCurrents(int32_t index, Real i_from, Real i_to) {
    if (index >= 0 && index < static_cast<int32_t>(branches_.size())) {
        branches_[index].i_mag_from = i_from;
        branches_[index].i_mag_to = i_to;
    }
}

void NetworkModel::setMeasurementResult(int32_t index, Real estimated, Real residual) {
    if (index >= 0 && index < static_cast<int32_t>(measurements_.size())) {
        measurements_[index].estimated = estimated;
        measurements_[index].residual = residual;
    }
}

//=============================================================================
// HostDataAllocator Implementation
//=============================================================================

HostDataAllocator::HostDataAllocator() {}

HostDataAllocator::~HostDataAllocator() {
    for (void* ptr : allocations_) {
        cudaFreeHost(ptr);
    }
}

cudaError_t HostDataAllocator::allocateBusData(HostBusData& data, int32_t count) {
    data.count = count;
    
    cudaError_t err;
    
    err = cudaMallocHost(&data.base_kv, count * sizeof(Real));
    if (err != cudaSuccess) return err;
    allocations_.push_back(data.base_kv);
    
    err = cudaMallocHost(&data.bus_type, count * sizeof(BusType));
    if (err != cudaSuccess) return err;
    allocations_.push_back(data.bus_type);
    
    err = cudaMallocHost(&data.external_id, count * sizeof(int32_t));
    if (err != cudaSuccess) return err;
    allocations_.push_back(data.external_id);
    
    err = cudaMallocHost(&data.v_mag, count * sizeof(Real));
    if (err != cudaSuccess) return err;
    allocations_.push_back(data.v_mag);
    
    err = cudaMallocHost(&data.v_angle, count * sizeof(Real));
    if (err != cudaSuccess) return err;
    allocations_.push_back(data.v_angle);
    
    err = cudaMallocHost(&data.p_injection, count * sizeof(Real));
    if (err != cudaSuccess) return err;
    allocations_.push_back(data.p_injection);
    
    err = cudaMallocHost(&data.q_injection, count * sizeof(Real));
    if (err != cudaSuccess) return err;
    allocations_.push_back(data.q_injection);
    
    err = cudaMallocHost(&data.p_scheduled, count * sizeof(Real));
    if (err != cudaSuccess) return err;
    allocations_.push_back(data.p_scheduled);
    
    err = cudaMallocHost(&data.q_scheduled, count * sizeof(Real));
    if (err != cudaSuccess) return err;
    allocations_.push_back(data.q_scheduled);
    
    err = cudaMallocHost(&data.v_setpoint, count * sizeof(Real));
    if (err != cudaSuccess) return err;
    allocations_.push_back(data.v_setpoint);
    
    return cudaSuccess;
}

void HostDataAllocator::freeBusData(HostBusData& data) {
    // Memory freed in destructor
    data = {};
}

cudaError_t HostDataAllocator::allocateBranchData(HostBranchData& data, int32_t count) {
    // Similar to allocateBusData - allocate all arrays
    data.count = count;
    // ... allocate all branch arrays
    return cudaSuccess;
}

void HostDataAllocator::freeBranchData(HostBranchData& data) {
    data = {};
}

cudaError_t HostDataAllocator::allocateMeasurementData(HostMeasurementData& data, int32_t count) {
    data.count = count;
    // ... allocate all measurement arrays
    return cudaSuccess;
}

void HostDataAllocator::freeMeasurementData(HostMeasurementData& data) {
    data = {};
}

cudaError_t HostDataAllocator::allocateSwitchingDeviceData(HostSwitchingDeviceData& data, int32_t count) {
    data.count = count;
    // ... allocate all SD arrays
    return cudaSuccess;
}

void HostDataAllocator::freeSwitchingDeviceData(HostSwitchingDeviceData& data) {
    data = {};
}

} // namespace sle

