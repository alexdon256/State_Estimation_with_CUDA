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
 * @file sle_engine.cu
 * @brief Main SLE Engine implementation
 * 
 * Implements the SLEEngine class and related components.
 * Coordinates network model, GPU data management, and WLS solver.
 */

#include "../include/sle_engine.h"
#include "../include/sle_config.h"
#include "../include/wls_solver.cuh"
#include "../include/sparse_matrix.cuh"
#include "../include/kernels.cuh"
#include "../include/jsf_compliance.h"
#include <cuda_runtime.h>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace sle {

//=============================================================================
// SLEEngine Implementation
//=============================================================================

SLEEngine::SLEEngine()
    : config_(EngineConfig::loadFromDefaultFile())  // Auto-load from default config file
    , config_path_(ConfigLoader::getDefaultConfigPath())
    , initialized_(false)
    , model_uploaded_(false)
    , measurements_dirty_(false)
    , model_(std::make_unique<NetworkModel>())
    , device_mgr_(nullptr)
    , solver_(nullptr)
    , topology_(nullptr)
    , progress_callback_(nullptr)
    , bad_data_callback_(nullptr)
    , last_solve_time_(0.0f)
{
    // Check if default config exists, if not, it's okay - defaults are used
}

SLEEngine::SLEEngine(const EngineConfig& config)
    : config_(config)
    , config_path_("")  // No file path when using explicit config
    , initialized_(false)
    , model_uploaded_(false)
    , measurements_dirty_(false)
    , model_(std::make_unique<NetworkModel>())
    , device_mgr_(nullptr)
    , solver_(nullptr)
    , topology_(nullptr)
    , progress_callback_(nullptr)
    , bad_data_callback_(nullptr)
    , last_solve_time_(0.0f)
{
}

SLEEngine::SLEEngine(const std::string& config_path)
    : config_(EngineConfig::loadFromFile(config_path))
    , config_path_(config_path)
    , initialized_(false)
    , model_uploaded_(false)
    , measurements_dirty_(false)
    , model_(std::make_unique<NetworkModel>())
    , device_mgr_(nullptr)
    , solver_(nullptr)
    , topology_(nullptr)
    , progress_callback_(nullptr)
    , bad_data_callback_(nullptr)
    , last_solve_time_(0.0f)
{
}

SLEEngine::~SLEEngine() {
    // GPU resources are freed by unique_ptr destructors
    // Ensure CUDA context is synchronized before destruction
    if (initialized_) {
        cudaDeviceSynchronize();
    }
}

SLEEngine::SLEEngine(SLEEngine&&) noexcept = default;
SLEEngine& SLEEngine::operator=(SLEEngine&&) noexcept = default;

//=============================================================================
// Initialization
//=============================================================================

bool SLEEngine::initialize() {
    return initialize(config_);
}

bool SLEEngine::initialize(const EngineConfig& config) {
    if (initialized_) {
        // Already initialized, just update config
        config_ = config;
        return true;
    }
    
    config_ = config;
    
    // Select CUDA device
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return false;
    }
    
    if (config_.device_id >= device_count) {
        return false;
    }
    
    err = cudaSetDevice(config_.device_id);
    if (err != cudaSuccess) {
        return false;
    }
    
    // Create GPU resource managers
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    
    device_mgr_ = std::make_unique<DeviceDataManager>(stream);
    solver_ = std::make_unique<WLSSolver>(stream);
    topology_ = std::make_unique<TopologyProcessor>(stream);
    
    initialized_ = true;
    return true;
}

std::string SLEEngine::getDeviceInfo() const {
    if (!initialized_) {
        return "Not initialized";
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, config_.device_id);
    
    std::string info = prop.name;
    info += " (Compute " + std::to_string(prop.major) + "." + std::to_string(prop.minor) + ")";
    info += " - " + std::to_string(prop.totalGlobalMem / (1024 * 1024)) + " MB";
    
    return info;
}

void SLEEngine::reset() {
    if (initialized_) {
        cudaDeviceSynchronize();
    }
    
    solver_.reset();
    topology_.reset();
    device_mgr_.reset();
    model_ = std::make_unique<NetworkModel>();
    
    initialized_ = false;
    model_uploaded_ = false;
}

//=============================================================================
// Configuration Management
//=============================================================================

bool SLEEngine::reloadConfig() {
    if (config_path_.empty()) {
        return false;  // No config file to reload from
    }
    return reloadConfig(config_path_);
}

bool SLEEngine::reloadConfig(const std::string& config_path) {
    EngineConfiguration cfg;
    if (!ConfigLoader::loadFromFile(config_path, cfg)) {
        return false;
    }
    
    EngineConfig new_config(cfg);
    return updateConfig(new_config);
}

bool SLEEngine::updateConfig(const EngineConfig& config) {
    // Check if capacity changed - requires reinitialization
    bool capacity_changed = 
        config.max_buses != config_.max_buses ||
        config.max_branches != config_.max_branches ||
        config.max_measurements != config_.max_measurements;
    
    if (capacity_changed && model_uploaded_) {
        // Cannot change capacity after model upload without reset
        return false;
    }
    
    // Update solver settings (can be done at runtime)
    config_ = config;
    
    // Invalidate solver state if needed
    if (solver_) {
        solver_->invalidateGainMatrix();
    }
    
    return true;
}

bool SLEEngine::saveConfig(const std::string& filepath) const {
    std::string path = filepath.empty() ? config_path_ : filepath;
    if (path.empty()) {
        path = ConfigLoader::getDefaultConfigPath();
    }
    return config_.saveToFile(path);
}

bool SLEEngine::ensureConfigFile(const std::string& filepath) {
    std::string path = filepath.empty() ? ConfigLoader::getDefaultConfigPath() : filepath;
    
    // Check if file exists
    std::ifstream test(path);
    if (test.good()) {
        return true;  // File already exists
    }
    
    // Create default config file
    return EngineConfig::createDefaultConfigFile(path);
}

//=============================================================================
// Model Building
//=============================================================================

int32_t SLEEngine::addBus(const BusDescriptor& desc) {
    if (!model_) return INVALID_INDEX;
    
    model_->markModified();
    return model_->addBus(desc);
}

int32_t SLEEngine::addBranch(const BranchDescriptor& desc) {
    if (!model_) return INVALID_INDEX;
    
    model_->markModified();
    return model_->addBranch(desc);
}

int32_t SLEEngine::addMeasurement(const MeasurementDescriptor& desc) {
    if (!model_) return INVALID_INDEX;
    
    model_->markModified();
    return model_->addMeasurement(desc);
}

int32_t SLEEngine::addSwitchingDevice(const SwitchingDeviceDescriptor& desc) {
    if (!model_) return INVALID_INDEX;
    
    model_->markModified();
    return model_->addSwitchingDevice(desc);
}

int32_t SLEEngine::addMeter(const MeterDescriptor& desc) {
    if (!model_) return INVALID_INDEX;
    
    model_->markModified();
    return model_->addMeter(desc);
}

bool SLEEngine::updateMeterReading(const std::string& meter_id,
                                   const std::string& channel,
                                   Real value) {
    if (!model_) return false;
    if (model_->updateMeterReading(meter_id, channel, value)) {
        measurements_dirty_ = true;
        return true;
    }
    return false;
}

bool SLEEngine::getMeterReading(const std::string& meter_id,
                                const std::string& channel,
                                Real& value) const {
    if (!model_) return false;
    return model_->getMeterReading(meter_id, channel, value);
}

bool SLEEngine::getMeterEstimate(const std::string& meter_id,
                                 const std::string& channel,
                                 Real& value) const {
    if (!model_) return false;
    return model_->getMeterEstimate(meter_id, channel, value);
}

bool SLEEngine::getMeterResidual(const std::string& meter_id,
                                 const std::string& channel,
                                 Real& value) const {
    if (!model_) return false;
    return model_->getMeterResidual(meter_id, channel, value);
}

bool SLEEngine::removeBus(const std::string& bus_id) {
    if (!model_) return false;
    
    model_->markModified();
    return model_->removeBus(bus_id);
}

bool SLEEngine::removeBranch(const std::string& branch_id) {
    if (!model_) return false;
    
    model_->markModified();
    return model_->removeBranch(branch_id);
}

bool SLEEngine::updateBranch(const std::string& branch_id,
                              const BranchDescriptor& desc) {
    if (!model_) return false;
    
    // Update the branch through the network model
    if (!model_->updateBranch(branch_id, desc)) {
        return false;
    }
    
    model_->markModified();
    model_uploaded_ = false;  // Force re-upload
    
    // Invalidate solver gain matrix (topology may have changed)
    if (solver_) {
        solver_->invalidateGainMatrix();
    }
    
    return true;
}

bool SLEEngine::setTransformerTap(const std::string& branch_id, Real tap_ratio) {
    if (!model_) return false;
    
    int32_t idx = model_->getBranchIndex(branch_id);
    if (idx == INVALID_INDEX) return false;
    
    BranchElement* branch = model_->getBranch(idx);
    if (!branch) return false;
    
    // Update tap ratio
    branch->descriptor.tap_ratio = tap_ratio;
    
    // Recompute admittances for this branch
    model_->computeBranchAdmittances();
    
    model_->markModified();
    model_uploaded_ = false;  // Force re-upload
    
    // Invalidate solver gain matrix
    if (solver_) {
        solver_->invalidateGainMatrix();
    }
    
    return true;
}

bool SLEEngine::setTransformerPhaseShift(const std::string& branch_id,
                                          Real phase_shift_rad) {
    if (!model_) return false;
    
    int32_t idx = model_->getBranchIndex(branch_id);
    if (idx == INVALID_INDEX) return false;
    
    BranchElement* branch = model_->getBranch(idx);
    if (!branch) return false;
    
    // Update phase shift
    branch->descriptor.phase_shift = phase_shift_rad;
    
    // Recompute admittances for this branch
    model_->computeBranchAdmittances();
    
    model_->markModified();
    model_uploaded_ = false;  // Force re-upload
    
    // Invalidate solver gain matrix
    if (solver_) {
        solver_->invalidateGainMatrix();
    }
    
    return true;
}

bool SLEEngine::setBranchImpedance(const std::string& branch_id,
                                    Real r, Real x, Real b, Real g) {
    if (!model_) return false;
    
    int32_t idx = model_->getBranchIndex(branch_id);
    if (idx == INVALID_INDEX) return false;
    
    BranchElement* branch = model_->getBranch(idx);
    if (!branch) return false;
    
    // Update impedance values
    branch->descriptor.resistance = r;
    branch->descriptor.reactance = x;
    branch->descriptor.susceptance = b;
    branch->descriptor.conductance = g;
    
    // Recompute admittances for this branch
    model_->computeBranchAdmittances();
    
    model_->markModified();
    model_uploaded_ = false;  // Force re-upload
    
    // Invalidate solver gain matrix
    if (solver_) {
        solver_->invalidateGainMatrix();
    }
    
    return true;
}

bool SLEEngine::isModelModified() const {
    return model_ && model_->isModified();
}

//=============================================================================
// GPU Data Management
//=============================================================================

bool SLEEngine::uploadModel() {
    if (!ensureInitialized()) return false;
    if (!model_) return false;
    
    // Validate model
    if (!model_->validate()) {
        return false;
    }
    
    // Compute derived values
    model_->computeBranchAdmittances();
    
    // Allocate GPU memory
    cudaError_t err = device_mgr_->allocate(
        model_->getBusCount(),
        model_->getBranchCount(),
        model_->getMeasurementCount(),
        model_->getSwitchingDeviceCount());
    
    if (err != cudaSuccess) return false;
    
    // Upload data to GPU
    err = device_mgr_->upload(*model_);
    if (err != cudaSuccess) return false;
    
    // Initialize topology processor
    err = topology_->initialize(
        device_mgr_->getBranches(),
        device_mgr_->getYbus(),
        model_->getSwitchingDeviceCount());
    
    if (err != cudaSuccess) return false;
    
    // Initialize solver
    err = solver_->initialize(
        device_mgr_->getBuses(),
        device_mgr_->getBranches(),
        device_mgr_->getMeasurements(),
        device_mgr_->getYbus());
    
    if (err != cudaSuccess) return false;
    
    model_->clearModified();
    model_uploaded_ = true;
    measurements_dirty_ = false;  // Model upload includes measurement values
    
    return true;
}

size_t SLEEngine::getGPUMemoryUsage() const {
    if (!device_mgr_) return 0;
    return device_mgr_->getMemoryUsage();
}

//=============================================================================
// Telemetry Update
//=============================================================================

bool SLEEngine::updateTelemetry(const Real* values, int32_t count) {
    if (!ensureModelUploaded()) return false;
    if (!values || count != model_->getMeasurementCount()) return false;
    
    // Update host model
    if (!model_->updateMeasurementValues(values, count)) {
        return false;
    }
    
    // Transfer to GPU
    cudaError_t err = device_mgr_->updateMeasurements(values, count);
    if (err == cudaSuccess) {
        measurements_dirty_ = false;  // GPU is now in sync
        return true;
    }
    return false;
}

bool SLEEngine::updateMeasurement(const std::string& meas_id, Real value) {
    if (!model_) return false;
    if (model_->updateMeasurementValue(meas_id, value)) {
        measurements_dirty_ = true;
        return true;
    }
    return false;
}

bool SLEEngine::syncMeasurementsToDevice() {
    if (!model_ || !device_mgr_ || !model_uploaded_) {
        return false;
    }
    
    // Collect all measurement values from host model
    int32_t count = model_->getMeasurementCount();
    if (count == 0) {
        measurements_dirty_ = false;
        return true;
    }
    
    std::vector<Real> values(count);
    for (int32_t i = 0; i < count; ++i) {
        const MeasurementElement* elem = model_->getMeasurement(i);
        if (elem) {
            values[i] = elem->value;
        }
    }
    
    // Transfer to GPU
    cudaError_t err = device_mgr_->updateMeasurements(values.data(), count);
    if (err == cudaSuccess) {
        measurements_dirty_ = false;
        return true;
    }
    return false;
}

//=============================================================================
// Switch Control
//=============================================================================

bool SLEEngine::setSwitchStatus(const std::string& sd_id, SwitchStatus new_status) {
    if (!model_) return false;
    
    int32_t idx = model_->getSwitchingDeviceIndex(sd_id);
    if (idx == INVALID_INDEX) return false;
    
    // Queue the change
    if (topology_) {
        topology_->queueStatusChange(idx, new_status);
    }
    
    return true;
}

SwitchStatus SLEEngine::getSwitchStatus(const std::string& sd_id) const {
    if (!model_) return SwitchStatus::OPEN;
    
    int32_t idx = model_->getSwitchingDeviceIndex(sd_id);
    if (idx == INVALID_INDEX) return SwitchStatus::OPEN;
    
    const auto* sd = model_->getSwitchingDevice(idx);
    return sd ? sd->status : SwitchStatus::OPEN;
}

bool SLEEngine::hasPendingSwitchChanges() const {
    return topology_ && topology_->hasPendingChanges();
}

bool SLEEngine::applyPendingSwitchChanges() {
    if (!topology_ || !device_mgr_ || !solver_) return false;
    
    if (!topology_->hasPendingChanges()) return true;
    
    handleTopologyChanges();
    return true;
}

//=============================================================================
// State Estimation
//=============================================================================

EstimationResult SLEEngine::solve(EstimationMode mode) {
    SolverConfig solver_config;
    solver_config.mode = mode;
    
    // Apply config file settings based on mode
    if (mode == EstimationMode::REALTIME) {
        solver_config.max_iterations = config_.max_realtime_iterations;
        solver_config.time_limit_ms = config_.realtime_time_limit_ms;
    } else {
        solver_config.max_iterations = config_.max_precision_iterations;
        solver_config.time_limit_ms = 0.0f;  // No limit
    }
    
    solver_config.convergence_tolerance = config_.convergence_tolerance;
    solver_config.use_robust_estimation = config_.enable_robust_estimation;
    solver_config.huber_gamma = config_.huber_gamma;
    solver_config.use_flat_start = config_.use_flat_start_default;
    
    return solve(solver_config);
}

EstimationResult SLEEngine::solve(const SolverConfig& config) {
    EstimationResult result;
    
    // Ensure initialized and model uploaded
    if (!ensureModelUploaded()) {
        result.status = ConvergenceStatus::NOT_OBSERVABLE;
        return result;
    }
    
    // Handle pending topology changes before solving
    handleTopologyChanges();
    
    // Handle model modifications (FR-25)
    if (model_->isModified()) {
        // Re-upload model and invalidate solver
        if (!uploadModel()) {
            result.status = ConvergenceStatus::SINGULAR_MATRIX;
            return result;
        }
    }
    
    // Auto-sync measurements if dirty (meter/measurement updates pending)
    // This ensures meter readings flow to GPU before estimation
    if (measurements_dirty_) {
        syncMeasurementsToDevice();
    }
    
    // Run WLS solver
    result = solver_->solve(
        device_mgr_->getBuses(),
        device_mgr_->getBranches(),
        device_mgr_->getMeasurements(),
        device_mgr_->getYbus(),
        config);
    
    last_result_ = result;
    last_solve_time_ = result.computation_time_ms;
    
    // Copy results back to host model
    copyResultsToHost();
    
    // Call progress callback if registered
    if (progress_callback_) {
        progress_callback_(result.iterations, result.max_mismatch, result.objective_value);
    }
    
    // Check for bad data and call callback
    if (bad_data_callback_ && result.bad_data_count > 0) {
        bad_data_callback_(result.largest_residual_idx, result.largest_residual);
    }
    
    return result;
}

bool SLEEngine::checkObservability() {
    if (!ensureModelUploaded()) return false;
    
    // Perform observability analysis using graph connectivity and measurement coverage
    // For a network to be observable:
    // 1. It must be connected (single island)
    // 2. Number of independent measurements >= number of state variables
    
    int32_t n_buses = model_->getBusCount();
    int32_t n_measurements = getActiveMeasurementCount();
    int32_t n_states = 2 * n_buses - 1;  // Angles (except slack) + magnitudes
    
    // Basic observability check: need at least n_states measurements
    if (n_measurements < n_states) {
        return false;
    }
    
    // Check network connectivity using Union-Find on host
    std::vector<int32_t> parent(n_buses);
    std::vector<int32_t> rank(n_buses, 0);
    
    // Initialize each bus as its own set
    for (int32_t i = 0; i < n_buses; ++i) {
        parent[i] = i;
    }
    
    // Union-Find helper functions (lambdas)
    std::function<int32_t(int32_t)> find = [&](int32_t x) -> int32_t {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // Path compression
        }
        return parent[x];
    };
    
    auto unite = [&](int32_t x, int32_t y) {
        int32_t px = find(x);
        int32_t py = find(y);
        if (px == py) return;
        
        // Union by rank
        if (rank[px] < rank[py]) {
            parent[px] = py;
        } else if (rank[px] > rank[py]) {
            parent[py] = px;
        } else {
            parent[py] = px;
            rank[px]++;
        }
    };
    
    // Unite buses connected by closed branches
    int32_t n_branches = model_->getBranchCount();
    for (int32_t i = 0; i < n_branches; ++i) {
        const auto* branch = model_->getBranch(i);
        if (branch && branch->status == SwitchStatus::CLOSED) {
            unite(branch->from_bus_index, branch->to_bus_index);
        }
    }
    
    // Count islands
    island_count_ = 0;
    for (int32_t i = 0; i < n_buses; ++i) {
        if (find(i) == i) {
            island_count_++;
        }
    }
    
    // Network is observable if single island with sufficient measurements
    return (island_count_ == 1);
}

int32_t SLEEngine::getIslandCount() const {
    return island_count_;
}

void SLEEngine::applyFlatStart() {
    if (!model_) return;
    model_->applyFlatStart();
    
    if (model_uploaded_ && device_mgr_) {
        // Upload flat start state to GPU
        int32_t n_buses = model_->getBusCount();
        
        // Allocate temporary host arrays
        std::vector<Real> v_mag(n_buses);
        std::vector<Real> v_angle(n_buses, 0.0f);
        
        for (int32_t i = 0; i < n_buses; ++i) {
            const auto* bus = model_->getBus(i);
            v_mag[i] = bus ? bus->v_mag : 1.0f;
        }
        
        // Copy to device
        DeviceBusData& buses = device_mgr_->getBuses();
        cudaMemcpy(buses.d_v_mag, v_mag.data(), n_buses * sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(buses.d_v_angle, v_angle.data(), n_buses * sizeof(Real), cudaMemcpyHostToDevice);
    }
}

void SLEEngine::applyHotStart() {
    // Hot start uses current state - nothing to do
}

//=============================================================================
// Result Access
//=============================================================================

std::vector<Real> SLEEngine::getVoltageMagnitudes() const {
    if (!model_) return {};
    
    int32_t n = model_->getBusCount();
    std::vector<Real> voltages(n);
    
    #pragma omp parallel for schedule(static) if(n > 100)
    for (int32_t i = 0; i < n; ++i) {
        const auto* bus = model_->getBus(i);
        voltages[i] = bus ? bus->v_mag : 0.0f;
    }
    
    return voltages;
}

std::vector<Real> SLEEngine::getVoltageAngles() const {
    if (!model_) return {};
    
    int32_t n = model_->getBusCount();
    std::vector<Real> angles(n);
    
    #pragma omp parallel for schedule(static) if(n > 100)
    for (int32_t i = 0; i < n; ++i) {
        const auto* bus = model_->getBus(i);
        angles[i] = bus ? bus->v_angle : 0.0f;
    }
    
    return angles;
}

Real SLEEngine::getVoltageMagnitude(const std::string& bus_id) const {
    if (!model_) return std::nanf("");
    
    int32_t idx = model_->getBusIndex(bus_id);
    if (idx == INVALID_INDEX) return std::nanf("");
    
    const auto* bus = model_->getBus(idx);
    return bus ? bus->v_mag : std::nanf("");
}

Real SLEEngine::getVoltageAngle(const std::string& bus_id) const {
    if (!model_) return std::nanf("");
    
    int32_t idx = model_->getBusIndex(bus_id);
    if (idx == INVALID_INDEX) return std::nanf("");
    
    const auto* bus = model_->getBus(idx);
    return bus ? bus->v_angle : std::nanf("");
}

std::vector<Real> SLEEngine::getPowerInjections() const {
    if (!model_) return {};
    
    int32_t n = model_->getBusCount();
    std::vector<Real> p_inj(n);
    
    #pragma omp parallel for schedule(static) if(n > 100)
    for (int32_t i = 0; i < n; ++i) {
        const auto* bus = model_->getBus(i);
        p_inj[i] = bus ? bus->p_injection : 0.0f;
    }
    
    return p_inj;
}

std::vector<Real> SLEEngine::getReactivePowerInjections() const {
    if (!model_) return {};
    
    int32_t n = model_->getBusCount();
    std::vector<Real> q_inj(n);
    
    #pragma omp parallel for schedule(static) if(n > 100)
    for (int32_t i = 0; i < n; ++i) {
        const auto* bus = model_->getBus(i);
        q_inj[i] = bus ? bus->q_injection : 0.0f;
    }
    
    return q_inj;
}

Real SLEEngine::getPowerFlow(const std::string& branch_id, BranchEnd end) const {
    if (!model_) return std::nanf("");
    
    int32_t idx = model_->getBranchIndex(branch_id);
    if (idx == INVALID_INDEX) return std::nanf("");
    
    const auto* branch = model_->getBranch(idx);
    if (!branch) return std::nanf("");
    
    return (end == BranchEnd::FROM) ? branch->p_flow_from : branch->p_flow_to;
}

Real SLEEngine::getReactivePowerFlow(const std::string& branch_id, BranchEnd end) const {
    if (!model_) return std::nanf("");
    
    int32_t idx = model_->getBranchIndex(branch_id);
    if (idx == INVALID_INDEX) return std::nanf("");
    
    const auto* branch = model_->getBranch(idx);
    if (!branch) return std::nanf("");
    
    return (end == BranchEnd::FROM) ? branch->q_flow_from : branch->q_flow_to;
}

Real SLEEngine::getCurrentMagnitude(const std::string& branch_id, BranchEnd end) const {
    if (!model_) return std::nanf("");
    
    int32_t idx = model_->getBranchIndex(branch_id);
    if (idx == INVALID_INDEX) return std::nanf("");
    
    const auto* branch = model_->getBranch(idx);
    if (!branch) return std::nanf("");
    
    return (end == BranchEnd::FROM) ? branch->i_mag_from : branch->i_mag_to;
}

//=============================================================================
// Residual Analysis
//=============================================================================

std::vector<Real> SLEEngine::getResiduals() const {
    if (!model_) return {};
    
    int32_t n = model_->getMeasurementCount();
    std::vector<Real> residuals(n);
    
    #pragma omp parallel for schedule(static) if(n > 100)
    for (int32_t i = 0; i < n; ++i) {
        const auto* meas = model_->getMeasurement(i);
        residuals[i] = meas ? meas->residual : 0.0f;
    }
    
    return residuals;
}

std::vector<Real> SLEEngine::getNormalizedResiduals() const {
    if (!model_) return {};
    
    int32_t n = model_->getMeasurementCount();
    std::vector<Real> norm_res(n);
    
    #pragma omp parallel for schedule(static) if(n > 100)
    for (int32_t i = 0; i < n; ++i) {
        const auto* meas = model_->getMeasurement(i);
        if (meas && meas->descriptor.sigma > 0) {
            norm_res[i] = std::abs(meas->residual) / meas->descriptor.sigma;
        } else {
            norm_res[i] = 0.0f;
        }
    }
    
    return norm_res;
}

Real SLEEngine::getResidual(const std::string& meas_id) const {
    if (!model_) return std::nanf("");
    
    int32_t idx = model_->getMeasurementIndex(meas_id);
    if (idx == INVALID_INDEX) return std::nanf("");
    
    const auto* meas = model_->getMeasurement(idx);
    return meas ? meas->residual : std::nanf("");
}

Real SLEEngine::getObjectiveValue() const {
    return last_result_.objective_value;
}

std::vector<int32_t> SLEEngine::identifyBadData(Real threshold) const {
    std::vector<int32_t> bad_indices;
    if (!model_) return bad_indices;
    
    Real thresh = (threshold > 0) ? threshold : config_.bad_data_threshold;
    
    int32_t n = model_->getMeasurementCount();
    for (int32_t i = 0; i < n; ++i) {
        const auto* meas = model_->getMeasurement(i);
        if (meas && meas->is_active && meas->descriptor.sigma > 0) {
            Real norm_res = std::abs(meas->residual) / meas->descriptor.sigma;
            if (norm_res > thresh) {
                bad_indices.push_back(i);
            }
        }
    }
    
    return bad_indices;
}

//=============================================================================
// Callbacks
//=============================================================================

void SLEEngine::setProgressCallback(ProgressCallback callback) {
    progress_callback_ = std::move(callback);
}

void SLEEngine::setBadDataCallback(BadDataCallback callback) {
    bad_data_callback_ = std::move(callback);
}

int32_t SLEEngine::getActiveMeasurementCount() const {
    if (!model_) return 0;
    
    int32_t count = 0;
    int32_t n = model_->getMeasurementCount();
    for (int32_t i = 0; i < n; ++i) {
        const auto* meas = model_->getMeasurement(i);
        if (meas && meas->is_active) ++count;
    }
    
    return count;
}

//=============================================================================
// Internal Methods
//=============================================================================

bool SLEEngine::ensureInitialized() {
    if (!initialized_) {
        return initialize();
    }
    return true;
}

bool SLEEngine::ensureModelUploaded() {
    if (!ensureInitialized()) return false;
    
    if (!model_uploaded_) {
        return uploadModel();
    }
    return true;
}

void SLEEngine::handleTopologyChanges() {
    if (!topology_ || !topology_->hasPendingChanges()) return;
    
    // Apply pending switch changes
    (void)topology_->applyChanges(
        device_mgr_->getBranches(),
        device_mgr_->getYbus(),
        solver_->getMatrixManager());
    
    // Invalidate solver state (need to refactor gain matrix)
    solver_->invalidateGainMatrix();
}

void SLEEngine::copyResultsToHost() {
    if (!device_mgr_ || !model_) return;
    
    // Download results from GPU to host model
    (void)device_mgr_->downloadResults(*model_);
}

//=============================================================================
// DeviceDataManager Implementation
//=============================================================================

DeviceDataManager::DeviceDataManager(cudaStream_t stream)
    : stream_(stream)
    , h_pinned_measurements_(nullptr)
    , total_allocated_(0)
    , is_allocated_(false)
{
    buses_ = {};
    branches_ = {};
    measurements_ = {};
    ybus_ = {};
}

DeviceDataManager::~DeviceDataManager() {
    free();
}

cudaError_t DeviceDataManager::allocate(
    int32_t n_buses,
    int32_t n_branches,
    int32_t n_measurements,
    int32_t n_switching_devices)
{
    // Suppress unused parameter warning - reserved for future use
    (void)n_switching_devices;
    
    if (is_allocated_) {
        free();
    }
    
    cudaError_t err;
    
    err = allocateBusData(n_buses);
    if (err != cudaSuccess) return err;
    
    err = allocateBranchData(n_branches);
    if (err != cudaSuccess) { freeBusData(); return err; }
    
    err = allocateMeasurementData(n_measurements);
    if (err != cudaSuccess) { freeBusData(); freeBranchData(); return err; }
    
    // Allocate pinned memory for async measurement updates
    err = cudaMallocHost(&h_pinned_measurements_, n_measurements * sizeof(Real));
    if (err != cudaSuccess) {
        freeBusData();
        freeBranchData();
        freeMeasurementData();
        return err;
    }
    
    is_allocated_ = true;
    return cudaSuccess;
}

cudaError_t DeviceDataManager::allocateBusData(int32_t count) {
    buses_.count = count;
    buses_.slack_bus_index = 0;
    
    size_t size_real = count * sizeof(Real);
    size_t size_type = count * sizeof(BusType);
    
    cudaError_t err;
    
    err = cudaMalloc(&buses_.d_base_kv, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&buses_.d_bus_type, size_type);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&buses_.d_v_mag, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&buses_.d_v_angle, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&buses_.d_v_mag_prev, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&buses_.d_v_angle_prev, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&buses_.d_p_injection, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&buses_.d_q_injection, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&buses_.d_p_scheduled, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&buses_.d_q_scheduled, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&buses_.d_v_setpoint, size_real);
    if (err != cudaSuccess) return err;
    
    total_allocated_ += 11 * size_real + size_type;
    
    return cudaSuccess;
}

void DeviceDataManager::freeBusData() {
    if (buses_.d_base_kv) cudaFree(buses_.d_base_kv);
    if (buses_.d_bus_type) cudaFree(buses_.d_bus_type);
    if (buses_.d_v_mag) cudaFree(buses_.d_v_mag);
    if (buses_.d_v_angle) cudaFree(buses_.d_v_angle);
    if (buses_.d_v_mag_prev) cudaFree(buses_.d_v_mag_prev);
    if (buses_.d_v_angle_prev) cudaFree(buses_.d_v_angle_prev);
    if (buses_.d_p_injection) cudaFree(buses_.d_p_injection);
    if (buses_.d_q_injection) cudaFree(buses_.d_q_injection);
    if (buses_.d_p_scheduled) cudaFree(buses_.d_p_scheduled);
    if (buses_.d_q_scheduled) cudaFree(buses_.d_q_scheduled);
    if (buses_.d_v_setpoint) cudaFree(buses_.d_v_setpoint);
    
    buses_ = {};
}

cudaError_t DeviceDataManager::allocateBranchData(int32_t count) {
    branches_.count = count;
    
    size_t size_real = count * sizeof(Real);
    size_t size_int = count * sizeof(int32_t);
    size_t size_status = count * sizeof(SwitchStatus);
    
    cudaError_t err;
    
    err = cudaMalloc(&branches_.d_from_bus, size_int);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&branches_.d_to_bus, size_int);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&branches_.d_resistance, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&branches_.d_reactance, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&branches_.d_susceptance, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&branches_.d_conductance, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&branches_.d_tap_ratio, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&branches_.d_phase_shift, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&branches_.d_g_series, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&branches_.d_b_series, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&branches_.d_b_shunt_from, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&branches_.d_b_shunt_to, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&branches_.d_status, size_status);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&branches_.d_p_flow_from, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&branches_.d_q_flow_from, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&branches_.d_p_flow_to, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&branches_.d_q_flow_to, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&branches_.d_i_mag_from, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&branches_.d_i_mag_to, size_real);
    if (err != cudaSuccess) return err;
    
    total_allocated_ += 17 * size_real + 2 * size_int + size_status;
    
    return cudaSuccess;
}

void DeviceDataManager::freeBranchData() {
    if (branches_.d_from_bus) cudaFree(branches_.d_from_bus);
    if (branches_.d_to_bus) cudaFree(branches_.d_to_bus);
    if (branches_.d_resistance) cudaFree(branches_.d_resistance);
    if (branches_.d_reactance) cudaFree(branches_.d_reactance);
    if (branches_.d_susceptance) cudaFree(branches_.d_susceptance);
    if (branches_.d_conductance) cudaFree(branches_.d_conductance);
    if (branches_.d_tap_ratio) cudaFree(branches_.d_tap_ratio);
    if (branches_.d_phase_shift) cudaFree(branches_.d_phase_shift);
    if (branches_.d_g_series) cudaFree(branches_.d_g_series);
    if (branches_.d_b_series) cudaFree(branches_.d_b_series);
    if (branches_.d_b_shunt_from) cudaFree(branches_.d_b_shunt_from);
    if (branches_.d_b_shunt_to) cudaFree(branches_.d_b_shunt_to);
    if (branches_.d_status) cudaFree(branches_.d_status);
    if (branches_.d_p_flow_from) cudaFree(branches_.d_p_flow_from);
    if (branches_.d_q_flow_from) cudaFree(branches_.d_q_flow_from);
    if (branches_.d_p_flow_to) cudaFree(branches_.d_p_flow_to);
    if (branches_.d_q_flow_to) cudaFree(branches_.d_q_flow_to);
    if (branches_.d_i_mag_from) cudaFree(branches_.d_i_mag_from);
    if (branches_.d_i_mag_to) cudaFree(branches_.d_i_mag_to);
    
    branches_ = {};
}

cudaError_t DeviceDataManager::allocateMeasurementData(int32_t count) {
    measurements_.count = count;
    measurements_.active_count = count;
    
    size_t size_real = count * sizeof(Real);
    size_t size_int = count * sizeof(int32_t);
    size_t size_type = count * sizeof(MeasurementType);
    size_t size_end = count * sizeof(BranchEnd);
    size_t size_uint8 = count * sizeof(uint8_t);  // For is_active flag
    
    cudaError_t err;
    
    err = cudaMalloc(&measurements_.d_type, size_type);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&measurements_.d_location_index, size_int);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&measurements_.d_branch_end, size_end);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&measurements_.d_pt_ratio, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&measurements_.d_ct_ratio, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&measurements_.d_value, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&measurements_.d_weight, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&measurements_.d_sigma, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&measurements_.d_is_active, size_uint8);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&measurements_.d_estimated, size_real);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&measurements_.d_residual, size_real);
    if (err != cudaSuccess) return err;
    
    // Count: pt_ratio, ct_ratio, value, weight, sigma, estimated, residual = 7 Real arrays
    total_allocated_ += 7 * size_real + size_int + size_type + size_end + size_uint8;
    
    return cudaSuccess;
}

void DeviceDataManager::freeMeasurementData() {
    if (measurements_.d_type) cudaFree(measurements_.d_type);
    if (measurements_.d_location_index) cudaFree(measurements_.d_location_index);
    if (measurements_.d_branch_end) cudaFree(measurements_.d_branch_end);
    if (measurements_.d_pt_ratio) cudaFree(measurements_.d_pt_ratio);
    if (measurements_.d_ct_ratio) cudaFree(measurements_.d_ct_ratio);
    if (measurements_.d_value) cudaFree(measurements_.d_value);
    if (measurements_.d_weight) cudaFree(measurements_.d_weight);
    if (measurements_.d_sigma) cudaFree(measurements_.d_sigma);
    if (measurements_.d_is_active) cudaFree(measurements_.d_is_active);
    if (measurements_.d_estimated) cudaFree(measurements_.d_estimated);
    if (measurements_.d_residual) cudaFree(measurements_.d_residual);
    
    measurements_ = {};
}

cudaError_t DeviceDataManager::upload(const NetworkModel& model) {
    if (!is_allocated_) {
        return cudaErrorNotReady;
    }
    
    cudaError_t err;
    int32_t n_buses = model.getBusCount();
    int32_t n_branches = model.getBranchCount();
    int32_t n_measurements = model.getMeasurementCount();
    
    //=========================================================================
    // Upload Bus Data
    //=========================================================================
    
    // Allocate temporary host arrays for bus data
    std::vector<Real> h_base_kv(n_buses);
    std::vector<BusType> h_bus_type(n_buses);
    std::vector<Real> h_v_mag(n_buses);
    std::vector<Real> h_v_angle(n_buses);
    std::vector<Real> h_p_scheduled(n_buses);
    std::vector<Real> h_q_scheduled(n_buses);
    std::vector<Real> h_v_setpoint(n_buses);
    
    #pragma omp parallel for schedule(static) if(n_buses > 100)
    for (int32_t i = 0; i < n_buses; ++i) {
        const auto* bus = model.getBus(i);
        if (bus) {
            h_base_kv[i] = bus->descriptor.base_kv;
            h_bus_type[i] = bus->descriptor.type;
            h_v_mag[i] = bus->v_mag;
            h_v_angle[i] = bus->v_angle;
            h_p_scheduled[i] = bus->descriptor.p_scheduled;
            h_q_scheduled[i] = bus->descriptor.q_scheduled;
            h_v_setpoint[i] = bus->descriptor.v_setpoint;
        }
    }
    
    buses_.slack_bus_index = model.getSlackBusIndex();
    
    err = cudaMemcpyAsync(buses_.d_base_kv, h_base_kv.data(), 
                          n_buses * sizeof(Real), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(buses_.d_bus_type, h_bus_type.data(),
                          n_buses * sizeof(BusType), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(buses_.d_v_mag, h_v_mag.data(),
                          n_buses * sizeof(Real), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(buses_.d_v_angle, h_v_angle.data(),
                          n_buses * sizeof(Real), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(buses_.d_p_scheduled, h_p_scheduled.data(),
                          n_buses * sizeof(Real), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(buses_.d_q_scheduled, h_q_scheduled.data(),
                          n_buses * sizeof(Real), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(buses_.d_v_setpoint, h_v_setpoint.data(),
                          n_buses * sizeof(Real), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    // Initialize derived quantities to zero
    cudaMemsetAsync(buses_.d_p_injection, 0, n_buses * sizeof(Real), stream_);
    cudaMemsetAsync(buses_.d_q_injection, 0, n_buses * sizeof(Real), stream_);
    
    //=========================================================================
    // Upload Branch Data
    //=========================================================================
    
    std::vector<int32_t> h_from_bus(n_branches);
    std::vector<int32_t> h_to_bus(n_branches);
    std::vector<Real> h_resistance(n_branches);
    std::vector<Real> h_reactance(n_branches);
    std::vector<Real> h_susceptance(n_branches);
    std::vector<Real> h_conductance(n_branches);
    std::vector<Real> h_tap_ratio(n_branches);
    std::vector<Real> h_phase_shift(n_branches);
    std::vector<Real> h_g_series(n_branches);
    std::vector<Real> h_b_series(n_branches);
    std::vector<Real> h_b_shunt_from(n_branches);
    std::vector<Real> h_b_shunt_to(n_branches);
    std::vector<SwitchStatus> h_status(n_branches);
    
    #pragma omp parallel for schedule(static) if(n_branches > 100)
    for (int32_t i = 0; i < n_branches; ++i) {
        const auto* branch = model.getBranch(i);
        if (branch) {
            h_from_bus[i] = branch->from_bus_index;
            h_to_bus[i] = branch->to_bus_index;
            h_resistance[i] = branch->descriptor.resistance;
            h_reactance[i] = branch->descriptor.reactance;
            h_susceptance[i] = branch->descriptor.susceptance;
            h_conductance[i] = branch->descriptor.conductance;
            h_tap_ratio[i] = branch->descriptor.tap_ratio;
            h_phase_shift[i] = branch->descriptor.phase_shift;
            h_g_series[i] = branch->g_series;
            h_b_series[i] = branch->b_series;
            h_b_shunt_from[i] = branch->b_shunt_from;
            h_b_shunt_to[i] = branch->b_shunt_to;
            h_status[i] = branch->status;
        }
    }
    
    err = cudaMemcpyAsync(branches_.d_from_bus, h_from_bus.data(),
                          n_branches * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(branches_.d_to_bus, h_to_bus.data(),
                          n_branches * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(branches_.d_resistance, h_resistance.data(),
                          n_branches * sizeof(Real), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(branches_.d_reactance, h_reactance.data(),
                          n_branches * sizeof(Real), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(branches_.d_susceptance, h_susceptance.data(),
                          n_branches * sizeof(Real), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(branches_.d_conductance, h_conductance.data(),
                          n_branches * sizeof(Real), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(branches_.d_tap_ratio, h_tap_ratio.data(),
                          n_branches * sizeof(Real), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(branches_.d_phase_shift, h_phase_shift.data(),
                          n_branches * sizeof(Real), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(branches_.d_g_series, h_g_series.data(),
                          n_branches * sizeof(Real), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(branches_.d_b_series, h_b_series.data(),
                          n_branches * sizeof(Real), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(branches_.d_b_shunt_from, h_b_shunt_from.data(),
                          n_branches * sizeof(Real), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(branches_.d_b_shunt_to, h_b_shunt_to.data(),
                          n_branches * sizeof(Real), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(branches_.d_status, h_status.data(),
                          n_branches * sizeof(SwitchStatus), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    // Initialize flow quantities to zero
    cudaMemsetAsync(branches_.d_p_flow_from, 0, n_branches * sizeof(Real), stream_);
    cudaMemsetAsync(branches_.d_q_flow_from, 0, n_branches * sizeof(Real), stream_);
    cudaMemsetAsync(branches_.d_p_flow_to, 0, n_branches * sizeof(Real), stream_);
    cudaMemsetAsync(branches_.d_q_flow_to, 0, n_branches * sizeof(Real), stream_);
    cudaMemsetAsync(branches_.d_i_mag_from, 0, n_branches * sizeof(Real), stream_);
    cudaMemsetAsync(branches_.d_i_mag_to, 0, n_branches * sizeof(Real), stream_);
    
    //=========================================================================
    // Upload Measurement Data
    //=========================================================================
    
    std::vector<MeasurementType> h_meas_type(n_measurements);
    std::vector<int32_t> h_location_index(n_measurements);
    std::vector<BranchEnd> h_branch_end(n_measurements);
    std::vector<Real> h_pt_ratio(n_measurements);
    std::vector<Real> h_ct_ratio(n_measurements);
    std::vector<Real> h_value(n_measurements);
    std::vector<Real> h_weight(n_measurements);
    std::vector<Real> h_sigma(n_measurements);
    // Note: Can't use std::vector<bool> because it's a special template that doesn't have .data()
    std::vector<uint8_t> h_is_active(n_measurements);
    
    int32_t active_count = 0;
    #pragma omp parallel for schedule(static) reduction(+:active_count) if(n_measurements > 100)
    for (int32_t i = 0; i < n_measurements; ++i) {
        const auto* meas = model.getMeasurement(i);
        if (meas) {
            h_meas_type[i] = meas->descriptor.type;
            h_location_index[i] = meas->location_index;
            h_branch_end[i] = meas->descriptor.branch_end;
            h_pt_ratio[i] = meas->descriptor.pt_ratio;
            h_ct_ratio[i] = meas->descriptor.ct_ratio;
            h_value[i] = meas->value;
            h_weight[i] = meas->weight;
            h_sigma[i] = meas->descriptor.sigma;
            h_is_active[i] = meas->is_active ? 1 : 0;
            if (meas->is_active) active_count++;
        }
    }
    
    measurements_.active_count = active_count;
    
    err = cudaMemcpyAsync(measurements_.d_type, h_meas_type.data(),
                          n_measurements * sizeof(MeasurementType), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(measurements_.d_location_index, h_location_index.data(),
                          n_measurements * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(measurements_.d_branch_end, h_branch_end.data(),
                          n_measurements * sizeof(BranchEnd), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(measurements_.d_pt_ratio, h_pt_ratio.data(),
                          n_measurements * sizeof(Real), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(measurements_.d_ct_ratio, h_ct_ratio.data(),
                          n_measurements * sizeof(Real), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(measurements_.d_value, h_value.data(),
                          n_measurements * sizeof(Real), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(measurements_.d_weight, h_weight.data(),
                          n_measurements * sizeof(Real), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyAsync(measurements_.d_sigma, h_sigma.data(),
                          n_measurements * sizeof(Real), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    // Note: sizeof(bool) == sizeof(uint8_t) == 1 in CUDA
    err = cudaMemcpyAsync(measurements_.d_is_active, h_is_active.data(),
                          n_measurements * sizeof(uint8_t), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) return err;
    
    // Initialize derived quantities to zero
    cudaMemsetAsync(measurements_.d_estimated, 0, n_measurements * sizeof(Real), stream_);
    cudaMemsetAsync(measurements_.d_residual, 0, n_measurements * sizeof(Real), stream_);
    
    // Synchronize to ensure all uploads complete
    return cudaStreamSynchronize(stream_);
}

cudaError_t DeviceDataManager::updateMeasurements(const Real* values, int32_t count) {
    if (!is_allocated_ || count != measurements_.count) {
        return cudaErrorInvalidValue;
    }
    
    // Copy to pinned memory first
    std::memcpy(h_pinned_measurements_, values, count * sizeof(Real));
    
    // Async copy to device
    return cudaMemcpyAsync(measurements_.d_value, h_pinned_measurements_,
                          count * sizeof(Real), cudaMemcpyHostToDevice, stream_);
}

cudaError_t DeviceDataManager::downloadResults(NetworkModel& model) {
    if (!is_allocated_) {
        return cudaErrorNotReady;
    }
    
    int32_t n_buses = buses_.count;
    int32_t n_branches = branches_.count;
    int32_t n_measurements = measurements_.count;
    
    cudaError_t err;
    
    // Download bus voltage results
    std::vector<Real> h_v_mag(n_buses);
    std::vector<Real> h_v_angle(n_buses);
    std::vector<Real> h_p_inj(n_buses);
    std::vector<Real> h_q_inj(n_buses);
    
    err = cudaMemcpy(h_v_mag.data(), buses_.d_v_mag, 
                     n_buses * sizeof(Real), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpy(h_v_angle.data(), buses_.d_v_angle,
                     n_buses * sizeof(Real), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpy(h_p_inj.data(), buses_.d_p_injection,
                     n_buses * sizeof(Real), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpy(h_q_inj.data(), buses_.d_q_injection,
                     n_buses * sizeof(Real), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;
    
    // Update bus data in model
    for (int32_t i = 0; i < n_buses; ++i) {
        model.setBusVoltage(i, h_v_mag[i], h_v_angle[i]);
        model.setBusPower(i, h_p_inj[i], h_q_inj[i]);
    }
    
    // Download branch flow results
    std::vector<Real> h_p_from(n_branches);
    std::vector<Real> h_q_from(n_branches);
    std::vector<Real> h_p_to(n_branches);
    std::vector<Real> h_q_to(n_branches);
    std::vector<Real> h_i_from(n_branches);
    std::vector<Real> h_i_to(n_branches);
    
    err = cudaMemcpy(h_p_from.data(), branches_.d_p_flow_from,
                     n_branches * sizeof(Real), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpy(h_q_from.data(), branches_.d_q_flow_from,
                     n_branches * sizeof(Real), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpy(h_p_to.data(), branches_.d_p_flow_to,
                     n_branches * sizeof(Real), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpy(h_q_to.data(), branches_.d_q_flow_to,
                     n_branches * sizeof(Real), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpy(h_i_from.data(), branches_.d_i_mag_from,
                     n_branches * sizeof(Real), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpy(h_i_to.data(), branches_.d_i_mag_to,
                     n_branches * sizeof(Real), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;
    
    // Update branch data in model
    for (int32_t i = 0; i < n_branches; ++i) {
        model.setBranchFlows(i, h_p_from[i], h_q_from[i], h_p_to[i], h_q_to[i]);
        model.setBranchCurrents(i, h_i_from[i], h_i_to[i]);
    }
    
    // Download measurement results (estimated values and residuals)
    std::vector<Real> h_estimated(n_measurements);
    std::vector<Real> h_residual(n_measurements);
    
    err = cudaMemcpy(h_estimated.data(), measurements_.d_estimated,
                     n_measurements * sizeof(Real), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpy(h_residual.data(), measurements_.d_residual,
                     n_measurements * sizeof(Real), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;
    
    // Update measurement data in model
    for (int32_t i = 0; i < n_measurements; ++i) {
        model.setMeasurementResult(i, h_estimated[i], h_residual[i]);
    }
    
    return cudaSuccess;
}

void DeviceDataManager::free() {
    if (!is_allocated_) return;
    
    freeBusData();
    freeBranchData();
    freeMeasurementData();
    
    if (h_pinned_measurements_) {
        cudaFreeHost(h_pinned_measurements_);
        h_pinned_measurements_ = nullptr;
    }
    
    // Free Ybus
    if (ybus_.d_row_ptr) cudaFree(ybus_.d_row_ptr);
    if (ybus_.d_col_ind) cudaFree(ybus_.d_col_ind);
    if (ybus_.d_g_values) cudaFree(ybus_.d_g_values);
    if (ybus_.d_b_values) cudaFree(ybus_.d_b_values);
    ybus_ = {};
    
    total_allocated_ = 0;
    is_allocated_ = false;
}

//=============================================================================
// TopologyProcessor Implementation
//=============================================================================

TopologyProcessor::TopologyProcessor(cudaStream_t stream)
    : stream_(stream)
    , d_sd_status_(nullptr)
    , d_sd_branch_index_(nullptr)
    , n_devices_(0)
{
}

TopologyProcessor::~TopologyProcessor() {
    if (d_sd_status_) cudaFree(d_sd_status_);
    if (d_sd_branch_index_) cudaFree(d_sd_branch_index_);
}

cudaError_t TopologyProcessor::initialize(
    DeviceBranchData& branches,
    DeviceYbusMatrix& ybus,
    int32_t n_switching_devices)
{
    n_devices_ = n_switching_devices;
    
    if (n_devices_ > 0) {
        cudaError_t err;
        
        err = cudaMalloc(&d_sd_status_, n_devices_ * sizeof(SwitchStatus));
        if (err != cudaSuccess) return err;
        
        err = cudaMalloc(&d_sd_branch_index_, n_devices_ * sizeof(int32_t));
        if (err != cudaSuccess) return err;
    }
    
    return cudaSuccess;
}

void TopologyProcessor::queueStatusChange(int32_t sd_index, SwitchStatus new_status) {
    pending_changes_.emplace_back(sd_index, new_status);
}

cudaError_t TopologyProcessor::applyChanges(
    DeviceBranchData& branches,
    DeviceYbusMatrix& ybus,
    SparseMatrixManager& matrix_mgr)
{
    if (pending_changes_.empty()) {
        return cudaSuccess;
    }
    
    affected_branches_.clear();
    
    // Apply each pending change
    for (const auto& change : pending_changes_) {
        // Update device-side status
        // This would require kernel launch or host->device copy
        affected_branches_.push_back(change.first);
    }
    
    // Mark Ybus as invalid (needs rebuild)
    matrix_mgr.invalidateYbus(ybus);
    
    pending_changes_.clear();
    
    return cudaSuccess;
}

} // namespace sle

