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
 * @file network_model.h
 * @brief Host-side network model management for SLE Engine
 * 
 * Implements FR-01 (Static Model Definition), FR-02 (Measurement Mapping),
 * and FR-25 (Model Modification). Uses boost::stable_vector for pointer
 * stability during modifications (NFR-18).
 * 
 * The NetworkModel class provides:
 * - High-performance element lookup via std::unordered_map (Section 5.2)
 * - Contiguous indexed arrays for GPU transfer
 * - String ID to internal index mapping
 * 
 * @note Follows JSF C++ compliance (NFR-25) with explicit constructors,
 *       [[nodiscard]] attributes, and no exceptions in hot paths.
 */

#ifndef NETWORK_MODEL_H
#define NETWORK_MODEL_H

#include "sle_types.cuh"
#include <boost/container/stable_vector.hpp>
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <optional>

namespace sle {

//=============================================================================
// SECTION 1: Forward Declarations
//=============================================================================

class NetworkModel;
class TopologyProcessor;

//=============================================================================
// SECTION 2: Element Descriptor Structures
//=============================================================================

/**
 * @brief Bus definition for model initialization (FR-01)
 */
struct BusDescriptor {
    std::string id;                 ///< External string identifier
    Real base_kv;                   ///< Base voltage [kV]
    BusType type;                   ///< Bus type (PQ, PV, SLACK)
    Real v_setpoint;                ///< Voltage setpoint [p.u.] for PV buses
    Real p_scheduled;               ///< Scheduled active power [p.u.]
    Real q_scheduled;               ///< Scheduled reactive power [p.u.]
    
    /// Default constructor
    explicit BusDescriptor(
        std::string bus_id = "",
        Real kv = 1.0f,
        BusType t = BusType::PQ,
        Real v_sp = 1.0f,
        Real p_sch = 0.0f,
        Real q_sch = 0.0f)
        : id(std::move(bus_id))
        , base_kv(kv)
        , type(t)
        , v_setpoint(v_sp)
        , p_scheduled(p_sch)
        , q_scheduled(q_sch) {}
};

/**
 * @brief Branch definition for model initialization (FR-01)
 */
struct BranchDescriptor {
    std::string id;                 ///< External string identifier
    std::string from_bus_id;        ///< From bus string ID
    std::string to_bus_id;          ///< To bus string ID
    Real resistance;                ///< Series resistance [p.u.]
    Real reactance;                 ///< Series reactance [p.u.]
    Real susceptance;               ///< Total line charging [p.u.]
    Real conductance;               ///< Shunt conductance [p.u.] (usually 0)
    Real tap_ratio;                 ///< Off-nominal tap ratio (1.0 for lines)
    Real phase_shift;               ///< Phase shift [radians]
    bool is_transformer;            ///< True if transformer
    std::string sd_id;              ///< Associated switching device ID (empty if none)
    
    /// Default constructor for transmission line
    explicit BranchDescriptor(
        std::string branch_id = "",
        std::string from_id = "",
        std::string to_id = "",
        Real r = 0.0f,
        Real x = 0.01f,
        Real b = 0.0f,
        Real g = 0.0f,
        Real tap = 1.0f,
        Real shift = 0.0f,
        bool is_xfmr = false,
        std::string switch_id = "")
        : id(std::move(branch_id))
        , from_bus_id(std::move(from_id))
        , to_bus_id(std::move(to_id))
        , resistance(r)
        , reactance(x)
        , susceptance(b)
        , conductance(g)
        , tap_ratio(tap)
        , phase_shift(shift)
        , is_transformer(is_xfmr)
        , sd_id(std::move(switch_id)) {}
};

/**
 * @brief Measurement definition for model initialization (FR-02)
 */
struct MeasurementDescriptor {
    std::string id;                 ///< External string identifier
    MeasurementType type;           ///< Measurement type
    std::string location_id;        ///< Bus or branch string ID
    BranchEnd branch_end;           ///< For branch measurements
    Real sigma;                     ///< Standard deviation
    Real pt_ratio;                  ///< PT ratio (1.0 if not applicable)
    Real ct_ratio;                  ///< CT ratio (1.0 if not applicable)
    bool is_pseudo;                 ///< True for pseudo measurements
    
    /// Default constructor
    explicit MeasurementDescriptor(
        std::string meas_id = "",
        MeasurementType t = MeasurementType::V_MAG,
        std::string loc_id = "",
        BranchEnd end = BranchEnd::FROM,
        Real std_dev = 0.01f,
        Real pt = 1.0f,
        Real ct = 1.0f,
        bool pseudo = false)
        : id(std::move(meas_id))
        , type(t)
        , location_id(std::move(loc_id))
        , branch_end(end)
        , sigma(std_dev)
        , pt_ratio(pt)
        , ct_ratio(ct)
        , is_pseudo(pseudo) {}
};

/**
 * @brief Switching device definition (FR-03)
 */
struct SwitchingDeviceDescriptor {
    std::string id;                 ///< External string identifier
    std::string branch_id;          ///< Associated branch ID
    SwitchStatus initial_status;    ///< Initial status
    
    explicit SwitchingDeviceDescriptor(
        std::string sd_id = "",
        std::string br_id = "",
        SwitchStatus status = SwitchStatus::CLOSED)
        : id(std::move(sd_id))
        , branch_id(std::move(br_id))
        , initial_status(status) {}
};

//=============================================================================
// SECTION 3: Internal Element Structures (with indices)
//=============================================================================

/**
 * @brief Internal bus representation with computed indices
 */
struct BusElement {
    int32_t index;                  ///< Internal array index
    BusDescriptor descriptor;       ///< Original descriptor
    
    // State variables (updated during estimation)
    Real v_mag;                     ///< Current voltage magnitude [p.u.]
    Real v_angle;                   ///< Current voltage angle [rad]
    Real p_injection;               ///< Calculated injection
    Real q_injection;               ///< Calculated injection
    
    explicit BusElement(int32_t idx = INVALID_INDEX)
        : index(idx), v_mag(1.0f), v_angle(0.0f)
        , p_injection(0.0f), q_injection(0.0f) {}
};

/**
 * @brief Internal branch representation with computed indices
 */
struct BranchElement {
    int32_t index;                  ///< Internal array index
    BranchDescriptor descriptor;    ///< Original descriptor
    int32_t from_bus_index;         ///< Resolved from bus index
    int32_t to_bus_index;           ///< Resolved to bus index
    int32_t sd_index;               ///< Associated SD index (-1 if none)
    SwitchStatus status;            ///< Current status
    
    // Computed admittance values
    Real g_series;                  ///< Series conductance
    Real b_series;                  ///< Series susceptance
    Real b_shunt_from;              ///< From-end shunt susceptance
    Real b_shunt_to;                ///< To-end shunt susceptance
    
    // Flow calculations
    Real p_flow_from, q_flow_from;
    Real p_flow_to, q_flow_to;
    Real i_mag_from, i_mag_to;
    
    explicit BranchElement(int32_t idx = INVALID_INDEX)
        : index(idx), from_bus_index(INVALID_INDEX)
        , to_bus_index(INVALID_INDEX), sd_index(INVALID_INDEX)
        , status(SwitchStatus::CLOSED)
        , g_series(0), b_series(0), b_shunt_from(0), b_shunt_to(0)
        , p_flow_from(0), q_flow_from(0), p_flow_to(0), q_flow_to(0)
        , i_mag_from(0), i_mag_to(0) {}
};

/**
 * @brief Internal measurement representation
 */
struct MeasurementElement {
    int32_t index;                  ///< Internal array index
    MeasurementDescriptor descriptor;
    int32_t location_index;         ///< Resolved bus/branch index
    Real weight;                    ///< Computed weight (1/sigma^2)
    Real value;                     ///< Current measured value
    Real estimated;                 ///< Estimated h(x)
    Real residual;                  ///< z - h(x)
    bool is_active;                 ///< Active flag
    
    explicit MeasurementElement(int32_t idx = INVALID_INDEX)
        : index(idx), location_index(INVALID_INDEX), weight(1.0f)
        , value(0.0f), estimated(0.0f), residual(0.0f), is_active(true) {}
};

/**
 * @brief Internal switching device representation
 */
struct SwitchingDeviceElement {
    int32_t index;
    SwitchingDeviceDescriptor descriptor;
    int32_t branch_index;           ///< Resolved branch index
    SwitchStatus status;            ///< Current status
    SwitchStatus pending_status;    ///< Queued status
    bool has_pending_change;        ///< Pending change flag
    
    explicit SwitchingDeviceElement(int32_t idx = INVALID_INDEX)
        : index(idx), branch_index(INVALID_INDEX)
        , status(SwitchStatus::CLOSED), pending_status(SwitchStatus::CLOSED)
        , has_pending_change(false) {}
};

//=============================================================================
// SECTION 4: Hash Map Type Definitions (Section 5.2)
//=============================================================================

/**
 * @brief Custom string hasher using FNV-1a algorithm
 * @note Provides better distribution than std::hash<std::string> for typical IDs
 */
struct StringHasher {
    size_t operator()(const std::string& s) const {
        // FNV-1a hash for strings
        size_t hash = 14695981039346656037ULL;
        for (char c : s) {
            hash ^= static_cast<size_t>(c);
            hash *= 1099511628211ULL;
        }
        return hash;
    }
};

/// Map type for string ID to internal index lookup
using IdIndexMap = std::unordered_map<std::string, int32_t, StringHasher>;

//=============================================================================
// SECTION 5: Network Model Class
//=============================================================================

/**
 * @class NetworkModel
 * @brief Host-side network model management
 * 
 * Manages all network elements with:
 * - O(1) lookup from string IDs to internal indices (Section 5.2)
 * - Pointer-stable storage using boost::stable_vector (NFR-18)
 * - Efficient GPU data transfer preparation
 * 
 * Thread-safety: NOT thread-safe. External synchronization required.
 */
class NetworkModel {
public:
    /**
     * @brief Default constructor
     * 
     * Initializes empty model with pre-configured hash maps.
     */
    NetworkModel();
    
    /**
     * @brief Destructor
     */
    ~NetworkModel();
    
    // Disable copy (NFR-06: avoid allocations)
    NetworkModel(const NetworkModel&) = delete;
    NetworkModel& operator=(const NetworkModel&) = delete;
    
    // Enable move
    NetworkModel(NetworkModel&&) noexcept;
    NetworkModel& operator=(NetworkModel&&) noexcept;

    //=========================================================================
    // Model Building API (FR-01)
    //=========================================================================
    
    /**
     * @brief Add a bus to the network
     * 
     * @param desc Bus descriptor
     * @return Internal index of added bus, or INVALID_INDEX on failure
     */
    [[nodiscard]] int32_t addBus(const BusDescriptor& desc);
    
    /**
     * @brief Add a branch to the network
     * 
     * @param desc Branch descriptor (bus IDs must already exist)
     * @return Internal index of added branch, or INVALID_INDEX on failure
     */
    [[nodiscard]] int32_t addBranch(const BranchDescriptor& desc);
    
    /**
     * @brief Add a measurement to the network (FR-02)
     * 
     * @param desc Measurement descriptor
     * @return Internal index of added measurement, or INVALID_INDEX on failure
     */
    [[nodiscard]] int32_t addMeasurement(const MeasurementDescriptor& desc);
    
    /**
     * @brief Add a switching device (FR-03)
     * 
     * @param desc Switching device descriptor
     * @return Internal index of added device, or INVALID_INDEX on failure
     */
    [[nodiscard]] int32_t addSwitchingDevice(const SwitchingDeviceDescriptor& desc);

    //=========================================================================
    // Model Modification API (FR-25)
    //=========================================================================
    
    /**
     * @brief Remove a bus and all connected elements
     * 
     * @param bus_id Bus string ID
     * @return true if removed successfully
     */
    [[nodiscard]] bool removeBus(const std::string& bus_id);
    
    /**
     * @brief Remove a branch
     * 
     * @param branch_id Branch string ID
     * @return true if removed successfully
     */
    [[nodiscard]] bool removeBranch(const std::string& branch_id);
    
    /**
     * @brief Update branch parameters
     * 
     * @param branch_id Branch string ID
     * @param desc New branch descriptor
     * @return true if updated successfully
     */
    [[nodiscard]] bool updateBranch(const std::string& branch_id,
                                     const BranchDescriptor& desc);
    
    /**
     * @brief Mark model as modified (triggers full cycle)
     */
    void markModified() { model_modified_ = true; }
    
    /**
     * @brief Check if model has structural modifications
     */
    [[nodiscard]] bool isModified() const { return model_modified_; }
    
    /**
     * @brief Clear modification flag after handling
     */
    void clearModified() { model_modified_ = false; }

    //=========================================================================
    // Index Lookup API (Section 5.2)
    //=========================================================================
    
    /**
     * @brief Get bus index from string ID
     * 
     * @param id Bus string ID
     * @return Internal index or INVALID_INDEX if not found
     */
    [[nodiscard]] int32_t getBusIndex(const std::string& id) const;
    
    /**
     * @brief Get branch index from string ID
     */
    [[nodiscard]] int32_t getBranchIndex(const std::string& id) const;
    
    /**
     * @brief Get measurement index from string ID
     */
    [[nodiscard]] int32_t getMeasurementIndex(const std::string& id) const;
    
    /**
     * @brief Get switching device index from string ID
     */
    [[nodiscard]] int32_t getSwitchingDeviceIndex(const std::string& id) const;

    //=========================================================================
    // Element Access
    //=========================================================================
    
    /**
     * @brief Get number of buses
     */
    [[nodiscard]] int32_t getBusCount() const { 
        return static_cast<int32_t>(buses_.size()); 
    }
    
    /**
     * @brief Get number of branches
     */
    [[nodiscard]] int32_t getBranchCount() const { 
        return static_cast<int32_t>(branches_.size()); 
    }
    
    /**
     * @brief Get number of measurements
     */
    [[nodiscard]] int32_t getMeasurementCount() const { 
        return static_cast<int32_t>(measurements_.size()); 
    }
    
    /**
     * @brief Get number of switching devices
     */
    [[nodiscard]] int32_t getSwitchingDeviceCount() const { 
        return static_cast<int32_t>(switching_devices_.size()); 
    }
    
    /**
     * @brief Get slack bus index
     */
    [[nodiscard]] int32_t getSlackBusIndex() const { return slack_bus_index_; }
    
    /**
     * @brief Get bus element by index
     */
    [[nodiscard]] BusElement* getBus(int32_t index);
    [[nodiscard]] const BusElement* getBus(int32_t index) const;
    
    /**
     * @brief Get branch element by index
     */
    [[nodiscard]] BranchElement* getBranch(int32_t index);
    [[nodiscard]] const BranchElement* getBranch(int32_t index) const;
    
    /**
     * @brief Get measurement element by index
     */
    [[nodiscard]] MeasurementElement* getMeasurement(int32_t index);
    [[nodiscard]] const MeasurementElement* getMeasurement(int32_t index) const;
    
    /**
     * @brief Get switching device by index
     */
    [[nodiscard]] SwitchingDeviceElement* getSwitchingDevice(int32_t index);
    [[nodiscard]] const SwitchingDeviceElement* getSwitchingDevice(int32_t index) const;

    //=========================================================================
    // Telemetry Update (FR-07)
    //=========================================================================
    
    /**
     * @brief Update measurement values (real-time telemetry)
     * 
     * @param values Array of new measurement values
     * @param count Number of values
     * @return true if update successful
     */
    [[nodiscard]] bool updateMeasurementValues(const Real* values, int32_t count);
    
    /**
     * @brief Update single measurement value
     * 
     * @param meas_id Measurement string ID
     * @param value New value
     * @return true if update successful
     */
    [[nodiscard]] bool updateMeasurementValue(const std::string& meas_id, Real value);

    //=========================================================================
    // GPU Data Preparation
    //=========================================================================
    
    /**
     * @brief Pack bus data into host SoA structure for GPU transfer
     * 
     * @param data Pre-allocated host bus data structure
     */
    void packBusData(HostBusData& data) const;
    
    /**
     * @brief Pack branch data into host SoA structure
     * 
     * @param data Pre-allocated host branch data structure
     */
    void packBranchData(HostBranchData& data) const;
    
    /**
     * @brief Pack measurement data into host SoA structure
     * 
     * @param data Pre-allocated host measurement data structure
     */
    void packMeasurementData(HostMeasurementData& data) const;
    
    /**
     * @brief Pack switching device data into host SoA structure
     */
    void packSwitchingDeviceData(HostSwitchingDeviceData& data) const;
    
    /**
     * @brief Calculate total memory required for GPU data
     */
    [[nodiscard]] size_t calculateGPUMemoryRequired() const;

    //=========================================================================
    // Validation
    //=========================================================================
    
    /**
     * @brief Validate model integrity
     * 
     * Checks:
     * - All bus references are valid
     * - At least one slack bus exists
     * - No isolated buses (warning)
     * - Measurement locations exist
     * 
     * @return true if model is valid
     */
    [[nodiscard]] bool validate() const;
    
    /**
     * @brief Compute branch admittances from impedances
     * 
     * Calculates g_series, b_series, b_shunt for all branches.
     * Called after model building or modification.
     */
    void computeBranchAdmittances();
    
    /**
     * @brief Apply flat start initialization (1.0 p.u., 0 deg)
     */
    void applyFlatStart();

private:
    // Element storage using boost::stable_vector (NFR-18)
    boost::container::stable_vector<BusElement> buses_;
    boost::container::stable_vector<BranchElement> branches_;
    boost::container::stable_vector<MeasurementElement> measurements_;
    boost::container::stable_vector<SwitchingDeviceElement> switching_devices_;
    
    // String ID to index maps (Section 5.2)
    IdIndexMap bus_id_map_;
    IdIndexMap branch_id_map_;
    IdIndexMap measurement_id_map_;
    IdIndexMap sd_id_map_;
    
    // Model state
    int32_t slack_bus_index_;
    bool model_modified_;
    
    // Initialization helpers
    void initializeHashMaps();
    void resolveReferences();
    [[nodiscard]] bool isBusMeasurement(MeasurementType type) const;
};

//=============================================================================
// SECTION 6: Host Memory Allocator for SoA Structures
//=============================================================================

/**
 * @class HostDataAllocator
 * @brief Allocates and manages pinned memory for host SoA structures
 * 
 * Uses CUDA pinned memory (page-locked) for efficient DMA transfers (FR-06).
 * Provides RAII-style memory management.
 */
class HostDataAllocator {
public:
    HostDataAllocator();
    ~HostDataAllocator();
    
    // Disable copy
    HostDataAllocator(const HostDataAllocator&) = delete;
    HostDataAllocator& operator=(const HostDataAllocator&) = delete;
    
    /**
     * @brief Allocate host bus data arrays
     * 
     * @param data Structure to populate with allocated pointers
     * @param count Number of buses
     * @return cudaSuccess on success
     */
    [[nodiscard]] cudaError_t allocateBusData(HostBusData& data, int32_t count);
    
    /**
     * @brief Free host bus data arrays
     */
    void freeBusData(HostBusData& data);
    
    /**
     * @brief Allocate host branch data arrays
     */
    [[nodiscard]] cudaError_t allocateBranchData(HostBranchData& data, int32_t count);
    
    /**
     * @brief Free host branch data arrays
     */
    void freeBranchData(HostBranchData& data);
    
    /**
     * @brief Allocate host measurement data arrays
     */
    [[nodiscard]] cudaError_t allocateMeasurementData(HostMeasurementData& data, 
                                                       int32_t count);
    
    /**
     * @brief Free host measurement data arrays
     */
    void freeMeasurementData(HostMeasurementData& data);
    
    /**
     * @brief Allocate host switching device data arrays
     */
    [[nodiscard]] cudaError_t allocateSwitchingDeviceData(HostSwitchingDeviceData& data,
                                                           int32_t count);
    
    /**
     * @brief Free host switching device data arrays
     */
    void freeSwitchingDeviceData(HostSwitchingDeviceData& data);

private:
    // Track allocations for cleanup
    std::vector<void*> allocations_;
};

} // namespace sle

#endif // NETWORK_MODEL_H

