/**
 * SLE Engine - CUDA-Accelerated State Load Estimator
 * Copyright (c) 2025, Oleksandr Don
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
 * Meter-Centric Design:
 * - Meters (Voltmeters, Multimeters) are the PRIMARY source of measurement values
 * - You don't update bus voltages directly - you update the voltmeter on that bus
 * - You don't update branch flows directly - you update the multimeter on that branch
 * - Meter readings flow into the underlying measurements for state estimation
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
 * 
 * @note Prefer using Meters (addMeter) which automatically create measurements.
 *       Direct measurement creation is for advanced use cases.
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

/**
 * @brief Meter device types
 * 
 * Meters are the PRIMARY source of measurement values in the topology:
 * - VOLTMETER: Connected to a bus via PT, provides voltage measurement
 * - MULTIMETER: Connected to a branch via PT+CT, provides P, Q, V, I measurements
 * - AMMETER: Connected to a branch via CT, provides current measurement
 * - WATTMETER: Connected to a branch via PT+CT, provides P and Q measurements
 */
enum class MeterType {
    VOLTMETER = 0,      ///< Voltage meter on bus (PT only)
    MULTIMETER = 1,     ///< Multi-function meter on branch (PT + CT)
    AMMETER = 2,        ///< Current meter on branch (CT only)
    WATTMETER = 3       ///< Power meter on branch (PT + CT)
};

/**
 * @brief Meter device descriptor
 * 
 * Represents a physical metering device in the topology.
 * 
 * Design philosophy:
 * - Meters OWN measurements - they are the source of truth for measurement values
 * - To update bus voltage, update the voltmeter connected to that bus
 * - To update branch flows, update the multimeter on that branch
 * - Meter readings directly affect state estimation calculations
 */
struct MeterDescriptor {
    std::string id;                 ///< Meter ID (e.g., "VM1", "MM1")
    MeterType type;                 ///< Meter type
    std::string bus_id;             ///< Bus where meter is connected
    std::string branch_id;          ///< Branch ID (for branch meters)
    BranchEnd branch_end;           ///< Measurement point on branch
    Real pt_ratio;                  ///< Potential transformer ratio
    Real ct_ratio;                  ///< Current transformer ratio
    Real sigma_v;                   ///< Std dev for voltage measurements
    Real sigma_p;                   ///< Std dev for power measurements
    Real sigma_i;                   ///< Std dev for current measurements
    
    explicit MeterDescriptor(
        std::string meter_id = "",
        MeterType t = MeterType::VOLTMETER,
        std::string bus = "",
        std::string branch = "",
        BranchEnd end = BranchEnd::FROM,
        Real pt = 1.0f,
        Real ct = 1.0f,
        Real sv = 0.004f,
        Real sp = 0.01f,
        Real si = 0.01f)
        : id(std::move(meter_id))
        , type(t)
        , bus_id(std::move(bus))
        , branch_id(std::move(branch))
        , branch_end(end)
        , pt_ratio(pt)
        , ct_ratio(ct)
        , sigma_v(sv)
        , sigma_p(sp)
        , sigma_i(si) {}
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

/**
 * @brief Internal meter device representation
 * 
 * Maps meter channels to underlying measurement indices.
 * The meter "owns" these measurements - updating the meter updates the measurements.
 */
struct MeterElement {
    int32_t index;                  ///< Internal meter index
    MeterDescriptor descriptor;     ///< Original descriptor
    int32_t bus_index;              ///< Resolved bus index
    int32_t branch_index;           ///< Resolved branch index (-1 for voltmeter)
    
    // Measurement indices for each channel (-1 if not available)
    int32_t meas_idx_v;             ///< Voltage measurement index
    int32_t meas_idx_p;             ///< Active power measurement index
    int32_t meas_idx_q;             ///< Reactive power measurement index
    int32_t meas_idx_i;             ///< Current measurement index
    
    explicit MeterElement(int32_t idx = INVALID_INDEX)
        : index(idx), bus_index(INVALID_INDEX), branch_index(INVALID_INDEX)
        , meas_idx_v(INVALID_INDEX), meas_idx_p(INVALID_INDEX)
        , meas_idx_q(INVALID_INDEX), meas_idx_i(INVALID_INDEX) {}
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
 * Meter-Centric Workflow:
 * 1. Build topology: buses, branches
 * 2. Add meters: voltmeters on buses, multimeters on branches
 * 3. Update meter readings (primary telemetry input)
 * 4. Sync to GPU and solve
 * 5. Read estimated values from meters
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
     * @note Prefer using addMeter() which creates measurements automatically
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
    // Meter Device Management (PRIMARY telemetry interface)
    //=========================================================================
    
    /**
     * @brief Add a meter device to the model
     * 
     * Creates the meter and its underlying measurements automatically.
     * This is the PREFERRED way to add measurement capability to the model.
     * 
     * @param desc Meter descriptor
     * @return Meter index, or INVALID_INDEX on failure
     */
    [[nodiscard]] int32_t addMeter(const MeterDescriptor& desc);
    
    /**
     * @brief Update a meter reading by channel
     * 
     * This is the PRIMARY way to input telemetry values.
     * Updates flow to underlying measurements and affect state estimation.
     * 
     * @param meter_id Meter string ID
     * @param channel Channel name ("V", "kW", "kVAR", "A")
     * @param value Reading value
     * @return true on success
     */
    [[nodiscard]] bool updateMeterReading(const std::string& meter_id, 
                                          const std::string& channel, 
                                          Real value);
    
    /**
     * @brief Get meter reading by channel
     */
    [[nodiscard]] bool getMeterReading(const std::string& meter_id,
                                       const std::string& channel,
                                       Real& value) const;
    
    /**
     * @brief Get meter estimated value by channel (after solve)
     */
    [[nodiscard]] bool getMeterEstimate(const std::string& meter_id,
                                        const std::string& channel,
                                        Real& value) const;
    
    /**
     * @brief Get meter residual by channel (after solve)
     */
    [[nodiscard]] bool getMeterResidual(const std::string& meter_id,
                                        const std::string& channel,
                                        Real& value) const;

    //=========================================================================
    // Model Modification API (FR-25)
    //=========================================================================
    
    [[nodiscard]] bool removeBus(const std::string& bus_id);
    [[nodiscard]] bool removeBranch(const std::string& branch_id);
    [[nodiscard]] bool updateBranch(const std::string& branch_id,
                                     const BranchDescriptor& desc);
    
    void markModified() { model_modified_ = true; }
    [[nodiscard]] bool isModified() const { return model_modified_; }
    void clearModified() { model_modified_ = false; }

    //=========================================================================
    // Index Lookup API (Section 5.2)
    //=========================================================================
    
    [[nodiscard]] int32_t getBusIndex(const std::string& id) const;
    [[nodiscard]] int32_t getBranchIndex(const std::string& id) const;
    [[nodiscard]] int32_t getMeasurementIndex(const std::string& id) const;
    [[nodiscard]] int32_t getSwitchingDeviceIndex(const std::string& id) const;
    [[nodiscard]] int32_t getMeterIndex(const std::string& id) const;

    //=========================================================================
    // Element Access
    //=========================================================================
    
    [[nodiscard]] int32_t getBusCount() const { return static_cast<int32_t>(buses_.size()); }
    [[nodiscard]] int32_t getBranchCount() const { return static_cast<int32_t>(branches_.size()); }
    [[nodiscard]] int32_t getMeasurementCount() const { return static_cast<int32_t>(measurements_.size()); }
    [[nodiscard]] int32_t getSwitchingDeviceCount() const { return static_cast<int32_t>(switching_devices_.size()); }
    [[nodiscard]] int32_t getMeterCount() const { return static_cast<int32_t>(meters_.size()); }
    [[nodiscard]] int32_t getSlackBusIndex() const { return slack_bus_index_; }
    
    [[nodiscard]] BusElement* getBus(int32_t index);
    [[nodiscard]] const BusElement* getBus(int32_t index) const;
    [[nodiscard]] BranchElement* getBranch(int32_t index);
    [[nodiscard]] const BranchElement* getBranch(int32_t index) const;
    [[nodiscard]] MeasurementElement* getMeasurement(int32_t index);
    [[nodiscard]] const MeasurementElement* getMeasurement(int32_t index) const;
    [[nodiscard]] SwitchingDeviceElement* getSwitchingDevice(int32_t index);
    [[nodiscard]] const SwitchingDeviceElement* getSwitchingDevice(int32_t index) const;
    [[nodiscard]] MeterElement* getMeter(int32_t index);
    [[nodiscard]] const MeterElement* getMeter(int32_t index) const;

    //=========================================================================
    // Telemetry Update (FR-07) - Low-level interface
    //=========================================================================
    
    /**
     * @brief Update measurement values (real-time telemetry) - bulk update
     * @note Prefer updateMeterReading() for meter-centric workflow
     */
    [[nodiscard]] bool updateMeasurementValues(const Real* values, int32_t count);
    
    /**
     * @brief Update single measurement value - low-level
     * @note Prefer updateMeterReading() for meter-centric workflow
     */
    [[nodiscard]] bool updateMeasurementValue(const std::string& meas_id, Real value);

    //=========================================================================
    // GPU Data Preparation
    //=========================================================================
    
    void packBusData(HostBusData& data) const;
    void packBranchData(HostBranchData& data) const;
    void packMeasurementData(HostMeasurementData& data) const;
    void packSwitchingDeviceData(HostSwitchingDeviceData& data) const;
    [[nodiscard]] size_t calculateGPUMemoryRequired() const;

    //=========================================================================
    // Validation
    //=========================================================================
    
    [[nodiscard]] bool validate() const;
    void computeBranchAdmittances();
    void applyFlatStart();
    
    //=========================================================================
    // Result Setters (for GPU result download)
    //=========================================================================
    
    void setBusVoltage(int32_t index, Real v_mag, Real v_angle);
    void setBusPower(int32_t index, Real p_inj, Real q_inj);
    void setBranchFlows(int32_t index, Real p_from, Real q_from, Real p_to, Real q_to);
    void setBranchCurrents(int32_t index, Real i_from, Real i_to);
    void setMeasurementResult(int32_t index, Real estimated, Real residual);

private:
    // Element storage using boost::stable_vector (NFR-18)
    boost::container::stable_vector<BusElement> buses_;
    boost::container::stable_vector<BranchElement> branches_;
    boost::container::stable_vector<MeasurementElement> measurements_;
    boost::container::stable_vector<SwitchingDeviceElement> switching_devices_;
    boost::container::stable_vector<MeterElement> meters_;
    
    // String ID to index maps (Section 5.2)
    IdIndexMap bus_id_map_;
    IdIndexMap branch_id_map_;
    IdIndexMap measurement_id_map_;
    IdIndexMap sd_id_map_;
    IdIndexMap meter_id_map_;
    
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
 */
class HostDataAllocator {
public:
    HostDataAllocator();
    ~HostDataAllocator();
    
    HostDataAllocator(const HostDataAllocator&) = delete;
    HostDataAllocator& operator=(const HostDataAllocator&) = delete;
    
    [[nodiscard]] cudaError_t allocateBusData(HostBusData& data, int32_t count);
    void freeBusData(HostBusData& data);
    [[nodiscard]] cudaError_t allocateBranchData(HostBranchData& data, int32_t count);
    void freeBranchData(HostBranchData& data);
    [[nodiscard]] cudaError_t allocateMeasurementData(HostMeasurementData& data, int32_t count);
    void freeMeasurementData(HostMeasurementData& data);
    [[nodiscard]] cudaError_t allocateSwitchingDeviceData(HostSwitchingDeviceData& data, int32_t count);
    void freeSwitchingDeviceData(HostSwitchingDeviceData& data);

private:
    std::vector<void*> allocations_;
};

} // namespace sle

#endif // NETWORK_MODEL_H
