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
 * @file sle_engine.h
 * @brief Main SLE Engine class - public API for state estimation
 * 
 * This is the primary interface for the CUDA-Accelerated State Load Estimator.
 * The SLEEngine class coordinates all components:
 * - NetworkModel: Host-side network data management (FR-01, FR-02)
 * - TopologyProcessor: Switch status and Ybus updates (FR-03, FR-05)
 * - WLSSolver: GPU-based WLS estimation (FR-08)
 * - DeviceDataManager: GPU memory management (FR-06)
 * 
 * Usage pattern:
 * 1. Create engine and initialize CUDA device
 * 2. Build network model (buses, branches, measurements)
 * 3. Upload model to GPU
 * 4. Run estimation cycles as telemetry arrives
 * 5. Retrieve results
 * 
 * @note This header is safe to include in host-only C++ code.
 *       It does not require CUDA compilation.
 */

#ifndef SLE_ENGINE_H
#define SLE_ENGINE_H

#include "sle_types.cuh"
#include "sle_config.h"
#include "network_model.h"
#include <memory>
#include <functional>
#include <string>
#include <vector>

namespace sle {

// Forward declarations (implementation details hidden)
class WLSSolver;
class SparseMatrixManager;
class TopologyProcessor;
class DeviceDataManager;

//=============================================================================
// SECTION 1: Engine Configuration
//=============================================================================

/**
 * @brief Engine initialization configuration
 * 
 * Configuration can be loaded from a .config file using ConfigLoader,
 * or set programmatically. All values have sensible defaults but should
 * be tuned for production deployments.
 * 
 * @see ConfigLoader for file-based configuration
 * @see EngineConfiguration for the complete configuration structure
 */
struct EngineConfig {
    int32_t device_id;              ///< CUDA device ID (default 0)
    bool enable_profiling;          ///< Enable CUDA event timing
    bool enable_robust_estimation;  ///< Enable Huber M-estimator by default
    Real huber_gamma;               ///< Default Huber threshold
    size_t max_buses;               ///< Pre-allocate for this many buses
    size_t max_branches;            ///< Pre-allocate for this many branches
    size_t max_measurements;        ///< Pre-allocate for this many measurements
    size_t max_switching_devices;   ///< Pre-allocate for switching devices
    
    // Additional configuration from file
    Real convergence_tolerance;     ///< WLS convergence tolerance
    int32_t max_realtime_iterations;///< Max iterations for real-time mode
    int32_t max_precision_iterations;///< Max iterations for precision mode
    Real realtime_time_limit_ms;    ///< Time limit for real-time mode [ms]
    bool use_flat_start_default;    ///< Default to flat start initialization
    bool use_cuda_graphs;           ///< Use CUDA graphs for iteration
    Real voltage_min_pu;            ///< Minimum voltage limit [p.u.]
    Real voltage_max_pu;            ///< Maximum voltage limit [p.u.]
    Real bad_data_threshold;        ///< Normalized residual threshold
    bool auto_bad_data_rejection;   ///< Automatically reject bad data
    
    /// Default configuration (hardcoded fallback if no config file)
    EngineConfig() :
        device_id(0),
        enable_profiling(false),
        enable_robust_estimation(false),
        huber_gamma(DEFAULT_HUBER_GAMMA),
        max_buses(200000),
        max_branches(400000),
        max_measurements(2000000),
        max_switching_devices(100000),
        convergence_tolerance(SLE_CONVERGENCE_TOL),
        max_realtime_iterations(MAX_REALTIME_ITERATIONS),
        max_precision_iterations(MAX_PRECISION_ITERATIONS),
        realtime_time_limit_ms(20.0f),
        use_flat_start_default(false),
        use_cuda_graphs(true),
        voltage_min_pu(0.5f),
        voltage_max_pu(1.5f),
        bad_data_threshold(3.0f),
        auto_bad_data_rejection(false) {}
    
    /**
     * @brief Construct from EngineConfiguration (loaded from file)
     * @param cfg Configuration loaded via ConfigLoader
     */
    explicit EngineConfig(const EngineConfiguration& cfg) :
        device_id(cfg.device_id),
        enable_profiling(cfg.enable_profiling),
        enable_robust_estimation(cfg.enable_robust_estimation),
        huber_gamma(cfg.huber_gamma),
        max_buses(cfg.max_buses),
        max_branches(cfg.max_branches),
        max_measurements(cfg.max_measurements),
        max_switching_devices(cfg.max_switching_devices),
        convergence_tolerance(cfg.convergence_tolerance),
        max_realtime_iterations(cfg.max_realtime_iterations),
        max_precision_iterations(cfg.max_precision_iterations),
        realtime_time_limit_ms(cfg.realtime_time_limit_ms),
        use_flat_start_default(cfg.use_flat_start_default),
        use_cuda_graphs(cfg.use_cuda_graphs),
        voltage_min_pu(cfg.voltage_min_pu),
        voltage_max_pu(cfg.voltage_max_pu),
        bad_data_threshold(cfg.bad_data_threshold),
        auto_bad_data_rejection(cfg.auto_bad_data_rejection) {}
    
    /**
     * @brief Load configuration from file
     * 
     * Loads settings from specified config file. Falls back to defaults
     * for any missing values.
     * 
     * @param filepath Path to .config file
     * @return EngineConfig with loaded values
     */
    [[nodiscard]] static EngineConfig loadFromFile(const std::string& filepath) {
        EngineConfiguration cfg;
        if (ConfigLoader::loadFromFile(filepath, cfg)) {
            return EngineConfig(cfg);
        }
        // Return defaults if file load fails
        return EngineConfig();
    }
    
    /**
     * @brief Load configuration from default file path
     * 
     * Attempts to load from "sle_engine.config" in current directory.
     * 
     * @return EngineConfig with loaded values or defaults
     */
    [[nodiscard]] static EngineConfig loadFromDefaultFile() {
        return loadFromFile(ConfigLoader::getDefaultConfigPath());
    }
    
    /**
     * @brief Save current configuration to file
     * 
     * @param filepath Output file path
     * @return true if saved successfully
     */
    [[nodiscard]] bool saveToFile(const std::string& filepath) const {
        EngineConfiguration cfg;
        cfg.device_id = device_id;
        cfg.enable_profiling = enable_profiling;
        cfg.enable_robust_estimation = enable_robust_estimation;
        cfg.huber_gamma = huber_gamma;
        cfg.max_buses = max_buses;
        cfg.max_branches = max_branches;
        cfg.max_measurements = max_measurements;
        cfg.max_switching_devices = max_switching_devices;
        cfg.convergence_tolerance = convergence_tolerance;
        cfg.max_realtime_iterations = max_realtime_iterations;
        cfg.max_precision_iterations = max_precision_iterations;
        cfg.realtime_time_limit_ms = realtime_time_limit_ms;
        cfg.use_flat_start_default = use_flat_start_default;
        cfg.use_cuda_graphs = use_cuda_graphs;
        cfg.voltage_min_pu = voltage_min_pu;
        cfg.voltage_max_pu = voltage_max_pu;
        cfg.bad_data_threshold = bad_data_threshold;
        cfg.auto_bad_data_rejection = auto_bad_data_rejection;
        
        return ConfigLoader::saveToFile(filepath, cfg);
    }
    
    /**
     * @brief Create default configuration file
     * 
     * Creates a new configuration file with all default values and comments.
     * 
     * @param filepath Output file path
     * @return true if created successfully
     */
    [[nodiscard]] static bool createDefaultConfigFile(const std::string& filepath) {
        return ConfigLoader::createDefaultConfigFile(filepath);
    }
};

/**
 * @brief Callback type for progress reporting
 * 
 * Parameters: iteration number, max mismatch, objective value
 */
using ProgressCallback = std::function<void(int32_t, Real, Real)>;

/**
 * @brief Callback type for bad data detection
 * 
 * Parameters: measurement index, residual value
 */
using BadDataCallback = std::function<void(int32_t, Real)>;

//=============================================================================
// SECTION 2: Main SLE Engine Class
//=============================================================================

/**
 * @class SLEEngine
 * @brief High-level API for CUDA-accelerated state estimation
 * 
 * This class provides a complete state estimation solution:
 * 
 * **Model Management (FR-01, FR-02, FR-25)**
 * - Add/modify network elements (buses, branches, transformers)
 * - Define measurements with PT/CT ratios
 * - Track switching device status
 * 
 * **Telemetry Handling (FR-07)**
 * - Efficient measurement value updates via pinned memory
 * - Asynchronous GPU data transfer
 * 
 * **State Estimation (FR-08 through FR-12)**
 * - Real-time mode with hot start and gain matrix reuse
 * - Precision mode with full refactorization
 * - Robust estimation with Huber M-estimator
 * 
 * **Result Access (FR-10, FR-14, FR-15)**
 * - State vector (V, theta at all buses)
 * - Derived quantities (P, Q flows and injections)
 * - Residuals and bad data identification
 * 
 * Thread-safety: The engine is NOT thread-safe. If multiple threads
 * need to perform estimation, use separate engine instances.
 * 
 * @code{.cpp}
 * // Example usage
 * sle::SLEEngine engine;
 * engine.initialize();
 * 
 * // Build model
 * engine.addBus({"Bus1", 138.0, sle::BusType::SLACK});
 * engine.addBus({"Bus2", 138.0, sle::BusType::PQ});
 * engine.addBranch({"Line1", "Bus1", "Bus2", 0.01, 0.1, 0.02});
 * engine.addMeasurement({"V1", sle::MeasurementType::V_MAG, "Bus1"});
 * 
 * // Upload and solve
 * engine.uploadModel();
 * auto result = engine.solve(sle::EstimationMode::PRECISION);
 * 
 * // Get results
 * std::vector<sle::Real> voltages = engine.getVoltageMagnitudes();
 * @endcode
 */
class SLEEngine {
public:
    /**
     * @brief Default constructor
     * 
     * Creates engine and automatically attempts to load configuration from
     * the default config file "sle_engine.config" in the current directory.
     * Falls back to hardcoded defaults if the file is not found.
     * 
     * Call initialize() before use.
     */
    SLEEngine();
    
    /**
     * @brief Constructor with configuration
     * 
     * @param config Engine configuration (may be loaded via EngineConfig::loadFromFile)
     */
    explicit SLEEngine(const EngineConfig& config);
    
    /**
     * @brief Constructor with config file path
     * 
     * Loads configuration from specified file. Falls back to defaults
     * if file cannot be read.
     * 
     * @param config_path Path to .config file
     */
    explicit SLEEngine(const std::string& config_path);
    
    /**
     * @brief Destructor - releases all resources
     */
    ~SLEEngine();
    
    // Disable copy (large GPU resources)
    SLEEngine(const SLEEngine&) = delete;
    SLEEngine& operator=(const SLEEngine&) = delete;
    
    // Enable move
    SLEEngine(SLEEngine&&) noexcept;
    SLEEngine& operator=(SLEEngine&&) noexcept;

    //=========================================================================
    // Initialization
    //=========================================================================
    
    /**
     * @brief Initialize CUDA device and allocate resources
     * 
     * Must be called before any other operations.
     * Selects CUDA device and allocates initial buffers.
     * 
     * @return true on success, false on failure
     */
    [[nodiscard]] bool initialize();
    
    /**
     * @brief Initialize with specific configuration
     * 
     * @param config Engine configuration
     * @return true on success
     */
    [[nodiscard]] bool initialize(const EngineConfig& config);
    
    /**
     * @brief Check if engine is initialized
     */
    [[nodiscard]] bool isInitialized() const { return initialized_; }
    
    /**
     * @brief Get CUDA device properties
     */
    [[nodiscard]] std::string getDeviceInfo() const;
    
    /**
     * @brief Reset engine to initial state
     * 
     * Clears model and releases GPU memory. Call initialize() again after.
     */
    void reset();

    //=========================================================================
    // Configuration Management
    //=========================================================================
    
    /**
     * @brief Get current configuration
     * 
     * @return Copy of current engine configuration
     */
    [[nodiscard]] const EngineConfig& getConfig() const { return config_; }
    
    /**
     * @brief Reload configuration from file
     * 
     * Reloads settings from the config file used during construction.
     * Only affects solver settings (capacity changes require restart).
     * 
     * @return true if reload successful
     */
    [[nodiscard]] bool reloadConfig();
    
    /**
     * @brief Reload configuration from specific file
     * 
     * @param config_path Path to .config file
     * @return true if reload successful
     */
    [[nodiscard]] bool reloadConfig(const std::string& config_path);
    
    /**
     * @brief Update configuration at runtime
     * 
     * Applies new configuration. Only solver settings can be changed
     * without reinitializing. Capacity changes require reset() and
     * re-initialization.
     * 
     * @param config New configuration
     * @return true if applied successfully
     */
    [[nodiscard]] bool updateConfig(const EngineConfig& config);
    
    /**
     * @brief Get path to loaded config file
     * 
     * @return Config file path, or empty string if using defaults
     */
    [[nodiscard]] const std::string& getConfigPath() const { return config_path_; }
    
    /**
     * @brief Save current configuration to file
     * 
     * @param filepath Output file path (uses current config path if empty)
     * @return true if saved successfully
     */
    [[nodiscard]] bool saveConfig(const std::string& filepath = "") const;
    
    /**
     * @brief Create default configuration file if it doesn't exist
     * 
     * @param filepath Output file path (uses default path if empty)
     * @return true if file created or already exists
     */
    [[nodiscard]] static bool ensureConfigFile(const std::string& filepath = "");

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
     * @brief Add a measurement (FR-02)
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
    
    /**
     * @brief Get reference to network model for advanced access
     */
    [[nodiscard]] NetworkModel& getModel() { return *model_; }
    [[nodiscard]] const NetworkModel& getModel() const { return *model_; }

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
     * @brief Update branch parameters (FR-25)
     * 
     * Updates branch impedance, transformer settings, etc.
     * Triggers model re-upload on next solve.
     * 
     * @param branch_id Branch string ID
     * @param desc New branch parameters (bus IDs ignored for update)
     * @return true if updated successfully
     */
    [[nodiscard]] bool updateBranch(const std::string& branch_id,
                                     const BranchDescriptor& desc);
    
    /**
     * @brief Update transformer tap ratio (FR-01 controllable taps)
     * 
     * @param branch_id Transformer branch ID
     * @param tap_ratio New tap ratio (e.g., 1.05 for +5%)
     * @return true if updated successfully
     */
    [[nodiscard]] bool setTransformerTap(const std::string& branch_id,
                                          Real tap_ratio);
    
    /**
     * @brief Update transformer phase shift angle
     * 
     * @param branch_id Transformer branch ID
     * @param phase_shift_rad Phase shift angle in radians
     * @return true if updated successfully
     */
    [[nodiscard]] bool setTransformerPhaseShift(const std::string& branch_id,
                                                 Real phase_shift_rad);
    
    /**
     * @brief Update branch impedance values (FR-25 impedance modification)
     * 
     * @param branch_id Branch string ID
     * @param r Series resistance [p.u.]
     * @param x Series reactance [p.u.]
     * @param b Total line charging susceptance [p.u.]
     * @param g Shunt conductance [p.u.] (default 0)
     * @return true if updated successfully
     */
    [[nodiscard]] bool setBranchImpedance(const std::string& branch_id,
                                           Real r, Real x, Real b, Real g = 0.0f);
    
    /**
     * @brief Check if model has been modified since last upload
     */
    [[nodiscard]] bool isModelModified() const;

    //=========================================================================
    // GPU Data Management (FR-06)
    //=========================================================================
    
    /**
     * @brief Upload network model to GPU
     * 
     * Transfers all static model data to GPU memory.
     * Must be called after model building and before first solve.
     * Automatically called if needed when solve() is invoked.
     * 
     * @return true on success
     */
    [[nodiscard]] bool uploadModel();
    
    /**
     * @brief Check if model is uploaded to GPU
     */
    [[nodiscard]] bool isModelUploaded() const { return model_uploaded_; }
    
    /**
     * @brief Force model re-upload on next solve
     */
    void invalidateUpload() { model_uploaded_ = false; }
    
    /**
     * @brief Get GPU memory usage in bytes
     */
    [[nodiscard]] size_t getGPUMemoryUsage() const;

    //=========================================================================
    // Telemetry Update API (FR-07)
    //=========================================================================
    
    /**
     * @brief Update all measurement values
     * 
     * Efficiently copies new measurement values to GPU using async DMA.
     * Values must be in the same order as measurements were added.
     * 
     * @param values Array of measurement values
     * @param count Number of values (must match measurement count)
     * @return true on success
     */
    [[nodiscard]] bool updateTelemetry(const Real* values, int32_t count);
    
    /**
     * @brief Update single measurement value
     * 
     * @param meas_id Measurement string ID
     * @param value New value
     * @return true on success
     */
    [[nodiscard]] bool updateMeasurement(const std::string& meas_id, Real value);

    //=========================================================================
    // Switching Device Control (FR-03, FR-05)
    //=========================================================================
    
    /**
     * @brief Queue switch status change
     * 
     * Changes are applied just before the next solve (FR-05).
     * 
     * @param sd_id Switching device string ID
     * @param new_status New status (OPEN or CLOSED)
     * @return true if device found and change queued
     */
    [[nodiscard]] bool setSwitchStatus(const std::string& sd_id, 
                                        SwitchStatus new_status);
    
    /**
     * @brief Get current switch status
     * 
     * @param sd_id Switching device string ID
     * @return Current status, or OPEN if not found
     */
    [[nodiscard]] SwitchStatus getSwitchStatus(const std::string& sd_id) const;
    
    /**
     * @brief Check if any switch changes are pending
     */
    [[nodiscard]] bool hasPendingSwitchChanges() const;
    
    /**
     * @brief Apply all pending switch changes immediately
     * 
     * Normally changes are applied automatically before solve().
     * Call this to force immediate topology update.
     * 
     * @return true on success
     */
    [[nodiscard]] bool applyPendingSwitchChanges();

    //=========================================================================
    // State Estimation (FR-08 through FR-12)
    //=========================================================================
    
    /**
     * @brief Run state estimation
     * 
     * Executes WLS algorithm with specified mode:
     * - REALTIME: Limited iterations, hot start, gain matrix reuse (FR-11)
     * - PRECISION: Full convergence, refactorization (FR-12)
     * 
     * @param mode Estimation mode
     * @return Estimation result with convergence status
     */
    [[nodiscard]] EstimationResult solve(EstimationMode mode = EstimationMode::REALTIME);
    
    /**
     * @brief Run state estimation with custom configuration
     * 
     * @param config Solver configuration
     * @return Estimation result
     */
    [[nodiscard]] EstimationResult solve(const SolverConfig& config);
    
    /**
     * @brief Run observability analysis (FR-13)
     * 
     * Checks if network is observable before solving.
     * 
     * @return true if network is observable
     */
    [[nodiscard]] bool checkObservability();
    
    /**
     * @brief Get number of unobservable islands
     */
    [[nodiscard]] int32_t getIslandCount() const;
    
    /**
     * @brief Apply flat start initialization (1.0 p.u., 0 deg)
     */
    void applyFlatStart();
    
    /**
     * @brief Apply previous solution as initial state (hot start)
     */
    void applyHotStart();

    //=========================================================================
    // Result Access (FR-10)
    //=========================================================================
    
    /**
     * @brief Get voltage magnitudes at all buses
     * 
     * @return Vector of voltage magnitudes [p.u.]
     */
    [[nodiscard]] std::vector<Real> getVoltageMagnitudes() const;
    
    /**
     * @brief Get voltage angles at all buses
     * 
     * @return Vector of voltage angles [radians]
     */
    [[nodiscard]] std::vector<Real> getVoltageAngles() const;
    
    /**
     * @brief Get voltage magnitude at specific bus
     * 
     * @param bus_id Bus string ID
     * @return Voltage magnitude [p.u.], or NaN if not found
     */
    [[nodiscard]] Real getVoltageMagnitude(const std::string& bus_id) const;
    
    /**
     * @brief Get voltage angle at specific bus
     * 
     * @param bus_id Bus string ID
     * @return Voltage angle [radians], or NaN if not found
     */
    [[nodiscard]] Real getVoltageAngle(const std::string& bus_id) const;
    
    /**
     * @brief Get active power injections at all buses
     */
    [[nodiscard]] std::vector<Real> getPowerInjections() const;
    
    /**
     * @brief Get reactive power injections at all buses
     */
    [[nodiscard]] std::vector<Real> getReactivePowerInjections() const;
    
    /**
     * @brief Get active power flow on branch
     * 
     * @param branch_id Branch string ID
     * @param end FROM or TO end
     * @return Power flow [p.u.], or NaN if not found
     */
    [[nodiscard]] Real getPowerFlow(const std::string& branch_id, 
                                     BranchEnd end = BranchEnd::FROM) const;
    
    /**
     * @brief Get reactive power flow on branch
     */
    [[nodiscard]] Real getReactivePowerFlow(const std::string& branch_id,
                                             BranchEnd end = BranchEnd::FROM) const;
    
    /**
     * @brief Get current magnitude on branch
     */
    [[nodiscard]] Real getCurrentMagnitude(const std::string& branch_id,
                                            BranchEnd end = BranchEnd::FROM) const;

    //=========================================================================
    // Residual Analysis (FR-14, FR-15)
    //=========================================================================
    
    /**
     * @brief Get measurement residuals
     * 
     * @return Vector of residuals (z - h(x))
     */
    [[nodiscard]] std::vector<Real> getResiduals() const;
    
    /**
     * @brief Get normalized residuals
     * 
     * @return Vector of normalized residuals (r / sigma)
     */
    [[nodiscard]] std::vector<Real> getNormalizedResiduals() const;
    
    /**
     * @brief Get residual for specific measurement
     * 
     * @param meas_id Measurement string ID
     * @return Residual value, or NaN if not found
     */
    [[nodiscard]] Real getResidual(const std::string& meas_id) const;
    
    /**
     * @brief Get objective function value J(x) = r^T W r
     */
    [[nodiscard]] Real getObjectiveValue() const;
    
    /**
     * @brief Identify suspected bad data points
     * 
     * Returns measurements with normalized residuals exceeding threshold.
     * 
     * @param threshold Normalized residual threshold (default 3.0)
     * @return Vector of measurement indices
     */
    [[nodiscard]] std::vector<int32_t> identifyBadData(Real threshold = 3.0f) const;

    //=========================================================================
    // Callbacks
    //=========================================================================
    
    /**
     * @brief Set progress callback
     * 
     * Called after each WLS iteration with current state.
     */
    void setProgressCallback(ProgressCallback callback);
    
    /**
     * @brief Set bad data detection callback
     * 
     * Called when bad data is detected during estimation.
     */
    void setBadDataCallback(BadDataCallback callback);

    //=========================================================================
    // Statistics and Timing
    //=========================================================================
    
    /**
     * @brief Get last solve computation time
     * 
     * @return Computation time in milliseconds
     */
    [[nodiscard]] Real getLastSolveTime() const { return last_solve_time_; }
    
    /**
     * @brief Get last result
     */
    [[nodiscard]] const EstimationResult& getLastResult() const { return last_result_; }
    
    /**
     * @brief Get number of active measurements
     */
    [[nodiscard]] int32_t getActiveMeasurementCount() const;

private:
    // Configuration
    EngineConfig config_;
    std::string config_path_;           ///< Path to loaded config file
    bool initialized_;
    bool model_uploaded_;
    
    // Network model (host side)
    std::unique_ptr<NetworkModel> model_;
    
    // GPU components
    std::unique_ptr<DeviceDataManager> device_mgr_;
    std::unique_ptr<WLSSolver> solver_;
    std::unique_ptr<TopologyProcessor> topology_;
    
    // Callbacks
    ProgressCallback progress_callback_;
    BadDataCallback bad_data_callback_;
    
    // Statistics
    Real last_solve_time_;
    EstimationResult last_result_;
    mutable int32_t island_count_;      ///< Number of islands from observability check
    
    // Internal methods
    [[nodiscard]] bool ensureInitialized();
    [[nodiscard]] bool ensureModelUploaded();
    void handleTopologyChanges();
    void copyResultsToHost();
};

//=============================================================================
// SECTION 3: Device Data Manager (FR-06)
//=============================================================================

/**
 * @class DeviceDataManager
 * @brief Manages GPU memory for network data
 * 
 * Handles:
 * - Static model data allocation and persistence (FR-06)
 * - Pinned memory for telemetry (FR-06)
 * - Async data transfer (FR-07)
 * - Ybus matrix management
 */
class DeviceDataManager {
public:
    explicit DeviceDataManager(cudaStream_t stream = nullptr);
    ~DeviceDataManager();
    
    // Disable copy
    DeviceDataManager(const DeviceDataManager&) = delete;
    DeviceDataManager& operator=(const DeviceDataManager&) = delete;

    /**
     * @brief Allocate device memory for network model
     */
    [[nodiscard]] cudaError_t allocate(
        int32_t n_buses,
        int32_t n_branches,
        int32_t n_measurements,
        int32_t n_switching_devices);
    
    /**
     * @brief Upload host model data to device
     */
    [[nodiscard]] cudaError_t upload(const NetworkModel& model);
    
    /**
     * @brief Update measurement values only (FR-07)
     */
    [[nodiscard]] cudaError_t updateMeasurements(const Real* values, int32_t count);
    
    /**
     * @brief Download results from device to host
     */
    [[nodiscard]] cudaError_t downloadResults(NetworkModel& model);
    
    /**
     * @brief Free all device memory
     */
    void free();
    
    // Data access
    DeviceBusData& getBuses() { return buses_; }
    DeviceBranchData& getBranches() { return branches_; }
    DeviceMeasurementData& getMeasurements() { return measurements_; }
    DeviceYbusMatrix& getYbus() { return ybus_; }
    
    const DeviceBusData& getBuses() const { return buses_; }
    const DeviceBranchData& getBranches() const { return branches_; }
    const DeviceMeasurementData& getMeasurements() const { return measurements_; }
    const DeviceYbusMatrix& getYbus() const { return ybus_; }
    
    /**
     * @brief Get total allocated GPU memory
     */
    [[nodiscard]] size_t getMemoryUsage() const { return total_allocated_; }

private:
    cudaStream_t stream_;
    
    DeviceBusData buses_;
    DeviceBranchData branches_;
    DeviceMeasurementData measurements_;
    DeviceYbusMatrix ybus_;
    
    // Pinned host memory for async transfers
    Real* h_pinned_measurements_;
    
    size_t total_allocated_;
    bool is_allocated_;
    
    [[nodiscard]] cudaError_t allocateBusData(int32_t count);
    [[nodiscard]] cudaError_t allocateBranchData(int32_t count);
    [[nodiscard]] cudaError_t allocateMeasurementData(int32_t count);
    void freeBusData();
    void freeBranchData();
    void freeMeasurementData();
};

//=============================================================================
// SECTION 4: Topology Processor (FR-03, FR-05)
//=============================================================================

/**
 * @class TopologyProcessor
 * @brief Handles switching device status and Ybus updates
 * 
 * Implements:
 * - Switch status vector management (FR-03)
 * - Zero-impedance branch handling (FR-04)
 * - GPU-accelerated re-topology (FR-05)
 */
class TopologyProcessor {
public:
    explicit TopologyProcessor(cudaStream_t stream = nullptr);
    ~TopologyProcessor();
    
    /**
     * @brief Initialize with device data
     */
    [[nodiscard]] cudaError_t initialize(
        DeviceBranchData& branches,
        DeviceYbusMatrix& ybus,
        int32_t n_switching_devices);
    
    /**
     * @brief Queue switch status change
     */
    void queueStatusChange(int32_t sd_index, SwitchStatus new_status);
    
    /**
     * @brief Check for pending changes
     */
    [[nodiscard]] bool hasPendingChanges() const { return !pending_changes_.empty(); }
    
    /**
     * @brief Apply all pending changes (FR-05)
     * 
     * Updates branch status and Ybus on GPU.
     */
    [[nodiscard]] cudaError_t applyChanges(
        DeviceBranchData& branches,
        DeviceYbusMatrix& ybus,
        SparseMatrixManager& matrix_mgr);
    
    /**
     * @brief Get list of affected branch indices after changes
     */
    [[nodiscard]] const std::vector<int32_t>& getAffectedBranches() const {
        return affected_branches_;
    }

private:
    cudaStream_t stream_;
    
    // Pending status changes (SD index -> new status)
    std::vector<std::pair<int32_t, SwitchStatus>> pending_changes_;
    
    // Affected branches for partial Ybus update
    std::vector<int32_t> affected_branches_;
    
    // Device-side SD data
    SwitchStatus* d_sd_status_;
    int32_t* d_sd_branch_index_;
    int32_t n_devices_;
};

} // namespace sle

#endif // SLE_ENGINE_H

