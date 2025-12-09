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
 * @file sle_api.h
 * @brief C-compatible DLL API for the SLE Engine (NFR-20)
 * 
 * This is a STANDALONE header file that can be included in any C or C++
 * project without requiring CUDA or any other SLE headers. It provides
 * the complete public interface for the SLE Engine DLL.
 * 
 * Usage:
 * @code
 * // In your application
 * #include "sle_api.h"
 * 
 * int main() {
 *     SLE_Handle engine;
 *     sle_Create(&engine);
 *     sle_Initialize(engine);
 *     // ... use the engine ...
 *     sle_Destroy(engine);
 *     return 0;
 * }
 * @endcode
 * 
 * Linking:
 * - Windows: Link against SLE.lib (import library for SLE.dll)
 * - Linux: Link against libSLE.so
 * 
 * Error handling:
 * - All functions return SLE_StatusCode
 * - Use sle_GetLastError() for detailed error messages
 * 
 * Thread safety:
 * - Each SLE_Handle represents an independent engine instance
 * - Different handles can be used from different threads
 * - Same handle must NOT be used concurrently
 * 
 * Memory management:
 * - The DLL manages all internal memory
 * - Arrays returned by functions are valid until next call or handle destruction
 * - Caller must NOT free pointers returned by the DLL
 */

#ifndef SLE_API_H
#define SLE_API_H

/* Standard C headers only - no CUDA dependencies */
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*=============================================================================
 * DLL Export/Import Macros and Calling Convention
 *===========================================================================*/

#if defined(_WIN32) || defined(_WIN64)
    /* Windows DLL export/import */
    #ifdef SLE_EXPORTS
        #define SLE_API __declspec(dllexport)
    #else
        #define SLE_API __declspec(dllimport)
    #endif
    /* Use stdcall on Windows for better compatibility with other languages */
    #define SLE_CALL __stdcall
#else
    /* Unix/Linux shared library */
    #ifdef SLE_EXPORTS
        #define SLE_API __attribute__((visibility("default")))
    #else
        #define SLE_API
    #endif
    #define SLE_CALL
#endif

/*=============================================================================
 * Version Information
 *===========================================================================*/

#define SLE_VERSION_MAJOR 0
#define SLE_VERSION_MINOR 20
#define SLE_VERSION_PATCH 0
#define SLE_VERSION_STRING "0.20.0"

/*=============================================================================
 * Basic Type Definitions
 *===========================================================================*/

/**
 * @brief Opaque handle to SLE engine instance
 * 
 * This handle is returned by sle_Create() and must be passed to all
 * subsequent API calls. Call sle_Destroy() to release resources.
 */
typedef struct SLE_EngineImpl* SLE_Handle;

/**
 * @brief Floating-point type for numerical values
 * 
 * By default, single precision (float) is used for maximum GPU performance.
 * Define SLE_USE_DOUBLE before including this header to use double precision.
 */
#ifdef SLE_USE_DOUBLE
    typedef double SLE_Real;
#else
    typedef float SLE_Real;
#endif

/*=============================================================================
 * Enumeration Types
 *===========================================================================*/

/**
 * @brief Status codes returned by all API functions
 */
typedef enum SLE_StatusCode {
    SLE_OK = 0,                      /**< Success */
    SLE_ERROR_INVALID_HANDLE = -1,   /**< NULL or invalid handle */
    SLE_ERROR_NOT_INITIALIZED = -2,  /**< Engine not initialized */
    SLE_ERROR_CUDA_FAILURE = -3,     /**< CUDA operation failed */
    SLE_ERROR_INVALID_ARGUMENT = -4, /**< Invalid function argument */
    SLE_ERROR_MODEL_INCOMPLETE = -5, /**< Model missing required elements */
    SLE_ERROR_NOT_OBSERVABLE = -6,   /**< Network not observable */
    SLE_ERROR_SINGULAR_MATRIX = -7,  /**< Gain matrix singular */
    SLE_ERROR_DIVERGED = -8,         /**< Solution diverged (NaN/Inf) */
    SLE_ERROR_MAX_ITERATIONS = -9,   /**< Max iterations reached */
    SLE_ERROR_OUT_OF_MEMORY = -10,   /**< Memory allocation failed */
    SLE_ERROR_ELEMENT_NOT_FOUND = -11, /**< Element ID not found */
    SLE_ERROR_DUPLICATE_ID = -12,    /**< Duplicate element ID */
    SLE_ERROR_CONFIG_FILE = -13,     /**< Configuration file error */
    SLE_ERROR_UNKNOWN = -99          /**< Unknown error */
} SLE_StatusCode;

/**
 * @brief Bus type classification
 */
typedef enum SLE_BusType {
    SLE_BUS_PQ = 0,      /**< Load bus (P and Q specified) */
    SLE_BUS_PV = 1,      /**< Generator bus (P and V specified) */
    SLE_BUS_SLACK = 2    /**< Reference/Swing bus (V and theta specified) */
} SLE_BusType;

/**
 * @brief Switching device status
 */
typedef enum SLE_SwitchStatus {
    SLE_SWITCH_OPEN = 0,     /**< Branch disconnected */
    SLE_SWITCH_CLOSED = 1    /**< Branch connected */
} SLE_SwitchStatus;

/**
 * @brief Measurement type classification
 */
typedef enum SLE_MeasurementType {
    SLE_MEAS_V_MAG = 0,          /**< Bus voltage magnitude */
    SLE_MEAS_V_ANGLE = 1,        /**< Bus voltage angle (PMU) */
    SLE_MEAS_P_INJECTION = 2,    /**< Bus active power injection */
    SLE_MEAS_Q_INJECTION = 3,    /**< Bus reactive power injection */
    SLE_MEAS_P_FLOW = 4,         /**< Branch active power flow */
    SLE_MEAS_Q_FLOW = 5,         /**< Branch reactive power flow */
    SLE_MEAS_I_MAG = 6,          /**< Branch current magnitude */
    SLE_MEAS_P_PSEUDO = 7,       /**< Pseudo active power measurement */
    SLE_MEAS_Q_PSEUDO = 8        /**< Pseudo reactive power measurement */
} SLE_MeasurementType;

/**
 * @brief Branch end indicator for flow measurements
 */
typedef enum SLE_BranchEnd {
    SLE_BRANCH_FROM = 0,     /**< From bus end */
    SLE_BRANCH_TO = 1        /**< To bus end */
} SLE_BranchEnd;

/**
 * @brief Estimation mode selection
 */
typedef enum SLE_EstimationMode {
    SLE_MODE_REALTIME = 0,   /**< Fast, limited iterations, hot start */
    SLE_MODE_PRECISION = 1   /**< Full convergence, refactorization */
} SLE_EstimationMode;

/**
 * @brief Convergence status of estimation
 */
typedef enum SLE_ConvergenceStatus {
    SLE_CONVERGED = 0,           /**< Solution converged within tolerance */
    SLE_MAX_ITERATIONS = 1,      /**< Iteration limit reached */
    SLE_DIVERGED = 2,            /**< NaN/Inf detected */
    SLE_SINGULAR = 3,            /**< Gain matrix singular */
    SLE_NOT_OBSERVABLE = 4,      /**< Network not observable */
    SLE_IN_PROGRESS = 5          /**< Estimation still running */
} SLE_ConvergenceStatus;

/*=============================================================================
 * Data Structures for API Input/Output
 *===========================================================================*/

/**
 * @brief Bus definition structure
 */
typedef struct SLE_BusInfo {
    const char* id;          /**< Bus string identifier (must not be NULL) */
    SLE_Real base_kv;        /**< Base voltage [kV] */
    SLE_BusType type;        /**< Bus type (PQ, PV, SLACK) */
    SLE_Real v_setpoint;     /**< Voltage setpoint [p.u.] for PV/SLACK buses */
    SLE_Real p_scheduled;    /**< Scheduled active power [p.u.] */
    SLE_Real q_scheduled;    /**< Scheduled reactive power [p.u.] */
} SLE_BusInfo;

/**
 * @brief Branch definition structure
 */
typedef struct SLE_BranchInfo {
    const char* id;          /**< Branch string identifier (must not be NULL) */
    const char* from_bus;    /**< From bus ID (must exist) */
    const char* to_bus;      /**< To bus ID (must exist) */
    SLE_Real resistance;     /**< Series resistance R [p.u.] */
    SLE_Real reactance;      /**< Series reactance X [p.u.] */
    SLE_Real susceptance;    /**< Total line charging B [p.u.] */
    SLE_Real conductance;    /**< Shunt conductance G [p.u.] (usually 0) */
    SLE_Real tap_ratio;      /**< Off-nominal tap ratio (1.0 for lines) */
    SLE_Real phase_shift;    /**< Phase shift angle [radians] */
    int32_t is_transformer;  /**< 1 if transformer, 0 if transmission line */
    const char* sd_id;       /**< Associated switching device ID (NULL if none) */
} SLE_BranchInfo;

/**
 * @brief Measurement definition structure
 */
typedef struct SLE_MeasurementInfo {
    const char* id;              /**< Measurement string identifier */
    SLE_MeasurementType type;    /**< Measurement type */
    const char* location;        /**< Bus ID (for bus meas) or Branch ID (for flow meas) */
    SLE_BranchEnd branch_end;    /**< For branch measurements: FROM or TO */
    SLE_Real sigma;              /**< Standard deviation for weighting */
    SLE_Real pt_ratio;           /**< Potential transformer ratio (1.0 if N/A) */
    SLE_Real ct_ratio;           /**< Current transformer ratio (1.0 if N/A) */
    int32_t is_pseudo;           /**< 1 if pseudo measurement, 0 otherwise */
} SLE_MeasurementInfo;

/**
 * @brief Switching device definition structure
 */
typedef struct SLE_SwitchInfo {
    const char* id;              /**< Switch string identifier */
    const char* branch_id;       /**< Associated branch ID */
    SLE_SwitchStatus status;     /**< Initial status (OPEN or CLOSED) */
} SLE_SwitchInfo;

/**
 * @brief Estimation result structure
 */
typedef struct SLE_Result {
    SLE_ConvergenceStatus status;    /**< Convergence outcome */
    int32_t iterations;              /**< Number of iterations performed */
    SLE_Real max_mismatch;           /**< Maximum state mismatch */
    SLE_Real objective;              /**< Objective function J(x) value */
    SLE_Real computation_time_ms;    /**< GPU computation time [ms] */
    SLE_Real largest_residual;       /**< Largest measurement residual */
    int32_t largest_residual_idx;    /**< Index of largest residual */
    int32_t bad_data_count;          /**< Number of suspected bad data points */
} SLE_Result;

/**
 * @brief Complete engine configuration structure
 * 
 * All values can be loaded from sle_engine.config file using sle_LoadConfigFile().
 */
typedef struct SLE_Config {
    /* Device settings */
    int32_t device_id;               /**< CUDA device ID (0 = first GPU) */
    
    /* Capacity settings (NFR-01) - pre-allocation sizes */
    size_t max_buses;                /**< Maximum number of buses */
    size_t max_branches;             /**< Maximum number of branches */
    size_t max_measurements;         /**< Maximum number of measurements */
    size_t max_switching_devices;    /**< Maximum number of switching devices */
    
    /* Solver settings (FR-11, FR-12, FR-17) */
    int32_t enable_robust;           /**< Enable Huber M-estimator (1/0) */
    SLE_Real huber_gamma;            /**< Huber threshold parameter */
    SLE_Real convergence_tolerance;  /**< WLS convergence tolerance */
    int32_t max_realtime_iterations; /**< Max iterations for real-time mode */
    int32_t max_precision_iterations;/**< Max iterations for precision mode */
    SLE_Real realtime_time_limit_ms; /**< Time limit for real-time mode [ms] */
    int32_t use_flat_start_default;  /**< Default to flat start (1/0) */
    
    /* Performance settings */
    int32_t enable_profiling;        /**< Enable CUDA event timing (1/0) */
    int32_t use_cuda_graphs;         /**< Use CUDA graphs for iteration (1/0) */
    int32_t block_size_standard;     /**< Standard kernel block size */
    int32_t block_size_reduction;    /**< Reduction kernel block size */
    
    /* Numerical settings */
    int32_t use_double_precision;    /**< Use fp64 instead of fp32 (1/0) */
    SLE_Real voltage_min_pu;         /**< Minimum voltage limit [p.u.] */
    SLE_Real voltage_max_pu;         /**< Maximum voltage limit [p.u.] */
    
    /* Bad data detection (FR-15) */
    SLE_Real bad_data_threshold;     /**< Normalized residual threshold (e.g., 3.0) */
    int32_t auto_bad_data_rejection; /**< Automatically reject bad data (1/0) */
} SLE_Config;

/**
 * @brief Solver configuration for individual solve calls
 */
typedef struct SLE_SolverConfig {
    SLE_EstimationMode mode;         /**< Estimation mode (REALTIME or PRECISION) */
    int32_t max_iterations;          /**< Maximum WLS iterations */
    SLE_Real tolerance;              /**< Convergence tolerance */
    SLE_Real huber_gamma;            /**< Huber threshold (if robust enabled) */
    int32_t use_robust;              /**< Use Huber M-estimator (1/0) */
    int32_t use_flat_start;          /**< Force flat start initialization (1/0) */
    SLE_Real time_limit_ms;          /**< Time limit for real-time mode [ms] */
} SLE_SolverConfig;

/*=============================================================================
 * Engine Lifecycle Functions
 *===========================================================================*/

/**
 * @brief Create a new SLE engine instance
 * 
 * Creates an engine using default configuration. The configuration can be
 * loaded from sle_engine.config if it exists in the current directory.
 * 
 * @param[out] handle Pointer to receive the engine handle (must not be NULL)
 * @return SLE_OK on success, error code on failure
 */
SLE_API SLE_StatusCode SLE_CALL sle_Create(SLE_Handle* handle);

/**
 * @brief Create engine with explicit configuration
 * 
 * @param[out] handle Pointer to receive the engine handle
 * @param[in] config Configuration structure (must not be NULL)
 * @return SLE_OK on success, error code on failure
 */
SLE_API SLE_StatusCode SLE_CALL sle_CreateWithConfig(SLE_Handle* handle, 
                                                      const SLE_Config* config);

/**
 * @brief Create engine from configuration file
 * 
 * @param[out] handle Pointer to receive the engine handle
 * @param[in] config_path Path to .config file (must not be NULL)
 * @return SLE_OK on success, error code on failure
 */
SLE_API SLE_StatusCode SLE_CALL sle_CreateFromConfigFile(SLE_Handle* handle,
                                                          const char* config_path);

/**
 * @brief Initialize the engine
 * 
 * Selects CUDA device and allocates initial GPU resources.
 * Must be called before any model building or solving operations.
 * 
 * @param[in] handle Engine handle
 * @return SLE_OK on success, SLE_ERROR_CUDA_FAILURE if GPU unavailable
 */
SLE_API SLE_StatusCode SLE_CALL sle_Initialize(SLE_Handle handle);

/**
 * @brief Destroy engine and release all resources
 * 
 * After calling this function, the handle is invalid and must not be used.
 * 
 * @param[in] handle Engine handle (may be NULL, in which case nothing happens)
 */
SLE_API void SLE_CALL sle_Destroy(SLE_Handle handle);

/*=============================================================================
 * Configuration Management Functions
 *===========================================================================*/

/**
 * @brief Load configuration from file
 * 
 * Parses a .config file and populates the configuration structure.
 * 
 * @param[in] config_path Path to .config file
 * @param[out] config Configuration structure to populate
 * @return SLE_OK on success, SLE_ERROR_CONFIG_FILE if file cannot be read
 */
SLE_API SLE_StatusCode SLE_CALL sle_LoadConfigFile(const char* config_path,
                                                    SLE_Config* config);

/**
 * @brief Save configuration to file
 * 
 * @param[in] config_path Output file path
 * @param[in] config Configuration to save
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_SaveConfigFile(const char* config_path,
                                                    const SLE_Config* config);

/**
 * @brief Reload configuration from file
 * 
 * Reloads settings from the config file used during engine creation.
 * Only affects solver settings; capacity changes require restart.
 * 
 * @param[in] handle Engine handle
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_ReloadConfig(SLE_Handle handle);

/**
 * @brief Reload configuration from specific file
 * 
 * @param[in] handle Engine handle
 * @param[in] config_path Path to .config file
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_ReloadConfigFromFile(SLE_Handle handle,
                                                          const char* config_path);

/**
 * @brief Get default configuration file path
 * 
 * @return Default path "sle_engine.config"
 */
SLE_API const char* SLE_CALL sle_GetDefaultConfigPath(void);

/**
 * @brief Get path to loaded configuration file
 * 
 * @param[in] handle Engine handle
 * @return Config file path, or empty string if using defaults
 */
SLE_API const char* SLE_CALL sle_GetConfigPath(SLE_Handle handle);

/**
 * @brief Create default configuration file if it doesn't exist
 * 
 * @param[in] config_path Output file path (NULL for default path)
 * @return SLE_OK if file created or already exists
 */
SLE_API SLE_StatusCode SLE_CALL sle_EnsureConfigFile(const char* config_path);

/**
 * @brief Get current engine configuration
 * 
 * @param[in] handle Engine handle
 * @param[out] config Configuration structure to populate
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_GetConfig(SLE_Handle handle, SLE_Config* config);

/**
 * @brief Update engine configuration at runtime
 * 
 * Only solver settings can be changed without reinitializing.
 * 
 * @param[in] handle Engine handle
 * @param[in] config New configuration
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_SetConfig(SLE_Handle handle, const SLE_Config* config);

/*=============================================================================
 * Error Handling Functions
 *===========================================================================*/

/**
 * @brief Get last error message
 * 
 * @param[in] handle Engine handle
 * @return Error message string (valid until next API call on same handle)
 */
SLE_API const char* SLE_CALL sle_GetLastError(SLE_Handle handle);

/**
 * @brief Get CUDA device information string
 * 
 * @param[in] handle Engine handle
 * @return Device info string (name, compute capability, memory)
 */
SLE_API const char* SLE_CALL sle_GetDeviceInfo(SLE_Handle handle);

/*=============================================================================
 * Model Building Functions (FR-01)
 *===========================================================================*/

/**
 * @brief Add a bus to the model
 * 
 * @param[in] handle Engine handle
 * @param[in] info Bus definition (id must be unique)
 * @return SLE_OK on success, SLE_ERROR_DUPLICATE_ID if bus ID already exists
 */
SLE_API SLE_StatusCode SLE_CALL sle_AddBus(SLE_Handle handle, const SLE_BusInfo* info);

/**
 * @brief Add multiple buses at once (batch operation)
 * 
 * @param[in] handle Engine handle
 * @param[in] buses Array of bus definitions
 * @param[in] count Number of buses in array
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_AddBuses(SLE_Handle handle, 
                                              const SLE_BusInfo* buses, 
                                              int32_t count);

/**
 * @brief Add a branch to the model
 * 
 * @param[in] handle Engine handle
 * @param[in] info Branch definition (referenced buses must exist)
 * @return SLE_OK on success, SLE_ERROR_ELEMENT_NOT_FOUND if buses don't exist
 */
SLE_API SLE_StatusCode SLE_CALL sle_AddBranch(SLE_Handle handle, const SLE_BranchInfo* info);

/**
 * @brief Add multiple branches at once (batch operation)
 */
SLE_API SLE_StatusCode SLE_CALL sle_AddBranches(SLE_Handle handle,
                                                 const SLE_BranchInfo* branches,
                                                 int32_t count);

/**
 * @brief Add a measurement to the model (FR-02)
 * 
 * @param[in] handle Engine handle
 * @param[in] info Measurement definition
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_AddMeasurement(SLE_Handle handle, 
                                                    const SLE_MeasurementInfo* info);

/**
 * @brief Add multiple measurements at once (batch operation)
 */
SLE_API SLE_StatusCode SLE_CALL sle_AddMeasurements(SLE_Handle handle,
                                                     const SLE_MeasurementInfo* meas,
                                                     int32_t count);

/**
 * @brief Add a switching device (FR-03)
 * 
 * @param[in] handle Engine handle
 * @param[in] info Switch definition
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_AddSwitch(SLE_Handle handle, const SLE_SwitchInfo* info);

/**
 * @brief Remove a bus and all connected elements (FR-25)
 * 
 * Also removes branches connected to this bus and measurements at this bus.
 * 
 * @param[in] handle Engine handle
 * @param[in] bus_id Bus string ID
 * @return SLE_OK on success, SLE_ERROR_ELEMENT_NOT_FOUND if bus doesn't exist
 */
SLE_API SLE_StatusCode SLE_CALL sle_RemoveBus(SLE_Handle handle, const char* bus_id);

/**
 * @brief Remove a branch (FR-25)
 * 
 * @param[in] handle Engine handle
 * @param[in] branch_id Branch string ID
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_RemoveBranch(SLE_Handle handle, const char* branch_id);

/**
 * @brief Update branch parameters (FR-25)
 * 
 * Updates all branch parameters. For partial updates, use specific functions below.
 * Triggers model re-upload on next solve.
 * 
 * @param[in] handle Engine handle
 * @param[in] info Updated branch definition (id must match existing branch)
 * @return SLE_OK on success, SLE_ERROR_ELEMENT_NOT_FOUND if branch doesn't exist
 */
SLE_API SLE_StatusCode SLE_CALL sle_UpdateBranch(SLE_Handle handle, const SLE_BranchInfo* info);

/**
 * @brief Update transformer tap ratio (FR-01 controllable taps)
 * 
 * Modifies the off-nominal tap ratio of a transformer branch.
 * Valid range typically 0.9 to 1.1 (±10%).
 * 
 * @param[in] handle Engine handle
 * @param[in] branch_id Transformer branch string ID
 * @param[in] tap_ratio New tap ratio (1.0 = nominal, 1.05 = +5%)
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_SetTransformerTap(SLE_Handle handle,
                                                       const char* branch_id,
                                                       SLE_Real tap_ratio);

/**
 * @brief Update transformer phase shift angle
 * 
 * Modifies the phase shift angle of a phase-shifting transformer.
 * 
 * @param[in] handle Engine handle
 * @param[in] branch_id Transformer branch string ID
 * @param[in] phase_shift Phase shift angle in radians
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_SetTransformerPhaseShift(SLE_Handle handle,
                                                              const char* branch_id,
                                                              SLE_Real phase_shift);

/**
 * @brief Update branch impedance parameters (FR-25 impedance modification)
 * 
 * Modifies the impedance values of any branch (line or transformer).
 * Typically used for:
 * - Correcting model errors
 * - Temperature-dependent resistance adjustments
 * - Transformer impedance updates
 * 
 * @param[in] handle Engine handle
 * @param[in] branch_id Branch string ID
 * @param[in] r Series resistance [p.u.]
 * @param[in] x Series reactance [p.u.]
 * @param[in] b Total line charging susceptance [p.u.]
 * @param[in] g Shunt conductance [p.u.] (usually 0)
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_SetBranchImpedance(SLE_Handle handle,
                                                        const char* branch_id,
                                                        SLE_Real r,
                                                        SLE_Real x,
                                                        SLE_Real b,
                                                        SLE_Real g);

/**
 * @brief Get transformer tap ratio
 * 
 * @param[in] handle Engine handle
 * @param[in] branch_id Transformer branch string ID
 * @return Current tap ratio, or 1.0 if not found
 */
SLE_API SLE_Real SLE_CALL sle_GetTransformerTap(SLE_Handle handle, const char* branch_id);

/**
 * @brief Get transformer phase shift angle
 * 
 * @param[in] handle Engine handle
 * @param[in] branch_id Transformer branch string ID
 * @return Phase shift angle in radians, or 0.0 if not found
 */
SLE_API SLE_Real SLE_CALL sle_GetTransformerPhaseShift(SLE_Handle handle, const char* branch_id);

/*=============================================================================
 * GPU Data Management (FR-06, FR-07)
 *===========================================================================*/

/**
 * @brief Upload network model to GPU
 * 
 * Transfers all static model data to GPU memory.
 * Must be called after model building and before first solve.
 * Automatically called by sle_Solve() if needed.
 * 
 * @param[in] handle Engine handle
 * @return SLE_OK on success, SLE_ERROR_MODEL_INCOMPLETE if validation fails
 */
SLE_API SLE_StatusCode SLE_CALL sle_UploadModel(SLE_Handle handle);

/**
 * @brief Update measurement values (FR-07 - high-speed telemetry update)
 * 
 * Efficiently transfers new measurement values to GPU using async DMA.
 * Values must be in the same order as measurements were added.
 * 
 * @param[in] handle Engine handle
 * @param[in] values Array of measurement values
 * @param[in] count Number of values (must match measurement count)
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_UpdateTelemetry(SLE_Handle handle,
                                                     const SLE_Real* values,
                                                     int32_t count);

/**
 * @brief Update single measurement value by ID
 * 
 * @param[in] handle Engine handle
 * @param[in] meas_id Measurement string ID
 * @param[in] value New measurement value
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_UpdateMeasurement(SLE_Handle handle,
                                                       const char* meas_id,
                                                       SLE_Real value);

/*=============================================================================
 * Switch Control (FR-03, FR-05)
 *===========================================================================*/

/**
 * @brief Queue switch status change
 * 
 * Changes are applied just before the next solve (FR-05).
 * 
 * @param[in] handle Engine handle
 * @param[in] sd_id Switching device string ID
 * @param[in] status New status (OPEN or CLOSED)
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_SetSwitchStatus(SLE_Handle handle,
                                                     const char* sd_id,
                                                     SLE_SwitchStatus status);

/**
 * @brief Get current switch status
 * 
 * @param[in] handle Engine handle
 * @param[in] sd_id Switching device string ID
 * @return Current status, or SLE_SWITCH_OPEN if not found
 */
SLE_API SLE_SwitchStatus SLE_CALL sle_GetSwitchStatus(SLE_Handle handle, const char* sd_id);

/**
 * @brief Apply all pending switch changes immediately (FR-05)
 * 
 * Normally changes are applied automatically before solve().
 * 
 * @param[in] handle Engine handle
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_ApplySwitchChanges(SLE_Handle handle);

/*=============================================================================
 * State Estimation (FR-08 through FR-12)
 *===========================================================================*/

/**
 * @brief Run state estimation
 * 
 * Executes WLS algorithm with specified mode:
 * - SLE_MODE_REALTIME: Limited iterations, hot start, gain matrix reuse (FR-11)
 * - SLE_MODE_PRECISION: Full convergence, refactorization (FR-12)
 * 
 * @param[in] handle Engine handle
 * @param[in] mode Estimation mode
 * @param[out] result Result structure (may be NULL if not needed)
 * @return SLE_OK if converged, otherwise error code indicating outcome
 */
SLE_API SLE_StatusCode SLE_CALL sle_Solve(SLE_Handle handle,
                                           SLE_EstimationMode mode,
                                           SLE_Result* result);

/**
 * @brief Run state estimation with custom configuration
 * 
 * @param[in] handle Engine handle
 * @param[in] config Solver configuration
 * @param[out] result Result structure (may be NULL)
 * @return SLE_OK if converged
 */
SLE_API SLE_StatusCode SLE_CALL sle_SolveWithConfig(SLE_Handle handle,
                                                     const SLE_SolverConfig* config,
                                                     SLE_Result* result);

/**
 * @brief Check network observability (FR-13)
 * 
 * Determines if the network is mathematically solvable given
 * current topology and available measurements.
 * 
 * @param[in] handle Engine handle
 * @param[out] is_observable Set to 1 if observable, 0 otherwise (may be NULL)
 * @param[out] island_count Number of connected islands (may be NULL)
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_CheckObservability(SLE_Handle handle,
                                                        int32_t* is_observable,
                                                        int32_t* island_count);

/**
 * @brief Apply flat start initialization (V=1.0 p.u., theta=0)
 * 
 * @param[in] handle Engine handle
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_ApplyFlatStart(SLE_Handle handle);

/*=============================================================================
 * Result Access (FR-10)
 *===========================================================================*/

/**
 * @brief Get voltage magnitudes at all buses
 * 
 * @param[in] handle Engine handle
 * @param[out] values Pre-allocated array to receive values [count]
 * @param[in] count Number of buses (array size)
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_GetVoltageMagnitudes(SLE_Handle handle,
                                                          SLE_Real* values,
                                                          int32_t count);

/**
 * @brief Get voltage angles at all buses
 * 
 * @param[in] handle Engine handle
 * @param[out] values Pre-allocated array [count], angles in radians
 * @param[in] count Number of buses
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_GetVoltageAngles(SLE_Handle handle,
                                                      SLE_Real* values,
                                                      int32_t count);

/**
 * @brief Get voltage magnitude at specific bus
 * 
 * @param[in] handle Engine handle
 * @param[in] bus_id Bus string ID
 * @return Voltage magnitude [p.u.], or 0.0 if not found
 */
SLE_API SLE_Real SLE_CALL sle_GetBusVoltage(SLE_Handle handle, const char* bus_id);

/**
 * @brief Get voltage angle at specific bus
 * 
 * @param[in] handle Engine handle
 * @param[in] bus_id Bus string ID
 * @return Voltage angle [radians], or 0.0 if not found
 */
SLE_API SLE_Real SLE_CALL sle_GetBusAngle(SLE_Handle handle, const char* bus_id);

/**
 * @brief Get power injections at all buses
 * 
 * @param[in] handle Engine handle
 * @param[out] p_values Active power injections [p.u.] (may be NULL)
 * @param[out] q_values Reactive power injections [p.u.] (may be NULL)
 * @param[in] count Number of buses
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_GetPowerInjections(SLE_Handle handle,
                                                        SLE_Real* p_values,
                                                        SLE_Real* q_values,
                                                        int32_t count);

/**
 * @brief Get branch power flows
 * 
 * @param[in] handle Engine handle
 * @param[out] p_from Active power at from end [count] (may be NULL)
 * @param[out] q_from Reactive power at from end (may be NULL)
 * @param[out] p_to Active power at to end (may be NULL)
 * @param[out] q_to Reactive power at to end (may be NULL)
 * @param[in] count Number of branches
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_GetBranchFlows(SLE_Handle handle,
                                                    SLE_Real* p_from,
                                                    SLE_Real* q_from,
                                                    SLE_Real* p_to,
                                                    SLE_Real* q_to,
                                                    int32_t count);

/**
 * @brief Get active power flow on specific branch
 */
SLE_API SLE_Real SLE_CALL sle_GetBranchP(SLE_Handle handle, 
                                          const char* branch_id,
                                          SLE_BranchEnd end);

/**
 * @brief Get reactive power flow on specific branch
 */
SLE_API SLE_Real SLE_CALL sle_GetBranchQ(SLE_Handle handle,
                                          const char* branch_id,
                                          SLE_BranchEnd end);

/**
 * @brief Get measurement residuals (FR-10)
 * 
 * Returns r = z_measured - h(x_estimated) for each measurement.
 * 
 * @param[in] handle Engine handle
 * @param[out] values Pre-allocated array [count]
 * @param[in] count Number of measurements
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_GetResiduals(SLE_Handle handle,
                                                  SLE_Real* values,
                                                  int32_t count);

/**
 * @brief Get objective function value J(x)
 * 
 * @param[in] handle Engine handle
 * @return Weighted sum of squared residuals
 */
SLE_API SLE_Real SLE_CALL sle_GetObjective(SLE_Handle handle);

/**
 * @brief Identify suspected bad data points (FR-15)
 * 
 * Returns indices of measurements with normalized residuals exceeding threshold.
 * 
 * @param[in] handle Engine handle
 * @param[in] threshold Normalized residual threshold (e.g., 3.0 for 3-sigma)
 * @param[out] indices Pre-allocated array to receive bad data indices
 * @param[in] max_count Maximum number of indices to return (array size)
 * @param[out] actual_count Actual number of bad data points found
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_IdentifyBadData(SLE_Handle handle,
                                                     SLE_Real threshold,
                                                     int32_t* indices,
                                                     int32_t max_count,
                                                     int32_t* actual_count);

/*=============================================================================
 * Model Information Functions
 *===========================================================================*/

/**
 * @brief Get number of buses in model
 */
SLE_API int32_t SLE_CALL sle_GetBusCount(SLE_Handle handle);

/**
 * @brief Get number of branches in model
 */
SLE_API int32_t SLE_CALL sle_GetBranchCount(SLE_Handle handle);

/**
 * @brief Get number of measurements in model
 */
SLE_API int32_t SLE_CALL sle_GetMeasurementCount(SLE_Handle handle);

/**
 * @brief Get GPU memory usage in bytes
 */
SLE_API size_t SLE_CALL sle_GetGPUMemoryUsage(SLE_Handle handle);

/**
 * @brief Get last solve computation time
 * 
 * @return Time in milliseconds
 */
SLE_API SLE_Real SLE_CALL sle_GetLastSolveTime(SLE_Handle handle);

/*=============================================================================
 * Test Case Functions (FR-18)
 *===========================================================================*/

/**
 * @brief Load IEEE 14-bus test case
 * 
 * Populates the engine with the standard IEEE 14-bus test system
 * including buses, branches, and measurements.
 * 
 * @param[in] handle Engine handle
 * @return SLE_OK on success
 */
SLE_API SLE_StatusCode SLE_CALL sle_LoadIEEE14Bus(SLE_Handle handle);

/*=============================================================================
 * Version and System Information
 *===========================================================================*/

/**
 * @brief Get SLE library version string
 * 
 * @return Version string (e.g., "0.20.0")
 */
SLE_API const char* SLE_CALL sle_GetVersion(void);

/**
 * @brief Get CUDA runtime version
 * 
 * @return CUDA version as integer (e.g., 12060 for CUDA 12.6)
 */
SLE_API int32_t SLE_CALL sle_GetCUDAVersion(void);

/**
 * @brief Check if GPU is available
 * 
 * @return 1 if CUDA-capable GPU is available, 0 otherwise
 */
SLE_API int32_t SLE_CALL sle_IsGPUAvailable(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SLE_API_H */
