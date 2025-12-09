# SLE Engine API Reference

Complete reference for the C-compatible DLL API (`sle_api.h`).

## Table of Contents

- [Overview](#overview)
- [Data Types](#data-types)
- [Engine Lifecycle](#engine-lifecycle)
- [Configuration Management](#configuration-management)
- [Model Building](#model-building)
- [Model Modification](#model-modification)
- [GPU Data Management](#gpu-data-management)
- [Switch Control](#switch-control)
- [State Estimation](#state-estimation)
- [Result Access](#result-access)
- [Error Handling](#error-handling)

## Overview

The SLE Engine provides a C-compatible API exported from a Windows DLL. All functions return `SLE_StatusCode` indicating success or error type.

### Header File

```c
#include "sle_api.h"
```

### Linking

- **Windows**: Link against `SLE.lib` (import library)
- **Linux**: Link against `libSLE.so`

### Thread Safety

- Each `SLE_Handle` represents an independent engine instance
- Different handles can be used from different threads
- The same handle must NOT be used concurrently

## Data Types

### Basic Types

```c
typedef struct SLE_EngineImpl* SLE_Handle;  // Opaque engine handle

#ifdef SLE_USE_DOUBLE
    typedef double SLE_Real;
#else
    typedef float SLE_Real;  // Default: single precision
#endif
```

### Status Codes

```c
typedef enum SLE_StatusCode {
    SLE_OK = 0,                      // Success
    SLE_ERROR_INVALID_HANDLE = -1,   // NULL or invalid handle
    SLE_ERROR_NOT_INITIALIZED = -2,  // Engine not initialized
    SLE_ERROR_CUDA_FAILURE = -3,     // CUDA operation failed
    SLE_ERROR_INVALID_ARGUMENT = -4, // Invalid function argument
    SLE_ERROR_MODEL_INCOMPLETE = -5, // Model missing required elements
    SLE_ERROR_NOT_OBSERVABLE = -6,   // Network not observable
    SLE_ERROR_SINGULAR_MATRIX = -7,  // Gain matrix singular
    SLE_ERROR_DIVERGED = -8,         // Solution diverged (NaN/Inf)
    SLE_ERROR_MAX_ITERATIONS = -9,   // Max iterations reached
    SLE_ERROR_OUT_OF_MEMORY = -10,   // Memory allocation failed
    SLE_ERROR_ELEMENT_NOT_FOUND = -11, // Element ID not found
    SLE_ERROR_DUPLICATE_ID = -12,    // Duplicate element ID
    SLE_ERROR_CONFIG_FILE = -13,     // Configuration file error
    SLE_ERROR_UNKNOWN = -99          // Unknown error
} SLE_StatusCode;
```

### Enumerations

```c
// Bus types
typedef enum SLE_BusType {
    SLE_BUS_PQ = 0,      // Load bus (P and Q specified)
    SLE_BUS_PV = 1,      // Generator bus (P and V specified)
    SLE_BUS_SLACK = 2    // Reference/Swing bus (V and theta specified)
} SLE_BusType;

// Switch status
typedef enum SLE_SwitchStatus {
    SLE_SWITCH_OPEN = 0,     // Branch disconnected
    SLE_SWITCH_CLOSED = 1    // Branch connected
} SLE_SwitchStatus;

// Measurement types
typedef enum SLE_MeasurementType {
    SLE_MEAS_V_MAG = 0,          // Bus voltage magnitude
    SLE_MEAS_V_ANGLE = 1,        // Bus voltage angle (PMU)
    SLE_MEAS_P_INJECTION = 2,    // Bus active power injection
    SLE_MEAS_Q_INJECTION = 3,    // Bus reactive power injection
    SLE_MEAS_P_FLOW = 4,         // Branch active power flow
    SLE_MEAS_Q_FLOW = 5,         // Branch reactive power flow
    SLE_MEAS_I_MAG = 6,          // Branch current magnitude
    SLE_MEAS_P_PSEUDO = 7,       // Pseudo active power
    SLE_MEAS_Q_PSEUDO = 8        // Pseudo reactive power
} SLE_MeasurementType;

// Estimation modes
typedef enum SLE_EstimationMode {
    SLE_MODE_REALTIME = 0,   // Fast, limited iterations, hot start
    SLE_MODE_PRECISION = 1   // Full convergence, refactorization
} SLE_EstimationMode;
```

### Structures

#### SLE_BusInfo

```c
typedef struct SLE_BusInfo {
    const char* id;          // Bus string identifier (must not be NULL)
    SLE_Real base_kv;       // Base voltage [kV]
    SLE_BusType type;       // Bus type (PQ, PV, SLACK)
    SLE_Real v_setpoint;    // Voltage setpoint [p.u.] for PV/SLACK buses
    SLE_Real p_scheduled;   // Scheduled active power [p.u.]
    SLE_Real q_scheduled;   // Scheduled reactive power [p.u.]
} SLE_BusInfo;
```

#### SLE_BranchInfo

```c
typedef struct SLE_BranchInfo {
    const char* id;          // Branch string identifier
    const char* from_bus;    // From bus ID (must exist)
    const char* to_bus;      // To bus ID (must exist)
    SLE_Real resistance;     // Series resistance R [p.u.]
    SLE_Real reactance;     // Series reactance X [p.u.]
    SLE_Real susceptance;   // Total line charging B [p.u.]
    SLE_Real conductance;   // Shunt conductance G [p.u.] (usually 0)
    SLE_Real tap_ratio;     // Off-nominal tap ratio (1.0 for lines)
    SLE_Real phase_shift;    // Phase shift angle [radians]
    int32_t is_transformer;  // 1 if transformer, 0 if transmission line
    const char* sd_id;       // Associated switching device ID (NULL if none)
} SLE_BranchInfo;
```

#### SLE_MeasurementInfo

```c
typedef struct SLE_MeasurementInfo {
    const char* id;              // Measurement string identifier
    SLE_MeasurementType type;   // Measurement type
    const char* location;        // Bus ID (for bus meas) or Branch ID (for flow meas)
    SLE_BranchEnd branch_end;    // For branch measurements: FROM or TO
    SLE_Real sigma;             // Standard deviation for weighting
    SLE_Real pt_ratio;          // Potential transformer ratio (1.0 if N/A)
    SLE_Real ct_ratio;          // Current transformer ratio (1.0 if N/A)
    int32_t is_pseudo;          // 1 if pseudo measurement, 0 otherwise
} SLE_MeasurementInfo;
```

#### SLE_Result

```c
typedef struct SLE_Result {
    SLE_ConvergenceStatus status;    // Convergence outcome
    int32_t iterations;              // Number of iterations performed
    SLE_Real max_mismatch;           // Maximum state mismatch
    SLE_Real objective;              // Objective function J(x) value
    SLE_Real computation_time_ms;    // GPU computation time [ms]
    SLE_Real largest_residual;       // Largest measurement residual
    int32_t largest_residual_idx;    // Index of largest residual
    int32_t bad_data_count;          // Number of suspected bad data points
} SLE_Result;
```

## Engine Lifecycle

### sle_Create

Creates a new engine instance with default configuration.

```c
SLE_StatusCode sle_Create(SLE_Handle* handle);
```

**Parameters:**
- `handle` - [out] Pointer to receive the engine handle

**Returns:**
- `SLE_OK` on success
- `SLE_ERROR_INVALID_ARGUMENT` if handle is NULL
- `SLE_ERROR_OUT_OF_MEMORY` on allocation failure

**Example:**
```c
SLE_Handle engine;
if (sle_Create(&engine) != SLE_OK) {
    // Handle error
}
```

### sle_CreateWithConfig

Creates engine with explicit configuration.

```c
SLE_StatusCode sle_CreateWithConfig(SLE_Handle* handle, const SLE_Config* config);
```

### sle_CreateFromConfigFile

Creates engine from configuration file.

```c
SLE_StatusCode sle_CreateFromConfigFile(SLE_Handle* handle, const char* config_path);
```

### sle_Initialize

Initializes CUDA device and allocates GPU resources.

```c
SLE_StatusCode sle_Initialize(SLE_Handle handle);
```

**Returns:**
- `SLE_OK` on success
- `SLE_ERROR_CUDA_FAILURE` if GPU unavailable

**Note:** Must be called before any model building or solving operations.

### sle_Destroy

Destroys engine and releases all resources.

```c
void sle_Destroy(SLE_Handle handle);
```

**Note:** Handle becomes invalid after this call. Safe to call with NULL handle.

## Configuration Management

### sle_LoadConfigFile

Loads configuration from `.config` file.

```c
SLE_StatusCode sle_LoadConfigFile(const char* config_path, SLE_Config* config);
```

### sle_SaveConfigFile

Saves configuration to file.

```c
SLE_StatusCode sle_SaveConfigFile(const char* config_path, const SLE_Config* config);
```

### sle_GetConfig / sle_SetConfig

Gets/sets current engine configuration.

```c
SLE_StatusCode sle_GetConfig(SLE_Handle handle, SLE_Config* config);
SLE_StatusCode sle_SetConfig(SLE_Handle handle, const SLE_Config* config);
```

**Note:** Only solver settings can be changed at runtime. Capacity changes require restart.

## Model Building

### sle_AddBus

Adds a bus to the network model.

```c
SLE_StatusCode sle_AddBus(SLE_Handle handle, const SLE_BusInfo* info);
```

**Returns:**
- `SLE_OK` on success
- `SLE_ERROR_DUPLICATE_ID` if bus ID already exists

### sle_AddBranch

Adds a branch (line or transformer) to the network.

```c
SLE_StatusCode sle_AddBranch(SLE_Handle handle, const SLE_BranchInfo* info);
```

**Returns:**
- `SLE_OK` on success
- `SLE_ERROR_ELEMENT_NOT_FOUND` if referenced buses don't exist

### sle_AddMeasurement

Adds a measurement to the model.

```c
SLE_StatusCode sle_AddMeasurement(SLE_Handle handle, const SLE_MeasurementInfo* info);
```

**Note:** Measurements must be added in the order they will appear in telemetry vectors.

### sle_AddSwitch

Adds a switching device (breaker, switch, fuse).

```c
SLE_StatusCode sle_AddSwitch(SLE_Handle handle, const SLE_SwitchInfo* info);
```

### Batch Operations

```c
SLE_StatusCode sle_AddBuses(SLE_Handle handle, const SLE_BusInfo* buses, int32_t count);
SLE_StatusCode sle_AddBranches(SLE_Handle handle, const SLE_BranchInfo* branches, int32_t count);
SLE_StatusCode sle_AddMeasurements(SLE_Handle handle, const SLE_MeasurementInfo* meas, int32_t count);
```

## Model Modification

### sle_UpdateBranch

Updates all branch parameters.

```c
SLE_StatusCode sle_UpdateBranch(SLE_Handle handle, const SLE_BranchInfo* info);
```

**Note:** Triggers model re-upload on next solve.

### sle_SetTransformerTap

Updates transformer tap ratio (controllable taps).

```c
SLE_StatusCode sle_SetTransformerTap(SLE_Handle handle, const char* branch_id, SLE_Real tap_ratio);
```

**Example:**
```c
// Set tap to +5% (1.05)
sle_SetTransformerTap(engine, "Xfmr4-7", 1.05f);
```

### sle_SetTransformerPhaseShift

Updates transformer phase shift angle.

```c
SLE_StatusCode sle_SetTransformerPhaseShift(SLE_Handle handle, const char* branch_id, SLE_Real phase_shift);
```

### sle_SetBranchImpedance

Updates branch impedance values (R, X, B, G).

```c
SLE_StatusCode sle_SetBranchImpedance(SLE_Handle handle, const char* branch_id,
                                      SLE_Real r, SLE_Real x, SLE_Real b, SLE_Real g);
```

### sle_RemoveBus / sle_RemoveBranch

Removes elements from the model.

```c
SLE_StatusCode sle_RemoveBus(SLE_Handle handle, const char* bus_id);
SLE_StatusCode sle_RemoveBranch(SLE_Handle handle, const char* branch_id);
```

## GPU Data Management

### sle_UploadModel

Uploads network model to GPU memory.

```c
SLE_StatusCode sle_UploadModel(SLE_Handle handle);
```

**Note:** Must be called after model building and before first solve. Automatically called by `sle_Solve()` if needed.

### sle_UpdateTelemetry

Updates measurement values (high-speed telemetry update).

```c
SLE_StatusCode sle_UpdateTelemetry(SLE_Handle handle, const SLE_Real* values, int32_t count);
```

**Parameters:**
- `values` - Array of measurement values (must match measurement count)
- `count` - Number of values

**Note:** Values must be in the same order as measurements were added. Uses async DMA for efficient transfer.

### sle_UpdateMeasurement

Updates single measurement value by ID.

```c
SLE_StatusCode sle_UpdateMeasurement(SLE_Handle handle, const char* meas_id, SLE_Real value);
```

## Switch Control

### sle_SetSwitchStatus

Queues switch status change.

```c
SLE_StatusCode sle_SetSwitchStatus(SLE_Handle handle, const char* sd_id, SLE_SwitchStatus status);
```

**Note:** Changes are applied just before the next solve (FR-05).

### sle_GetSwitchStatus

Gets current switch status.

```c
SLE_SwitchStatus sle_GetSwitchStatus(SLE_Handle handle, const char* sd_id);
```

### sle_ApplySwitchChanges

Applies all pending switch changes immediately.

```c
SLE_StatusCode sle_ApplySwitchChanges(SLE_Handle handle);
```

**Note:** Normally changes are applied automatically before solve().

## State Estimation

### sle_Solve

Runs state estimation with specified mode.

```c
SLE_StatusCode sle_Solve(SLE_Handle handle, SLE_EstimationMode mode, SLE_Result* result);
```

**Modes:**
- `SLE_MODE_REALTIME`: Limited iterations, hot start, gain matrix reuse (FR-11)
- `SLE_MODE_PRECISION`: Full convergence, refactorization (FR-12)

**Returns:**
- `SLE_OK` if converged
- `SLE_ERROR_MAX_ITERATIONS` if iteration limit reached
- `SLE_ERROR_DIVERGED` if solution diverged
- `SLE_ERROR_SINGULAR_MATRIX` if gain matrix singular
- `SLE_ERROR_NOT_OBSERVABLE` if network not observable

### sle_SolveWithConfig

Runs estimation with custom solver configuration.

```c
SLE_StatusCode sle_SolveWithConfig(SLE_Handle handle, const SLE_SolverConfig* config, SLE_Result* result);
```

### sle_CheckObservability

Checks network observability before solving.

```c
SLE_StatusCode sle_CheckObservability(SLE_Handle handle, int32_t* is_observable, int32_t* island_count);
```

### sle_ApplyFlatStart

Applies flat start initialization (V=1.0 p.u., theta=0).

```c
SLE_StatusCode sle_ApplyFlatStart(SLE_Handle handle);
```

## Result Access

### Voltage Results

```c
SLE_StatusCode sle_GetVoltageMagnitudes(SLE_Handle handle, SLE_Real* values, int32_t count);
SLE_StatusCode sle_GetVoltageAngles(SLE_Handle handle, SLE_Real* values, int32_t count);
SLE_Real sle_GetBusVoltage(SLE_Handle handle, const char* bus_id);
SLE_Real sle_GetBusAngle(SLE_Handle handle, const char* bus_id);
```

### Power Results

```c
SLE_StatusCode sle_GetPowerInjections(SLE_Handle handle, SLE_Real* p_values, SLE_Real* q_values, int32_t count);
SLE_StatusCode sle_GetBranchFlows(SLE_Handle handle, SLE_Real* p_from, SLE_Real* q_from,
                                  SLE_Real* p_to, SLE_Real* q_to, int32_t count);
SLE_Real sle_GetBranchP(SLE_Handle handle, const char* branch_id, SLE_BranchEnd end);
SLE_Real sle_GetBranchQ(SLE_Handle handle, const char* branch_id, SLE_BranchEnd end);
```

### Residuals and Bad Data

```c
SLE_StatusCode sle_GetResiduals(SLE_Handle handle, SLE_Real* values, int32_t count);
SLE_Real sle_GetObjective(SLE_Handle handle);
SLE_StatusCode sle_IdentifyBadData(SLE_Handle handle, SLE_Real threshold, int32_t* indices,
                                   int32_t max_count, int32_t* actual_count);
```

## Error Handling

### sle_GetLastError

Gets last error message.

```c
const char* sle_GetLastError(SLE_Handle handle);
```

**Note:** Error messages are thread-local, not per-handle.

### sle_GetDeviceInfo

Gets CUDA device information string.

```c
const char* sle_GetDeviceInfo(SLE_Handle handle);
```

## Version Information

```c
const char* sle_GetVersion(void);        // Returns "0.20.0"
int32_t sle_GetCUDAVersion(void);        // CUDA runtime version
int32_t sle_IsGPUAvailable(void);        // 1 if GPU available, 0 otherwise
```

## Test Case Functions

### sle_LoadIEEE14Bus

Loads the standard IEEE 14-Bus test case.

```c
SLE_StatusCode sle_LoadIEEE14Bus(SLE_Handle handle);
```

Populates the engine with buses, branches, and measurements for the IEEE 14-Bus system.

## Complete Example

See `examples/example_usage.cpp` for a comprehensive example demonstrating all API features.

