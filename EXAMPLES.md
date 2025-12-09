# SLE Engine Examples

Usage examples and tutorials for the SLE Engine API.

## Table of Contents

- [Quick Start Example](#quick-start-example)
- [Basic Usage](#basic-usage)
- [Model Building](#model-building)
- [State Estimation](#state-estimation)
- [Topology Changes](#topology-changes)
- [Transformer Control](#transformer-control)
- [Bad Data Detection](#bad-data-detection)
- [Complete Example](#complete-example)

## Quick Start Example

Minimal example demonstrating engine creation and basic usage:

```c
#include "sle_api.h"
#include <stdio.h>

int main() {
    SLE_Handle engine;
    SLE_StatusCode status;
    
    // Create engine
    status = sle_Create(&engine);
    if (status != SLE_OK) {
        printf("Failed to create engine\n");
        return 1;
    }
    
    // Initialize CUDA
    status = sle_Initialize(engine);
    if (status != SLE_OK) {
        printf("Failed to initialize: %s\n", sle_GetLastError(engine));
        sle_Destroy(engine);
        return 1;
    }
    
    // Use engine...
    
    // Cleanup
    sle_Destroy(engine);
    return 0;
}
```

## Basic Usage

### Creating and Configuring Engine

```c
SLE_Handle engine;

// Option 1: Default configuration
sle_Create(&engine);

// Option 2: From config file
sle_CreateFromConfigFile(&engine, "my_config.config");

// Option 3: Explicit configuration
SLE_Config config = {};
config.max_buses = 100000;
config.max_measurements = 500000;
config.convergence_tolerance = 1e-5f;
sle_CreateWithConfig(&engine, &config);

// Initialize
sle_Initialize(engine);
```

### Loading IEEE 14-Bus Test Case

```c
SLE_Handle engine;
sle_Create(&engine);
sle_Initialize(engine);

// Load standard test case
if (sle_LoadIEEE14Bus(engine) == SLE_OK) {
    printf("IEEE 14-Bus loaded successfully\n");
    printf("Buses: %d\n", sle_GetBusCount(engine));
    printf("Branches: %d\n", sle_GetBranchCount(engine));
    printf("Measurements: %d\n", sle_GetMeasurementCount(engine));
}
```

## Model Building

### Adding Buses

```c
SLE_BusInfo bus = {};

// Slack bus
bus.id = "Bus1";
bus.base_kv = 138.0f;
bus.type = SLE_BUS_SLACK;
bus.v_setpoint = 1.06f;
bus.p_scheduled = 0.0f;
bus.q_scheduled = 0.0f;
sle_AddBus(engine, &bus);

// PV bus (generator)
bus.id = "Bus2";
bus.type = SLE_BUS_PV;
bus.v_setpoint = 1.045f;
bus.p_scheduled = 0.5f;  // 50 MW
sle_AddBus(engine, &bus);

// PQ bus (load)
bus.id = "Bus3";
bus.type = SLE_BUS_PQ;
bus.p_scheduled = -0.3f;  // -30 MW (load)
bus.q_scheduled = -0.1f;  // -10 MVar
sle_AddBus(engine, &bus);
```

### Adding Branches

```c
SLE_BranchInfo branch = {};

// Transmission line
branch.id = "Line1-2";
branch.from_bus = "Bus1";
branch.to_bus = "Bus2";
branch.resistance = 0.01938f;   // R [p.u.]
branch.reactance = 0.05917f;    // X [p.u.]
branch.susceptance = 0.0528f;   // B [p.u.]
branch.conductance = 0.0f;
branch.tap_ratio = 1.0f;        // No tap
branch.phase_shift = 0.0f;
branch.is_transformer = 0;
branch.sd_id = NULL;
sle_AddBranch(engine, &branch);

// Transformer
branch.id = "Xfmr4-7";
branch.from_bus = "Bus4";
branch.to_bus = "Bus7";
branch.resistance = 0.0f;
branch.reactance = 0.20912f;
branch.susceptance = 0.0f;
branch.tap_ratio = 0.978f;      // Off-nominal tap
branch.phase_shift = 0.0f;
branch.is_transformer = 1;
sle_AddBranch(engine, &branch);
```

### Adding Measurements

```c
SLE_MeasurementInfo meas = {};

// Voltage magnitude measurement
meas.id = "V1";
meas.type = SLE_MEAS_V_MAG;
meas.location = "Bus1";
meas.branch_end = SLE_BRANCH_FROM;  // Not used for bus measurements
meas.sigma = 0.004f;                // 0.4% standard deviation
meas.pt_ratio = 1.0f;               // No PT
meas.ct_ratio = 1.0f;               // No CT
meas.is_pseudo = 0;
sle_AddMeasurement(engine, &meas);

// Power injection measurement
meas.id = "P1";
meas.type = SLE_MEAS_P_INJECTION;
meas.location = "Bus1";
meas.sigma = 0.01f;                 // 1% standard deviation
meas.pt_ratio = 138.0f / 120.0f;    // PT ratio
meas.ct_ratio = 2000.0f / 5.0f;     // CT ratio
sle_AddMeasurement(engine, &meas);

// Power flow measurement
meas.id = "Pf_1-2";
meas.type = SLE_MEAS_P_FLOW;
meas.location = "Line1-2";
meas.branch_end = SLE_BRANCH_FROM;  // From end
meas.sigma = 0.008f;
sle_AddMeasurement(engine, &meas);

// Pseudo measurement (zero injection)
meas.id = "P7_pseudo";
meas.type = SLE_MEAS_P_PSEUDO;
meas.location = "Bus7";
meas.sigma = 0.001f;                // Very tight tolerance
meas.is_pseudo = 1;
sle_AddMeasurement(engine, &meas);
```

### Batch Operations

```c
// Add multiple buses at once
SLE_BusInfo buses[10];
// ... populate buses array ...
sle_AddBuses(engine, buses, 10);

// Add multiple branches
SLE_BranchInfo branches[20];
// ... populate branches array ...
sle_AddBranches(engine, branches, 20);
```

## State Estimation

### Basic Estimation

```c
// Upload model to GPU
sle_UploadModel(engine);

// Update telemetry
SLE_Real measurements[100];
// ... populate measurement values ...
sle_UpdateTelemetry(engine, measurements, 100);

// Run estimation
SLE_Result result;
SLE_StatusCode status = sle_Solve(engine, SLE_MODE_PRECISION, &result);

if (status == SLE_OK) {
    printf("Converged in %d iterations\n", result.iterations);
    printf("Objective: %f\n", result.objective);
    printf("Computation time: %f ms\n", result.computation_time_ms);
} else {
    printf("Estimation failed: %s\n", sle_GetLastError(engine));
}
```

### Real-Time Mode

```c
// Real-time mode: fast, limited iterations
SLE_Result result;
sle_Solve(engine, SLE_MODE_REALTIME, &result);

// Check if converged
if (result.status == SLE_CONVERGED) {
    // Use results immediately
} else if (result.status == SLE_MAX_ITERATIONS) {
    // Acceptable for real-time (may not be fully converged)
}
```

### Custom Solver Configuration

```c
SLE_SolverConfig config = {};
config.mode = SLE_MODE_PRECISION;
config.max_iterations = 50;
config.tolerance = 1e-6f;
config.use_robust = 1;              // Enable Huber M-estimator
config.huber_gamma = 1.5f;
config.use_flat_start = 0;          // Use hot start
config.time_limit_ms = 100.0f;      // 100ms time limit

SLE_Result result;
sle_SolveWithConfig(engine, &config, &result);
```

## Topology Changes

### Switching Device Operations

```c
// Add switching device
SLE_SwitchInfo sw = {};
sw.id = "SW1-5";
sw.branch_id = "Line1-5";
sw.status = SLE_SWITCH_CLOSED;
sle_AddSwitch(engine, &sw);

// Re-upload model (structural change)
sle_UploadModel(engine);

// Open switch (simulating breaker trip)
sle_SetSwitchStatus(engine, "SW1-5", SLE_SWITCH_OPEN);

// Run estimation (topology change triggers Ybus update)
SLE_Result result;
sle_Solve(engine, SLE_MODE_REALTIME, &result);

// Close switch back
sle_SetSwitchStatus(engine, "SW1-5", SLE_SWITCH_CLOSED);
sle_Solve(engine, SLE_MODE_REALTIME, &result);
```

### Applying Switch Changes Immediately

```c
// Queue multiple switch changes
sle_SetSwitchStatus(engine, "SW1", SLE_SWITCH_OPEN);
sle_SetSwitchStatus(engine, "SW2", SLE_SWITCH_CLOSED);

// Apply all changes immediately (normally done automatically before solve)
sle_ApplySwitchChanges(engine);

// Now solve with new topology
sle_Solve(engine, SLE_MODE_REALTIME, &result);
```

## Transformer Control

### Tap Ratio Control

```c
// Get current tap ratio
SLE_Real current_tap = sle_GetTransformerTap(engine, "Xfmr4-7");
printf("Current tap: %f\n", current_tap);

// Change tap ratio (voltage regulation)
sle_SetTransformerTap(engine, "Xfmr4-7", 1.05f);  // +5%

// Re-upload model (tap change affects admittance)
sle_UploadModel(engine);

// Solve with new tap setting
SLE_Result result;
sle_Solve(engine, SLE_MODE_PRECISION, &result);
```

### Phase Shift Control

```c
// Set phase shift angle (for phase-shifting transformer)
sle_SetTransformerPhaseShift(engine, "Xfmr4-7", 0.1f);  // 0.1 radians

// Re-upload and solve
sle_UploadModel(engine);
sle_Solve(engine, SLE_MODE_PRECISION, &result);
```

### Impedance Updates

```c
// Update branch impedance (e.g., temperature correction)
sle_SetBranchImpedance(engine, "Line1-2", 
                       0.020f,  // New R
                       0.060f,  // New X
                       0.055f,  // New B
                       0.0f);   // G (unchanged)

// Re-upload and solve
sle_UploadModel(engine);
sle_Solve(engine, SLE_MODE_PRECISION, &result);
```

## Bad Data Detection

### Identifying Bad Data

```c
// Run estimation
SLE_Result result;
sle_Solve(engine, SLE_MODE_PRECISION, &result);

// Check for bad data
if (result.bad_data_count > 0) {
    printf("Found %d bad measurements\n", result.bad_data_count);
    printf("Largest residual: %f at index %d\n", 
           result.largest_residual, result.largest_residual_idx);
}

// Get all bad data indices
int32_t bad_indices[100];
int32_t bad_count = 0;
sle_IdentifyBadData(engine, 3.0f,  // 3-sigma threshold
                    bad_indices, 100, &bad_count);

for (int i = 0; i < bad_count; ++i) {
    printf("Bad measurement at index %d\n", bad_indices[i]);
}

// Get residuals
SLE_Real residuals[100];
sle_GetResiduals(engine, residuals, 100);

for (int i = 0; i < bad_count; ++i) {
    printf("Residual[%d] = %f\n", bad_indices[i], residuals[bad_indices[i]]);
}
```

### Robust Estimation

```c
// Configure robust estimation (Huber M-estimator)
SLE_SolverConfig config = {};
config.mode = SLE_MODE_PRECISION;
config.use_robust = 1;
config.huber_gamma = 1.5f;  // Huber threshold
config.max_iterations = 50;

SLE_Result result;
sle_SolveWithConfig(engine, &config, &result);

// Robust estimation reduces impact of bad data
printf("Robust objective: %f\n", result.objective);
```

## Result Access

### Voltage Results

```c
// Get all voltage magnitudes
int32_t bus_count = sle_GetBusCount(engine);
SLE_Real* voltages = (SLE_Real*)malloc(bus_count * sizeof(SLE_Real));
sle_GetVoltageMagnitudes(engine, voltages, bus_count);

for (int i = 0; i < bus_count; ++i) {
    printf("Bus %d: V = %f p.u.\n", i + 1, voltages[i]);
}

// Get voltage for specific bus
SLE_Real v1 = sle_GetBusVoltage(engine, "Bus1");
printf("Bus1 voltage: %f p.u.\n", v1);

// Get voltage angles
SLE_Real* angles = (SLE_Real*)malloc(bus_count * sizeof(SLE_Real));
sle_GetVoltageAngles(engine, angles, bus_count);

for (int i = 0; i < bus_count; ++i) {
    float degrees = angles[i] * 180.0f / 3.14159265f;
    printf("Bus %d: Angle = %f degrees\n", i + 1, degrees);
}
```

### Power Results

```c
// Get power injections
int32_t bus_count = sle_GetBusCount(engine);
SLE_Real* p_inj = (SLE_Real*)malloc(bus_count * sizeof(SLE_Real));
SLE_Real* q_inj = (SLE_Real*)malloc(bus_count * sizeof(SLE_Real));

sle_GetPowerInjections(engine, p_inj, q_inj, bus_count);

for (int i = 0; i < bus_count; ++i) {
    printf("Bus %d: P = %f, Q = %f p.u.\n", i + 1, p_inj[i], q_inj[i]);
}

// Get branch flows
int32_t branch_count = sle_GetBranchCount(engine);
SLE_Real* p_from = (SLE_Real*)malloc(branch_count * sizeof(SLE_Real));
SLE_Real* q_from = (SLE_Real*)malloc(branch_count * sizeof(SLE_Real));
SLE_Real* p_to = (SLE_Real*)malloc(branch_count * sizeof(SLE_Real));
SLE_Real* q_to = (SLE_Real*)malloc(branch_count * sizeof(SLE_Real));

sle_GetBranchFlows(engine, p_from, q_from, p_to, q_to, branch_count);

// Get flow for specific branch
SLE_Real p_flow = sle_GetBranchP(engine, "Line1-2", SLE_BRANCH_FROM);
printf("Line1-2 P flow (from): %f p.u.\n", p_flow);
```

## Complete Example

See `examples/example_usage.cpp` for a comprehensive example demonstrating:

1. Engine creation and initialization
2. Model building (IEEE 14-Bus)
3. Model upload to GPU
4. Telemetry updates
5. Real-time and precision estimation modes
6. Topology changes (switch operations)
7. Transformer tap control
8. Structural modifications
9. Bad data detection
10. Robust estimation

### Running the Example

```powershell
# Build the example (see BUILD_INSTRUCTIONS.md)
cl /EHsc examples\example_usage.cpp /I"include" /link SLE.lib /LIBPATH:"bin\Release"

# Copy DLL to example directory
Copy-Item bin\Release\SLE.dll examples\

# Run
.\examples\example_usage.exe
```

## Error Handling Best Practices

```c
SLE_StatusCode status;

// Check return codes
status = sle_Create(&engine);
if (status != SLE_OK) {
    fprintf(stderr, "Error: %s\n", sle_GetLastError(engine));
    return 1;
}

// Check specific error types
status = sle_Solve(engine, SLE_MODE_PRECISION, &result);
switch (status) {
    case SLE_OK:
        // Success
        break;
    case SLE_ERROR_NOT_OBSERVABLE:
        fprintf(stderr, "Network is not observable\n");
        break;
    case SLE_ERROR_SINGULAR_MATRIX:
        fprintf(stderr, "Gain matrix is singular\n");
        break;
    case SLE_ERROR_DIVERGED:
        fprintf(stderr, "Solution diverged\n");
        // Try flat start
        sle_ApplyFlatStart(engine);
        sle_Solve(engine, SLE_MODE_PRECISION, &result);
        break;
    default:
        fprintf(stderr, "Error: %s\n", sle_GetLastError(engine));
        break;
}
```

## Performance Tips

1. **Use Real-Time Mode** for frequent updates:
   ```c
   sle_Solve(engine, SLE_MODE_REALTIME, &result);
   ```

2. **Batch Operations** when possible:
   ```c
   sle_AddBuses(engine, buses, count);  // Faster than individual adds
   ```

3. **Reuse Engine Instance**:
   - Create once, use many times
   - Avoid frequent create/destroy

4. **Pre-allocate Arrays**:
   - Allocate result arrays once
   - Reuse across estimation cycles

5. **Check Observability** before solving:
   ```c
   int32_t is_observable;
   sle_CheckObservability(engine, &is_observable, NULL);
   if (!is_observable) {
       // Handle unobservable case
   }
   ```

## Next Steps

- See [API Reference](API_REFERENCE.md) for complete function documentation
- See [Architecture](ARCHITECTURE.md) for system design details
- See [Configuration](CONFIGURATION.md) for configuration options

