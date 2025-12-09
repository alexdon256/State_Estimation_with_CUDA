# Configuration Guide

Complete reference for the SLE Engine configuration system.

## Table of Contents

- [Overview](#overview)
- [Configuration File Format](#configuration-file-format)
- [Configuration Parameters](#configuration-parameters)
- [Loading Configuration](#loading-configuration)
- [Runtime Configuration](#runtime-configuration)
- [Default Values](#default-values)

## Overview

The SLE Engine uses a text-based `.config` file format for all configuration parameters. This allows easy tuning without recompilation and supports different configurations for different deployment scenarios.

### Configuration File Location

- **Default**: `sle_engine.config` in current directory
- **Custom**: Specify path in constructor or API call
- **Auto-load**: Engine automatically loads default config if present

## Configuration File Format

### File Structure

The configuration file uses INI-style format with sections and key-value pairs:

```ini
# Comments start with #
# Empty lines are ignored

[section_name]
key = value
another_key = value

[another_section]
key = value
```

### Example Configuration File

```ini
# SLE Engine Configuration File
# Generated: 2024-01-01

[device]
device_id = 0

[capacity]
max_buses = 200000
max_branches = 400000
max_measurements = 2000000
max_switching_devices = 100000

[solver]
enable_robust = false
huber_gamma = 1.5
convergence_tolerance = 0.0001
max_realtime_iterations = 5
max_precision_iterations = 100
realtime_time_limit_ms = 20.0
use_flat_start_default = false

[performance]
enable_profiling = false
use_cuda_graphs = true

[numerical]
use_double_precision = false
voltage_min_pu = 0.5
voltage_max_pu = 1.5

[bad_data]
bad_data_threshold = 3.0
auto_bad_data_rejection = false
```

## Configuration Parameters

### [device] Section

#### device_id

CUDA device ID to use.

- **Type**: `int32_t`
- **Default**: `0`
- **Range**: `0` to (number of GPUs - 1)
- **Example**: `device_id = 0`

**Usage:**
```c
// Use first GPU (default)
device_id = 0

// Use second GPU
device_id = 1
```

### [capacity] Section

Pre-allocation sizes for network elements. These determine GPU memory allocation.

#### max_buses

Maximum number of buses to support.

- **Type**: `size_t`
- **Default**: `200000`
- **Range**: `1` to `10000000`
- **Example**: `max_buses = 200000`

**Note:** Larger values require more GPU memory but allow larger networks.

#### max_branches

Maximum number of branches (lines and transformers).

- **Type**: `size_t`
- **Default**: `400000`
- **Range**: `1` to `20000000`
- **Example**: `max_branches = 400000`

**Note:** Typically 2x `max_buses` for transmission networks.

#### max_measurements

Maximum number of measurements.

- **Type**: `size_t`
- **Default**: `2000000`
- **Range**: `1` to `100000000`
- **Example**: `max_measurements = 2000000`

**Note:** Includes all measurement types (voltage, power, current, pseudo).

#### max_switching_devices

Maximum number of switching devices (breakers, switches, fuses).

- **Type**: `size_t`
- **Default**: `100000`
- **Range**: `1` to `10000000`
- **Example**: `max_switching_devices = 100000`

### [solver] Section

Solver algorithm parameters.

#### enable_robust

Enable Huber M-estimator for robust estimation (FR-17).

- **Type**: `bool`
- **Default**: `false`
- **Values**: `true`, `false`
- **Example**: `enable_robust = true`

**Usage:**
- `true`: Use robust estimation (handles bad data better)
- `false`: Standard WLS estimation

#### huber_gamma

Huber function threshold parameter (FR-17).

- **Type**: `float`
- **Default**: `1.5`
- **Range**: `0.1` to `10.0`
- **Example**: `huber_gamma = 1.5`

**Note:** Lower values = more robust but slower convergence.

#### convergence_tolerance

WLS convergence tolerance (mismatch threshold).

- **Type**: `float`
- **Default**: `0.0001` (1e-4)
- **Range**: `1e-8` to `1e-2`
- **Example**: `convergence_tolerance = 0.0001`

**Note:** Tighter tolerance = more iterations but better accuracy.

#### max_realtime_iterations

Maximum iterations for real-time mode (FR-11).

- **Type**: `int32_t`
- **Default**: `5`
- **Range**: `1` to `50`
- **Example**: `max_realtime_iterations = 5`

**Note:** Lower values = faster but may not converge.

#### max_precision_iterations

Maximum iterations for precision mode (FR-12).

- **Type**: `int32_t`
- **Default**: `100`
- **Range**: `10` to `1000`
- **Example**: `max_precision_iterations = 100`

#### realtime_time_limit_ms

Time limit for real-time mode [milliseconds].

- **Type**: `float`
- **Default**: `20.0`
- **Range**: `1.0` to `1000.0`
- **Example**: `realtime_time_limit_ms = 20.0`

**Note:** Solver stops if time limit exceeded, even if not converged.

#### use_flat_start_default

Default to flat start initialization (V=1.0, theta=0).

- **Type**: `bool`
- **Default**: `false`
- **Values**: `true`, `false`
- **Example**: `use_flat_start_default = false`

**Usage:**
- `true`: Always start from flat start
- `false`: Use hot start (previous solution) when available

### [performance] Section

Performance and optimization settings.

#### enable_profiling

Enable CUDA event timing for performance measurement.

- **Type**: `bool`
- **Default**: `false`
- **Values**: `true`, `false`
- **Example**: `enable_profiling = true`

**Note:** Adds overhead, use only for profiling.

#### use_cuda_graphs

Use CUDA Graphs for iteration loop (reduces kernel launch overhead).

- **Type**: `bool`
- **Default**: `true`
- **Values**: `true`, `false`
- **Example**: `use_cuda_graphs = true`

**Note:** Requires CUDA 10.0+. May improve performance for small networks.

### [numerical] Section

Numerical computation settings.

#### use_double_precision

Use double precision (fp64) instead of single precision (fp32).

- **Type**: `bool`
- **Default**: `false`
- **Values**: `true`, `false`
- **Example**: `use_double_precision = false`

**Note:** 
- `false`: Faster, less memory (recommended for real-time)
- `true`: More accurate, slower, more memory (for offline analysis)

#### voltage_min_pu

Minimum voltage limit [per unit].

- **Type**: `float`
- **Default**: `0.5`
- **Range**: `0.1` to `1.0`
- **Example**: `voltage_min_pu = 0.5`

**Note:** Used for validation and clamping.

#### voltage_max_pu

Maximum voltage limit [per unit].

- **Type**: `float`
- **Default**: `1.5`
- **Range**: `1.0` to `2.0`
- **Example**: `voltage_max_pu = 1.5`

### [bad_data] Section

Bad data detection and handling (FR-15).

#### bad_data_threshold

Normalized residual threshold for bad data identification.

- **Type**: `float`
- **Default**: `3.0`
- **Range**: `1.0` to `10.0`
- **Example**: `bad_data_threshold = 3.0`

**Note:** Typically 3.0 for 3-sigma detection.

#### auto_bad_data_rejection

Automatically reject bad data during estimation.

- **Type**: `bool`
- **Default**: `false`
- **Values**: `true`, `false`
- **Example**: `auto_bad_data_rejection = false`

**Note:** 
- `true`: Automatically exclude bad measurements
- `false`: Report bad data but include in estimation

## Loading Configuration

### From File

```c
// Load configuration from file
SLE_Config config;
if (sle_LoadConfigFile("my_config.config", &config) == SLE_OK) {
    // Use config
    SLE_Handle engine;
    sle_CreateWithConfig(&engine, &config);
}
```

### Default File

```c
// Engine automatically loads from sle_engine.config
SLE_Handle engine;
sle_Create(&engine);  // Loads default config automatically
```

### Programmatic

```c
// Set configuration programmatically
SLE_Config config = {};
config.max_buses = 100000;
config.convergence_tolerance = 1e-5f;
config.enable_robust = true;

SLE_Handle engine;
sle_CreateWithConfig(&engine, &config);
```

## Runtime Configuration

### Reloading Configuration

```c
// Reload from original config file
sle_ReloadConfig(engine);

// Reload from specific file
sle_ReloadConfigFromFile(engine, "new_config.config");
```

### Updating Configuration

```c
// Get current configuration
SLE_Config config;
sle_GetConfig(engine, &config);

// Modify settings
config.convergence_tolerance = 1e-6f;
config.enable_robust = true;

// Apply changes
sle_SetConfig(engine, &config);
```

**Note:** Only solver settings can be changed at runtime. Capacity changes require engine restart.

### Saving Configuration

```c
// Save current configuration
SLE_Config config;
sle_GetConfig(engine, &config);
sle_SaveConfigFile("backup.config", &config);
```

## Default Values

Complete default configuration:

```ini
[device]
device_id = 0

[capacity]
max_buses = 200000
max_branches = 400000
max_measurements = 2000000
max_switching_devices = 100000

[solver]
enable_robust = false
huber_gamma = 1.5
convergence_tolerance = 0.0001
max_realtime_iterations = 5
max_precision_iterations = 100
realtime_time_limit_ms = 20.0
use_flat_start_default = false

[performance]
enable_profiling = false
use_cuda_graphs = true

[numerical]
use_double_precision = false
voltage_min_pu = 0.5
voltage_max_pu = 1.5

[bad_data]
bad_data_threshold = 3.0
auto_bad_data_rejection = false
```

## Configuration Best Practices

### For Real-Time Applications

```ini
[solver]
max_realtime_iterations = 3
realtime_time_limit_ms = 10.0
convergence_tolerance = 0.001

[performance]
use_cuda_graphs = true
enable_profiling = false

[numerical]
use_double_precision = false
```

### For Offline Analysis

```ini
[solver]
max_precision_iterations = 200
convergence_tolerance = 0.000001
enable_robust = true

[numerical]
use_double_precision = true
```

### For Large Networks

```ini
[capacity]
max_buses = 500000
max_branches = 1000000
max_measurements = 5000000

[performance]
use_cuda_graphs = false  # May not help for large networks
```

### For Development/Debugging

```ini
[performance]
enable_profiling = true

[solver]
max_realtime_iterations = 10
convergence_tolerance = 0.01
```

## Creating Default Config File

```c
// Create default config file if it doesn't exist
sle_EnsureConfigFile("sle_engine.config");

// Or use default path
sle_EnsureConfigFile(NULL);
```

## Configuration Validation

The engine validates configuration on load:

- **Capacity values**: Must be positive
- **Tolerance values**: Must be in valid range
- **Iteration limits**: Must be positive
- **Time limits**: Must be positive

Invalid values are replaced with defaults and a warning is logged.

## Environment-Specific Configuration

### Development

```ini
# dev_config.config
[performance]
enable_profiling = true

[solver]
convergence_tolerance = 0.01  # Looser for faster testing
```

### Production

```ini
# prod_config.config
[performance]
enable_profiling = false

[solver]
convergence_tolerance = 0.0001
max_realtime_iterations = 5
```

### Testing

```ini
# test_config.config
[capacity]
max_buses = 1000  # Smaller for unit tests

[solver]
max_precision_iterations = 10
```

## Troubleshooting

### Configuration Not Loading

1. Check file path
2. Verify file format (sections, keys, values)
3. Check for syntax errors (comments, quotes)

### Configuration Changes Not Applied

1. Capacity changes require engine restart
2. Call `sle_SetConfig()` after changes
3. Verify changes with `sle_GetConfig()`

### Performance Issues

1. Check `use_cuda_graphs` setting
2. Verify `max_iterations` not too high
3. Check `enable_profiling` is false in production

## See Also

- [API Reference](API_REFERENCE.md) - Configuration API functions
- [Examples](EXAMPLES.md) - Configuration usage examples
- [Architecture](ARCHITECTURE.md) - How configuration affects system behavior

