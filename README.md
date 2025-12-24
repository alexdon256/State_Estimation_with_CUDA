# SLE Engine — GPU-Accelerated Power Grid State Estimator

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%2064--bit-lightgrey.svg)](https://www.microsoft.com/windows)

## What is the SLE Engine?

The **State Load Estimator (SLE) Engine** is a high-performance computational library that determines the real-time electrical state of a power grid. It takes raw sensor measurements from across a power network and computes the most likely voltage magnitudes and phase angles at every bus (node) in the system.

This is the same class of technology used in Energy Management Systems (EMS) at utility control centers worldwide — but accelerated with NVIDIA CUDA to achieve **real-time performance** on networks with hundreds of thousands of buses.

## Why State Estimation Matters

Modern power grids are massive, interconnected systems with thousands of substations, transmission lines, and generators. Operators need to know the exact electrical state of the entire network to:

- **Detect equipment failures** before they cascade
- **Identify cyberattacks** that inject false sensor data
- **Optimize power flow** to reduce losses and costs
- **Prevent blackouts** by monitoring stability margins
- **Enable renewable integration** with accurate forecasting

The challenge? Sensors are imperfect. Measurements have noise, some sensors fail, and you can never measure everything. **State estimation** solves this by combining physics (power flow equations) with statistics (weighted least squares) to compute the best estimate of the true system state.

```
    ┌─────────────────────────────────────────────────────────────────┐
    │                    POWER GRID (Physical World)                  │
    │                                                                 │
    │   [Gen]──────[Bus A]══════════[Bus B]──────[Load]              │
    │                 │                │                              │
    │              [Meter]          [Meter]                          │
    │                 │                │                              │
    │            ┌────▼────────────────▼────┐                        │
    │            │   Noisy Measurements z    │                        │
    │            │  (Voltages, Power Flows)  │                        │
    │            └────────────┬─────────────┘                        │
    └─────────────────────────┼───────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                     SLE ENGINE (GPU)                            │
    │                                                                 │
    │   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     │
    │   │  Jacobian   │────▶│ Gain Matrix │────▶│  Cholesky   │     │
    │   │  H(x)       │     │ G = HᵀWH   │     │  L·Lᵀ = G   │     │
    │   └─────────────┘     └─────────────┘     └──────┬──────┘     │
    │                                                   │            │
    │                                           ┌──────▼──────┐     │
    │                                           │   Solve     │     │
    │                                           │  Δx = L⁻¹b  │     │
    │                                           └──────┬──────┘     │
    └──────────────────────────────────────────────────┼─────────────┘
                                                       │
                              ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    ESTIMATED STATE x̂                            │
    │                                                                 │
    │   • Voltage magnitudes |V| at every bus                        │
    │   • Voltage angles θ at every bus                              │
    │   • Power flows on every line                                  │
    │   • Bad data flags for suspicious measurements                 │
    └─────────────────────────────────────────────────────────────────┘
```

## Key Features

### ⚡ GPU-Native Architecture
- **Full CUDA implementation** — no CPU fallbacks in the solver loop
- **Parallel sparse matrix operations** using cuSPARSE and cuSOLVER
- **Coalesced memory access** via Structure-of-Arrays (SoA) layout
- **CUDA Graphs** for minimal kernel launch overhead

### 🚀 Real-Time Performance
- **<10ms latency** for incremental measurement updates
- **<2s initial solve** for 200,000 bus networks
- **Hot start** reuses previous solution for faster convergence
- **Gain matrix caching** eliminates redundant refactorization

### 📊 Production-Grade Capabilities
- **200,000+ buses** and **2,000,000+ measurements** supported
- **Robust estimation** with Huber M-Estimator for bad data handling
- **Dynamic topology** — circuit breaker operations handled on GPU
- **Observability analysis** ensures solvability before estimation

### 🔌 Easy Integration
- **C-compatible DLL API** for any language (C, C++, Python, C#, etc.)
- **Configuration file** for all tunable parameters
- **IEEE 14-Bus test case** included for validation

## Technical Approach

The SLE Engine implements the **Weighted Least Squares (WLS)** algorithm, the industry standard for power system state estimation. The core optimization problem is:

```
minimize    J(x) = [z - h(x)]ᵀ · W · [z - h(x)]
```

Where:
- **x** = state vector (voltage magnitudes and angles)
- **z** = measurement vector (from sensors)
- **h(x)** = measurement function (power flow equations)
- **W** = weight matrix (inverse measurement variances)

This is solved iteratively using the **Gauss-Newton method**:

1. **Linearize** the measurement function around the current estimate
2. **Form** the Jacobian matrix H = ∂h/∂x
3. **Compute** the Gain matrix G = HᵀWH
4. **Factor** using sparse Cholesky decomposition: G = LLᵀ
5. **Solve** for the state update: Δx = L⁻ᵀL⁻¹Hᵀ W(z - h(x))
6. **Update** the state: x ← x + Δx
7. **Repeat** until convergence

### What Makes This Fast?

| Optimization | Benefit |
|-------------|---------|
| **GPU Parallelism** | Thousands of CUDA cores compute Jacobian entries simultaneously |
| **Sparse Matrices** | Only non-zero elements stored and computed (power grids are sparse!) |
| **Symbolic Analysis Caching** | Cholesky pattern computed once, reused for same topology |
| **Fill-Reducing Reordering** | AMD/RCM algorithms minimize factorization fill-in |
| **Pinned Memory** | Zero-copy telemetry transfer via DMA |
| **Hot Start** | Previous solution seeds next iteration |

## Quick Start

### Prerequisites

- **Windows 10/11** (64-bit)
- **Visual Studio 2019+** with C++ workload
- **NVIDIA GPU** with Compute Capability 7.0+ (Volta, Turing, Ampere, Ada, Hopper)
- **CUDA Toolkit 11.0+** (12.x recommended)

### Building

```powershell
# 1. Clone the repository
git clone https://github.com/youruser/SLE.git
cd SLE

# 2. Restore NuGet packages (Boost)
nuget restore SLE.sln

# 3. Build with MSBuild
msbuild SLE.sln /p:Configuration=Release /p:Platform=x64

# Output: bin\Release\SLE.dll, SLE.lib, sle_engine.config
```

### Basic Usage (C++)

```cpp
#include "sle_api.h"

int main() {
    SLE_Handle engine;
    
    // Create engine (auto-loads sle_engine.config)
    sle_Create(&engine);
    sle_Initialize(engine);
    
    // Define a simple 2-bus system
    SLE_BusInfo slack = { "Bus1", 138.0f, SLE_BUS_SLACK, 1.0f };
    SLE_BusInfo load  = { "Bus2", 138.0f, SLE_BUS_PQ, 1.0f };
    sle_AddBus(engine, &slack);
    sle_AddBus(engine, &load);
    
    // Add a transmission line
    SLE_BranchInfo line = { "Line1-2", "Bus1", "Bus2", 0.01f, 0.1f, 0.02f };
    sle_AddBranch(engine, &line);
    
    // Add voltage measurements
    SLE_MeasurementInfo v1 = { "V1", SLE_MEAS_VMAG, "Bus1", 0.004f };
    SLE_MeasurementInfo v2 = { "V2", SLE_MEAS_VMAG, "Bus2", 0.004f };
    sle_AddMeasurement(engine, &v1);
    sle_AddMeasurement(engine, &v2);
    
    // Upload model to GPU
    sle_UploadModel(engine);
    
    // Set measurement values (from SCADA)
    float telemetry[] = { 1.02f, 0.98f };  // Voltage readings
    sle_UpdateTelemetry(engine, telemetry, 2);
    
    // Run state estimation
    SLE_Result result;
    sle_Solve(engine, SLE_MODE_REALTIME, &result);
    
    if (result.status == SLE_STATUS_CONVERGED) {
        printf("Converged in %d iterations (%.2f ms)\n", 
               result.iterations, result.time_ms);
        
        // Get estimated voltages
        float v_mag[2];
        sle_GetBusVoltageMagnitudes(engine, v_mag, 2);
        printf("Bus1: %.4f p.u., Bus2: %.4f p.u.\n", v_mag[0], v_mag[1]);
    }
    
    sle_Destroy(engine);
    return 0;
}
```

## Project Structure

```
SLE/
├── include/                    # Public headers
│   ├── sle_api.h              # C-compatible DLL API
│   ├── sle_engine.h           # C++ engine interface
│   ├── sle_types.cuh          # Core type definitions
│   ├── wls_solver.cuh         # WLS solver interface
│   ├── sparse_matrix.cuh      # Sparse matrix operations
│   ├── cholesky_optimized.cuh # Cholesky factorization
│   ├── kernels.cuh            # CUDA kernel declarations
│   ├── network_model.h        # Host-side network model
│   └── sle_config.h           # Configuration loader
│
├── src/                        # Implementation files
│   ├── sle_api.cu             # C API implementation
│   ├── sle_engine.cu          # Engine core logic
│   ├── wls_solver.cu          # WLS algorithm
│   ├── sparse_matrix.cu       # cuSPARSE wrappers
│   ├── cholesky_optimized.cu  # Optimized Cholesky
│   ├── kernels.cu             # CUDA kernels
│   └── network_model.cu       # Model management
│
├── data/                       # Test data
│   └── ieee14bus.h            # IEEE 14-bus test case
│
├── sle_engine.config          # Default configuration
├── SLE.vcxproj                # Visual Studio project
├── SRS.md                     # Requirements specification
├── BUILD_INSTRUCTIONS.md      # Detailed build guide
└── LICENSE                    # BSD 3-Clause
```

## Configuration

All engine parameters are configurable via `sle_engine.config`:

```ini
[solver]
convergence_tolerance = 1e-4      # WLS convergence threshold
max_realtime_iterations = 5       # Iteration limit for real-time mode
max_precision_iterations = 100    # Iteration limit for precision mode
enable_robust_estimation = false  # Enable Huber M-estimator
huber_gamma = 1.5                 # Huber threshold parameter

[capacity]
max_buses = 200000                # Pre-allocated bus capacity
max_branches = 400000             # Pre-allocated branch capacity
max_measurements = 2000000        # Pre-allocated measurement capacity

[performance]
use_cuda_graphs = true            # Enable CUDA graph optimization
block_size_standard = 256         # CUDA block size for kernels
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Initial Solve** | < 2 seconds | 200K bus network, cold start |
| **Incremental Update** | < 10 ms | Hot start, topology unchanged |
| **Memory Usage** | < 8 GB | GPU VRAM for 200K bus model |
| **Convergence** | 3-5 iterations | Real-time mode with hot start |

## Supported Measurements

| Type | Description | Location |
|------|-------------|----------|
| `V_MAG` | Voltage magnitude | Bus |
| `V_ANGLE` | Voltage angle (PMU) | Bus |
| `P_INJECTION` | Active power injection | Bus |
| `Q_INJECTION` | Reactive power injection | Bus |
| `P_FLOW` | Active power flow | Branch end |
| `Q_FLOW` | Reactive power flow | Branch end |
| `I_MAG` | Current magnitude | Branch end |
| `P_PSEUDO` | Pseudo-measurement (zero injection) | Bus |
| `Q_PSEUDO` | Pseudo-measurement (zero injection) | Bus |

## Roadmap

- [x] Core WLS solver with GPU acceleration
- [x] Robust estimation (Huber M-estimator)
- [x] Dynamic topology processing
- [x] C-compatible DLL API
- [ ] Multi-GPU support for ultra-large networks
- [ ] Extended Kalman Filter for dynamic state estimation
- [ ] Python bindings via ctypes/pybind11
- [ ] Linux support

## Author

**Oleksandr Don** — [GitHub](https://github.com/alexdon256)

## License

Copyright (c) 2025, Oleksandr Don

This project is licensed under the **BSD 3-Clause License**. See [LICENSE](LICENSE) for details.

## References

- A. Abur and A. G. Expósito, *Power System State Estimation: Theory and Implementation*, CRC Press, 2004.
- IEEE 14-Bus Test Case — included in `data/ieee14bus.h`
- [NVIDIA cuSPARSE Documentation](https://docs.nvidia.com/cuda/cusparse/)
- [NVIDIA cuSOLVER Documentation](https://docs.nvidia.com/cuda/cusolver/)
