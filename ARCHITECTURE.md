# SLE Engine Architecture

Technical architecture and design documentation for the CUDA-Accelerated State Load Estimator Engine.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Component Design](#component-design)
- [Data Flow](#data-flow)
- [Memory Management](#memory-management)
- [GPU Acceleration](#gpu-acceleration)
- [Performance Optimizations](#performance-optimizations)

## Overview

The SLE Engine is designed as a high-performance, GPU-accelerated state estimation system for power networks. It follows a component-based architecture with clear separation between host (CPU) and device (GPU) code.

### Design Principles

1. **GPU-First**: All computation-intensive operations run on GPU (NFR-01)
2. **Zero CPU Fallback**: No CPU implementations for CUDA operations
3. **Structure of Arrays (SoA)**: Optimized memory layout for coalesced access (NFR-03)
4. **Pinned Memory**: Efficient DMA transfers for telemetry (FR-06)
5. **Gain Matrix Reuse**: Minimize refactorization for real-time performance (FR-16)

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                    │
│              (C API via sle_api.h)                      │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  SLEEngine (C++ API)                    │
│  - Model Management                                     │
│  - Lifecycle Control                                    │
│  - Result Access                                        │
└───────┬───────────────────────────────┬──────────────────┘
        │                               │
┌───────▼──────────┐         ┌──────────▼──────────┐
│  NetworkModel    │         │  DeviceDataManager │
│  (Host-Side)     │◄────────┤  (GPU Memory)      │
│                  │  Upload │                    │
└──────────────────┘         └──────────┬─────────┘
                                         │
        ┌────────────────────────────────┼────────────────────┐
        │                                │                    │
┌───────▼──────────┐         ┌───────────▼─────────┐ ┌───────▼──────────┐
│ WLSSolver        │         │ TopologyProcessor   │ │ SparseMatrixMgr │
│ (GPU Kernels)    │         │ (GPU Kernels)       │ │ (cuSPARSE)      │
└──────────────────┘         └─────────────────────┘ └─────────────────┘
```

### Component Layers

1. **API Layer** (`sle_api.h`)
   - C-compatible DLL interface
   - Error handling and status codes
   - Thread-safe handle management

2. **Engine Layer** (`sle_engine.h`)
   - High-level C++ API
   - Component orchestration
   - Configuration management

3. **Model Layer** (`network_model.h`)
   - Host-side network data
   - String ID to index mapping
   - Model validation

4. **GPU Layer** (CUDA kernels)
   - Device memory management
   - Parallel computation kernels
   - cuSPARSE/cuSOLVER integration

## Component Design

### SLEEngine Class

**Responsibilities:**
- Engine lifecycle management
- Component coordination
- Configuration handling
- Result aggregation

**Key Methods:**
```cpp
class SLEEngine {
    // Lifecycle
    bool initialize();
    void reset();
    
    // Model Building
    int32_t addBus(const BusDescriptor& desc);
    int32_t addBranch(const BranchDescriptor& desc);
    int32_t addMeasurement(const MeasurementDescriptor& desc);
    
    // GPU Operations
    bool uploadModel();
    bool updateTelemetry(const Real* values, int32_t count);
    
    // Estimation
    EstimationResult solve(EstimationMode mode);
    
    // Results
    std::vector<Real> getVoltageMagnitudes() const;
    std::vector<Real> getVoltageAngles() const;
};
```

### NetworkModel Class

**Responsibilities:**
- Host-side network element storage
- String ID to index mapping (using `std::unordered_map`)
- Pointer-stable storage (using `boost::stable_vector`)
- Model validation

**Data Structures:**
```cpp
class NetworkModel {
    // Element storage (pointer-stable)
    boost::container::stable_vector<BusElement> buses_;
    boost::container::stable_vector<BranchElement> branches_;
    boost::container::stable_vector<MeasurementElement> measurements_;
    
    // Fast lookup (O(1))
    IdIndexMap bus_id_map_;
    IdIndexMap branch_id_map_;
    IdIndexMap measurement_id_map_;
};
```

**Design Rationale:**
- `boost::stable_vector`: Guarantees pointer stability (NFR-18)
- `std::unordered_map`: O(1) average-case lookups for ID-to-index mapping
- SoA layout: Optimized for GPU transfer

### DeviceDataManager Class

**Responsibilities:**
- GPU memory allocation and management
- Host-to-device data transfer
- Device-to-host result transfer
- Pinned memory management

**Memory Layout (SoA):**
```cpp
struct DeviceBusData {
    int32_t count;
    Real* d_v_mag;        // Array of magnitudes
    Real* d_v_angle;       // Array of angles
    Real* d_base_kv;       // Array of base voltages
    // ... more arrays
};
```

**Key Features:**
- Pre-allocated device memory (FR-06)
- Pinned host memory for async transfers
- Structure of Arrays for coalesced access

### WLSSolver Class

**Responsibilities:**
- Weighted Least Squares iteration loop
- Jacobian matrix computation
- Gain matrix formation and factorization
- State update computation

**Algorithm Flow:**
```
1. Compute Residuals: r = z - h(x)
2. Update Jacobian: H = ∂h/∂x
3. Form Gain Matrix: G = H^T W H
4. Factorize: G = L L^T (Cholesky)
5. Solve: L L^T Δx = H^T W r
6. Update State: x = x + Δx
7. Check Convergence
```

**GPU Implementation:**
- Parallel Jacobian computation (kernels)
- cuSPARSE for SpMV and SpGEMM
- cuSOLVER for Cholesky factorization

### TopologyProcessor Class

**Responsibilities:**
- Switching device status management
- Ybus matrix updates (FR-05)
- Zero-impedance branch handling (FR-04)
- Partial matrix updates

**Topology Change Flow:**
```
1. Queue switch status change
2. Identify affected branches
3. Update branch status on GPU
4. Partial Ybus update (GPU kernel)
5. Invalidate gain matrix factorization
```

### SparseMatrixManager Class

**Responsibilities:**
- CSR matrix allocation
- Ybus matrix management
- Sparse matrix operations (SpMV, SpGEMM)
- Cholesky factorization

**Integration:**
- cuSPARSE: Sparse matrix operations
- cuSOLVER: Cholesky factorization
- Custom kernels: Ybus construction

## Data Flow

### Model Upload Flow

```
Host (NetworkModel)
    │
    ├─ Pack to SoA format
    │
    ├─ Allocate GPU memory (DeviceDataManager)
    │
    ├─ Copy to pinned host memory
    │
    ├─ Async transfer to GPU (cudaMemcpyAsync)
    │
    └─ GPU (DeviceBusData, DeviceBranchData, etc.)
```

### Telemetry Update Flow

```
Application
    │
    ├─ Update measurement values
    │
    ├─ Copy to pinned memory (FR-07)
    │
    ├─ Async DMA transfer to GPU
    │
    └─ GPU measurement buffer updated
```

### Estimation Flow

```
sle_Solve()
    │
    ├─ Check observability (FR-13)
    │
    ├─ Handle topology changes (FR-05)
    │
    ├─ WLS Iteration Loop (FR-08)
    │   │
    │   ├─ Compute residuals (GPU kernel)
    │   │
    │   ├─ Update Jacobian (GPU kernel)
    │   │
    │   ├─ Form Gain Matrix (cuSPARSE SpGEMM)
    │   │
    │   ├─ Factorize (cuSOLVER Cholesky)
    │   │
    │   ├─ Solve (cuSOLVER forward/backward)
    │   │
    │   └─ Update state (GPU kernel)
    │
    ├─ Download results (DeviceDataManager)
    │
    └─ Return EstimationResult
```

### Result Access Flow

```
GPU (DeviceBusData)
    │
    ├─ Download to host (cudaMemcpy)
    │
    ├─ Host pinned memory
    │
    ├─ Copy to NetworkModel
    │
    └─ Application (via API)
```

## Memory Management

### Host-Side Memory

**Allocation Strategy:**
- `boost::stable_vector`: Dynamic growth with pointer stability
- `std::unordered_map`: Hash tables with `reserve()` for capacity hints
- Pinned memory: For async GPU transfers

**Memory Layout:**
- Structure of Arrays (SoA) for GPU transfer
- Cache-line aligned structures (NFR-11)

### GPU Memory

**Allocation Strategy:**
- Pre-allocated based on `max_buses`, `max_branches`, etc.
- Persistent allocation (FR-06)
- Reused across estimation cycles

**Memory Types:**
- Global memory: Network data, matrices
- Shared memory: Kernel optimization (Section 5.1)
- Constant memory: Configuration parameters

### Memory Transfer Optimization

**Pinned Memory:**
- Page-locked host memory
- Direct DMA access
- Async transfers with CUDA streams

**Transfer Overlap:**
- CUDA streams for async operations
- Overlap computation and transfer
- Pipeline multiple operations

## GPU Acceleration

### Kernel Design

**Optimization Techniques:**
- Shared memory tiling (Section 5.1)
- Coalesced memory access
- Warp-level operations
- Loop unrolling (NFR-23)

**Kernel Categories:**
1. **Element-wise**: Voltage, angle updates
2. **Reduction**: Residual computation, objective function
3. **Sparse**: Jacobian construction, Ybus updates
4. **Matrix**: SpMV, SpGEMM (cuSPARSE)

### Library Integration

**cuSPARSE:**
- Sparse Matrix-Vector Multiplication (SpMV)
- Sparse Matrix-Matrix Multiplication (SpGEMM)
- CSR format operations

**cuSOLVER:**
- Sparse Cholesky factorization
- Forward/backward substitution
- Matrix analysis and reordering

**Thrust:**
- Vector reductions
- Sorting operations
- Transformations

### Stream Management

**CUDA Streams:**
- Separate streams for different operations
- Overlap computation and transfer
- Dependency management

**CUDA Graphs (Optional):**
- Capture iteration loop
- Reduce kernel launch overhead
- Enable in config: `use_cuda_graphs = true`

## Performance Optimizations

### Host-Side Optimizations

**OpenMP Parallelization (NFR-09, NFR-21):**
```cpp
#pragma omp parallel for
for (int i = 0; i < n; ++i) {
    // Parallel processing
}
```

**SIMD Vectorization (NFR-14, NFR-22):**
```cpp
#pragma omp simd reduction(+:sum)
for (int i = 0; i < n; ++i) {
    sum += array[i];
}
```

**Branchless Code (NFR-12):**
- Replace conditionals with arithmetic
- Use `std::min`/`std::max`
- Conditional moves

**Cache Optimization (NFR-11):**
- Cache-line aligned structures
- Prefetching
- Data locality

### GPU-Side Optimizations

**Shared Memory Tiling:**
- Reduce global memory access
- Improve data reuse
- Block-level computation

**Coalesced Access:**
- SoA layout ensures coalescing
- Aligned memory access
- Warp-aligned operations

**Kernel Fusion:**
- Combine multiple operations
- Reduce kernel launch overhead
- Improve data locality

**Gain Matrix Reuse (FR-16):**
- Preserve factorization between iterations
- Only refactorize on topology change
- Critical for real-time performance

### Memory Optimizations

**Zero-Copy:**
- Pinned memory for DMA
- Async transfers
- Overlap with computation

**Memory Pool:**
- Pre-allocated buffers
- Avoid runtime allocation (NFR-06)
- Reuse across cycles

## Configuration System

### Configuration File Format

See [Configuration Guide](CONFIGURATION.md) for details.

**Key Parameters:**
- Capacity: `max_buses`, `max_branches`, `max_measurements`
- Solver: `convergence_tolerance`, `max_iterations`
- Performance: `use_cuda_graphs`, `enable_profiling`
- Numerical: `use_double_precision`, `voltage_min_pu`

### Runtime Configuration

- Load from `.config` file
- Override programmatically
- Reload without restart (solver settings only)

## Error Handling

### Error Propagation

```
CUDA Error
    │
    ├─ Check CUDA error code
    │
    ├─ Convert to SLE_StatusCode
    │
    ├─ Store error message (thread-local)
    │
    └─ Return to application
```

### Error Recovery

- Automatic flat start on divergence (NFR-05)
- Fallback to previous state
- Observability checks before solving

## Thread Safety

### Current Design

- **NOT thread-safe** per handle
- Multiple handles can be used concurrently
- Thread-local error messages

### Future Considerations

- Per-handle mutex for thread safety
- Lock-free data structures
- Concurrent estimation support

## Testing Strategy

### Unit Tests

- Component-level testing
- Mock GPU for host-side code
- Validation of data structures

### Integration Tests

- End-to-end estimation
- IEEE 14-Bus test case
- Performance benchmarks

### Performance Tests

- Latency measurements
- Throughput tests
- Memory usage profiling

## Future Enhancements

### Multi-GPU Support

- Partition network across GPUs
- Inter-GPU communication
- Load balancing

### Dynamic State Estimation

- Extended Kalman Filter
- Time-domain models
- Synchronous machine models

### Advanced Features

- Graph partitioning (Section 5.1)
- Matrix reordering
- Hierarchical sparse storage

