# Software Requirements Specification (SRS)

**Project Name:** CUDA-Accelerated State Load Estimator (SLE) Engine

**Version:** 0.20 (Single-GPU Focus)

**Status:** In Progress

---

## 1. Introduction

### 1.1 Purpose

The purpose of this document is to define the requirements for a high-performance, GPU-based State Load Estimator (SLE) engine. This system mimics the capabilities of industry standards like ETAP but utilizes NVIDIA CUDA to achieve real-time performance for large-scale power networks. Prioritized GPU acceleration accent.

### 1.2 Product Scope

The SLE Engine acts as a calculation core that ingests a physical network model (Buses, Transformers, Branches) and real-time telemetry (Voltmeters, Multimeters, Switching Device states). It computes the "State Vector" (Voltage Magnitudes and Angles at every bus) and derived quantities (Power Flows, Injections) using a Weighted Least Squares (WLS) algorithm accelerated by GPU parallel computing. The system is designed to handle dynamic topology changes (switching events) and high-frequency measurement updates in real-time, targeting hundreds of thousands of buses and millions of measurements.

### 1.3 Definitions and Acronyms

- **SLE:** State Load Estimator
- **WLS:** Weighted Least Squares (standard algorithm for state estimation)
- **Robust Estimation:** State estimation method that minimizes the influence of large measurement errors (bad data) using non-quadratic objective functions, such as the Huber M-Estimator
- **Huber Function:** A robust loss function used in WLS that is quadratic for small residuals and linear for large residuals, effectively limiting the influence of outliers
- **SD:** Switching Device (Breakers, Fuses, Switches)
- **Topology Processor:** The logic that determines electrical connectivity based on SD status
- **Telemetry:** Real-time measurement data (V, I, MW, MVar)
- **SoA vs. AoS:** Structure of Arrays vs. Array of Structures (Memory layout patterns critical for CUDA performance)
- **Residual:** The difference between a measured value and the calculated estimated value ($z - h(x)$)
- **Jacobian Matrix ($H$):** The matrix of first-order partial derivatives used in the linearization of the power flow equations
- **Gain Matrix ($G$):** The sparse matrix product $H^T W H$ used in the Normal Equations
- **CSR/CSC:** Compressed Sparse Row / Compressed Sparse Column (formats for storing sparse matrices)

---

## 2. Overall Description

### 2.1 User Characteristics

- **System Integrator/Developer:** Needs a C++/CUDA API to feed the engine with network graph data and telemetry
- **Grid Operator:** Needs real-time visual feedback on network health, estimated flows, and convergence status
- **Power System Engineer:** Performs offline analysis, looking at residual plots and observability limits

### 2.2 Product Perspective

This is a backend calculation engine intended to be integrated into a SCADA or Digital Twin platform. It runs on a Host (CPU) + Device (NVIDIA GPU) architecture.

### 2.3 Assumptions and Dependencies

- Host system must have an NVIDIA GPU with Compute Capability 7.0+ (Volta or newer recommended). The system is designed for and constrained to a single CUDA device
- Telemetry updates occur frequently (e.g., every 20ms to 1s)
- Topology changes (switching) occur less frequently than measurements but must be handled without halting the system

---

## 3. Functional Requirements

### 3.1 Network Model Management (Host Side)

**FR-01 - Static Model Definition:** The system shall accept a definition of static network components during initialization, including:

- **Buses:** Base kV, Type (Slack, PQ, PV)
- **Branches:** Resistance (R), Reactance (X), Susceptance (B), From-Bus, To-Bus
- **Transformers:** Turns ratio, Phase shift, Impedance, Controllable Tap Settings
- **Network Elements:** Loads, Motors, Generators (P, Q setpoints), and Utility/External Connection points
- **Devices & Measurement Scaling (CT/PT Ratios):** Voltmeters (VM) for Bus Magnitudes, Multimeters (MM) for Branch Flows and Injections, Switching Devices (SD), Current Transformers (CT), and Potential Transformers (PT) ratio information. The model must integrate these PT and CT ratios as fixed parameters into the measurement function $h(x)$ to correctly scale the estimated primary values to match the secondary measured values ($z$)

**FR-02 - Measurement Mapping (Critical Input Source & Performance):** The system shall map physical devices (Voltmeters and Multimeters) to the mathematical model (Bus and Branch associations). To ensure optimal data throughput (NFR-01, NFR-02), the system must utilize a contiguous, indexed Measurement Association Array (e.g., `meas_location_map_`) that provides constant-time lookups to link each raw measurement index in the telemetry vector ($z$) to its specific network location (Bus/Branch index, From/To side, Type). This high-performance structure is mandatory for the parallel construction of the Jacobian matrix ($H$)

**FR-25 - Model Modification and Full Cycle Trigger:** The system shall provide an API to dynamically add, delete, or modify static network components (Buses, Branches, Devices, SDs, and associated Ratios/Impedances) post-initialization. Any structural modification (including adding/deleting buses, branches, or measurement associations) must trigger a full WLS estimation cycle (as defined by FR-12) to re-factor the Gain Matrix $G$ and ensure the Jacobian $H$ correctly reflects the new state space dimensions

### 3.2 Dynamic Topology Processing

**FR-03 - Switch Status Vector:** The system shall maintain a "Status Vector" for all Switching Devices (SD) on the GPU

**FR-04 - Zero-Impedance Handling:** The engine must handle SDs effectively. When an SD is Closed, it represents a zero-impedance branch (merging two buses mathematically) or a very low impedance branch. When Open, the branch is removed from the $Y_{bus}$

**FR-05 - GPU-Accelerated Re-Topology:** Upon detection of an SD status change (Breaker Trip/Close), the engine shall queue the change. The actual Partial Update of the Admittance Matrix ($Y_{bus}$) and Jacobian structure must be consolidated and applied just prior to the start of the next WLS estimation cycle, executing the re-topology logic directly on the GPU to avoid a full data re-upload from the CPU

### 3.3 CUDA Estimation Engine (Device Side)

**FR-06 - Memory Persistence & Zero-Copy:**

- Static model data (topology graphs, branch parameters) must be loaded to GPU memory once
- Telemetry data buffers (measurements) on the Host must use Pinned Memory (Page-Locked) to allow direct Direct Memory Access (DMA) by the GPU

**FR-07 - Streamlined Data Ingestion:** The API must provide a function `UpdateTelemetry(float* measurements, int count)` that asynchronously copies only the new measurement values to the GPU Device symbol using `cudaMemcpyAsync`

**FR-08 - Parallel Solver Pipeline:** The WLS iteration loop must be executed entirely on the GPU to minimize kernel launch overhead. The solver steps must be:

1. **Jacobian Update:** Parallel computation of $H(x)$ terms
2. **Gain Matrix Formation:** Sparse Matrix-Matrix Multiplication (SpGEMM) to form $G = H^T W H$
3. **Factorization:** Sparse Cholesky Decomposition ($G = L L^T$)
4. **Forward/Backward Substitution:**
   - **Forward:** Solve $L y = H^T W (z - h(x))$ for $y$
   - **Backward:** Solve $L^T \Delta x = y$ for the state update $\Delta x$

**FR-09 - Iteration Control:** The WLS iteration loop (Calculate Residuals $\rightarrow$ Update Jacobian $\rightarrow$ Solve $\rightarrow$ Update State) must be orchestrated using CUDA Streams and/or CUDA Graphs to sequence asynchronous memory transfers (FR-07) and kernel executions. This structure is mandatory to overlap compute operations with data transfers and hide kernel launch overhead, enabling the efficient staging of computation as required for real-time performance

**FR-16 - Gain Matrix Reuse (Steady Topology):** When the network topology (SD status) is stable, the following artifacts must be preserved and reused for subsequent estimation cycles (telemetry updates):

- The structure (sparsity pattern) of the Jacobian Matrix ($H$)
- The structure and factor values of the Gain Matrix ($G$) factorization ($L$ and $L^T$)

The re-computation of $G$ and its factorization should only be triggered by a topology change (FR-05), a structural model update (FR-25), or the Offline/Precision mode (FR-12)

### 3.4 Calculation Outputs (Derived Data)

**FR-10 - On-Device Result Population:** All derived quantities must be calculated on the GPU before transfer back to Host. The full list of supported input measurements and calculated outputs is detailed in Section 3.7. The required calculated outputs include:

- **State Vector:** Voltage Magnitude ($|V|$) and Angle ($\theta$)
- **Flows:** $P_{flow}, Q_{flow}, I_{mag}$ for all branches (calculated from estimated $V$)
- **Injections:** $P_{inj}, Q_{inj}$ at all buses
- **Residuals:** The residual vector $r = z_{measured} - h(x_{estimated})$ and the Weighted Sum of Squares ($J(x)$)

### 3.5 Estimation Modes

**FR-11 - Real-Time Mode (Time-Constrained):**

- Fixed execution window (e.g., must return result within 20ms)
- Limits max iterations (e.g., 3-5)
- Uses "Hot Start" (initializes $x$ using the result of the previous timestep). This mode is used exclusively when only telemetry changes (including SD status changes) occur, leveraging Gain Matrix Reuse (FR-16)

**FR-12 - Offline/Precision Mode:**

- Runs until strict convergence tolerance (e.g., mismatch < $10^{-6}$)
- **Mandatory:** Recalculate and re-factor the Gain Matrix $G$ in the first iteration of this mode, regardless of topology status, to ensure maximum numerical accuracy. This mode must be triggered after any structural model change (FR-25)
- Suitable for planning studies or post-mortem analysis

### 3.6 Analysis & Validation

**FR-13 - Observability Analysis:** Before solving, the engine must run a fast graph-based observability check (e.g., Maximum Flow or Num-Islands check) on the GPU to ensure the network is mathematically solvable given the current active topology and available sensors

**FR-14 - Convergence Flags:** Return codes indicating: Converged, Max Iterations, Diverged (NaN/Inf detected), or Singular Matrix (Observability lost)

**FR-15 - Comparator:** A CUDA kernel that efficiently computes the absolute and percentage error between $z_{measured}$ and $h(x_{final})$ for all active measurements

**FR-17 - Robust Estimation (Huber M-Estimator):** The engine must support a robust estimation mode where the objective function uses the Huber Function to compute the contribution of the residual to the overall cost function. This mitigates the impact of single bad data points without full rejection, by dynamically re-weighting measurements based on the magnitude of their residual. The threshold parameter ($\gamma$) for the Huber function shall be configurable

**FR-18 - Sample Data Provision:** The initial delivery must include the standard IEEE 14-Bus Test Case as a self-contained data package (bus, branch, and measurement definitions in a text-readable format). This data must be sufficient to run the WLS solver and validate FR-01 and FR-02

**FR-26 - Comprehensive Engine Usage Example (Mandatory Delivery):** The initial delivery package (FR-18) must be expanded to include a complete, runnable C++ example demonstrating the full operational lifecycle of the SLE Engine. This example must showcase:

1. **Model Upload & Full Run:** Initial model upload and execution of the full WLS cycle (FR-12)
2. **Telemetry Update & Fast Run:** Subsequent fast, real-time WLS runs using only updated telemetry (FR-11, FR-16)
3. **SD Update & Fast Run:** A change in Switching Device (SD) status, triggering the GPU-accelerated re-topology (FR-05) followed by a fast WLS run
4. **Structural Update & Full Run:** A structural model change (e.g., adding a new branch/device via FR-25) that mandates a full WLS cycle (FR-12)

### 3.7 Measurement and Derived Quantity Support

The SLE Engine must support a comprehensive set of power system quantities for both inputs (measurements) and outputs (estimated and derived values) to model a large, complex grid accurately.

#### 3.7.1 Supported Input Measurements (Telemetry)

The system must be able to ingest, process, and map the following real-time measurement types ($z$) to the Jacobian matrix rows. This list aligns with the essential telemetry and status inputs of commercial State Estimators.

| Measurement Type | Symbol | Location | Source Device | Description |
|-----------------|--------|----------|--------------|-------------|
| Bus Voltage Magnitude | $V_{m}$ | Bus | Voltmeter (VM) via PT | Voltage magnitude (typically p.u.). PT ratio is applied by $h(x)$ |
| Bus Active Power Injection | $P_{inj}$ | Bus | Multimeter (MM) via PT/CT | Net active power injected into the bus. PT/CT ratios applied by $h(x)$ |
| Bus Reactive Power Injection | $Q_{inj}$ | Bus | Multimeter (MM) via PT/CT | Net reactive power injected into the bus. PT/CT ratios applied by $h(x)$ |
| Line Active Power Flow | $P_{flow}$ | Branch End (From/To) | Multimeter (MM) via PT/CT | Active power flowing through a line/transformer end. PT/CT ratios applied by $h(x)$ |
| Line Reactive Power Flow | $Q_{flow}$ | Branch End (From/To) | Multimeter (MM) via PT/CT | Reactive power flowing through a line/transformer end. PT/CT ratios applied by $h(x)$ |
| Line Current Magnitude | $I_{m}$ | Branch End (From/To) | Multimeter (MM) via CT | Current magnitude in a line/transformer end. CT ratio applied by $h(x)$ |
| Bus Voltage Angle | $\theta$ | Bus | PMUs | Bus voltage phase angle (from PMUs only) |
| Switching Device Status | $S_{SD}$ | Branch/Bus | SD Status | Status (Open/Closed) of breakers, switches, and fuses |
| Pseudo/Virtual Measurements | $P_{pseudo}, Q_{pseudo}$ | Bus/Branch | Model/Historical Data | Non-telemetered measurements (e.g., historical load profiles, zero injections) used for observability |

#### 3.7.2 Calculated Output Quantities

| Calculated Quantity | Symbol | Type | Description |
|---------------------|--------|------|-------------|
| State Vector (Bus Voltage Magnitude) | $x_V = V$ | Estimated | Estimated voltage magnitude at every bus |
| State Vector (Bus Voltage Angle) | $x_{\theta} = \theta$ | Estimated | Estimated voltage phase angle at every bus (relative to Slack) |
| Bus Active Power Injection | $\hat{P}_{inj}$ | Derived | Calculated net active power injection at every bus |
| Bus Reactive Power Injection | $\hat{Q}_{inj}$ | Derived | Calculated net reactive power injection at every bus |
| Line Active Power Flow | $\hat{P}_{flow}$ | Derived | Calculated active power flow through every branch |
| Line Reactive Power Flow | $\hat{Q}_{flow}$ | Derived | Calculated reactive power flow through every branch |
| Line Current Magnitude | $\hat{I}_{m}$ | Derived | Calculated current magnitude through every branch |
| Measurement Residual Vector | $r = z - h(x)$ | Validation | The difference between measured and estimated values for all $z$ |
| Weighted Sum of Squares | $J(x)$ | Validation | The objective function value (cost function) |

---

## 4. Non-Functional Requirements

### 4.1 Performance & Optimization

**NFR-01 - Throughput (Ultra-Scale Target):** The engine must support continuous operation for networks up to 200,000 buses and 2,000,000 measurements. The complete estimation cycle (including initial factorization) must be optimized to achieve a result in the shortest time possible, with an initial target cycle time of < 2 seconds. This target relies critically on Gain Matrix Reuse (FR-16) for all subsequent telemetry updates

**NFR-02 - Latency (Incremental):** "Glass-to-Glass" latency shall be < 10ms for incremental updates (i.e., when $G$ is reused and only forward/backward substitution is performed)

**NFR-03 - Memory Layout:** Heavy use of Structure of Arrays (SoA) for bus/branch data to ensure coalesced memory access threads

**NFR-04 - Single Precision Support:** The engine should support float (fp32) for maximum speed in real-time mode, with an option for double (fp64) for precision mode

### 4.2 Reliability

**NFR-05 - Divergence Handling:** If the estimator diverges, it must automatically reset to a "Flat Start" (1.0 p.u. voltage, 0 degrees) and re-attempt within the same frame if time permits, or return a "Previous Known Good" state with a warning flag

### 4.3 Host-Side C++ Optimization (Latency Focus)

The Host-side code (C++) is responsible for I/O, telemetry handling, and non-parallelizable sequential logic. Since NFR-02 dictates a glass-to-glass latency of < 10ms, these components must be aggressively optimized for speed and low latency.

**NFR-06 - Allocation Avoidance:** Dynamic memory allocation (new and delete) shall be avoided entirely within the real-time processing path (Telemetry updates), favoring pre-allocated memory pools or stack allocation

**NFR-07 - Compile-Time Optimization:** Utilize C++ template metaprogramming (TMP) to:

- Generate highly specialized, non-virtual code paths
- Define and validate static data structures (like bus and branch parameters) at compile time where possible

**NFR-08 - Compiler Hints ([[assume]]):** Use the C++ `[[assume(expression)]]` attribute (or compiler-specific equivalents like `__builtin_assume`) judiciously in performance-critical loops and conditional branches to provide the compiler with guaranteed invariants, maximizing optimization opportunities

**NFR-09 - OpenMP Parallelization:** Critical host-side preprocessing tasks, such as initial data sanity checks, measurement validation, and graph construction stages within the Topology Processor (before the GPU-accelerated re-topology), shall be parallelized using OpenMP directives

**NFR-10 - Modern C++ Constructs:** Favor modern, zero-overhead C++ features like constant expressions (constexpr), move semantics, and generic lambdas for clean, inlineable, and performant code

**NFR-11 - Cache-Line Alignment:** All critical Host-side data structures (e.g., measurement buffers, topology configuration objects) accessed by the CPU in sequential loops must be explicitly aligned to the CPU's cache line size (e.g., 64 bytes) to prevent false sharing and ensure optimal data locality

**NFR-12 - Branchless Code:** Where mathematically possible, replace performance-critical conditional logic (if/else) within tight loops with arithmetic operations or compiler intrinsics (e.g., bitwise operators, std::min/std::max, or conditional moves) to minimize CPU pipeline stalls due to mispredicted branches

**NFR-13 - Thread Affinity:** For Host-side OpenMP parallel regions or background threads that handle asynchronous I/O/telemetry, set explicit thread affinity (e.g., using `sched_setaffinity` or platform-specific methods) to bind threads to specific CPU cores. This avoids costly cache misses and context switching overhead from OS re-scheduling

**NFR-14 - Instruction Set Extensions (Vectorization):** The C++ build system must enable aggressive compiler flags (e.g., `-mavx2`, `-mavx512`) to leverage modern CPU vector instructions, particularly for array operations during initial telemetry processing and initial measurement transformation that must occur on the Host. The code should be structured to aid the compiler's auto-vectorization

**NFR-15 - Just-In-Time (JIT) Compilation for Topology:** For the Topology Processor's graph traversal logic, investigate the use of a lightweight JIT compiler (like LLVM components) to generate highly optimized, on-the-fly machine code tailored to the current, known static topology structure. This minimizes the overhead of generic graph algorithms

**NFR-16 - Function Inlining Control:** The implementation must use compiler function attributes (e.g., `[[gnu::noinline]]`, `__attribute__((always_inline))`, or vendor-specific pragmas/keywords) to manually control the inlining behavior of the most performance-sensitive Host-side routines. This is necessary to balance instruction cache usage and function call overhead for the low-latency targets

**NFR-17 - Strategic CPU/GPU Workload and Data Flow Optimization:** The architecture shall prioritize optimal execution throughput by strategically balancing the computational load and data flow between the Host (CPU) and Device (GPU). This includes:

- **a) Strategic Workload Decomposition:** Breaking up the WLS calculation into distinct, optimized kernel stages (FR-09) to allow for asynchronous execution and dependency management
- **b) Overhead Assessment:** Host-side control logic must assess the latency-to-computation ratio for small tasks, executing them on the CPU (leveraging NFR-09, NFR-14) only where necessary to prevent excessive kernel launch latency from blocking the overall pipeline
- **c) Data Structure Flow:** Ensuring that the design of the internal data structures (e.g., SoA) facilitates continuous, non-blocking data movement (FR-06, FR-07) and efficient memory staging across the pipeline stages

**NFR-18 - Host Container Stability:** The core Host-side data structures responsible for storing network elements (Buses, Branches, Measurements, Devices) that are referenced via pointers or indices must use containers that guarantee iterator and pointer stability upon element insertion or deletion. The implementation must explicitly utilize `boost::stable_vector` to prevent runtime corruption when small elements are added or removed during run-time maintenance of the large, static model

**NFR-21 - Explicit Loop Parallelization (CPU):** All suitable Host-side loops identified under NFR-09 must be explicitly marked for CPU parallelization using the OpenMP directive: `#pragma omp parallel for`. This mandates aggressive use of multi-core processing for large sequential preprocessing tasks

**NFR-22 - Explicit SIMD Vectorization and Reduction (CPU):** Host-side reduction and summation loops must be explicitly marked for compiler vectorization and parallel reduction using the OpenMP directive: `#pragma omp simd reduction(+:sum_variable)`. This ensures maximum utilization of modern CPU vector registers (AVX/AVX-512) for high-speed arithmetic operations (NFR-14)

**NFR-23 - Loop Unrolling (CPU/GPU):** Critical performance loops on both the Host (CPU) and Device (CUDA kernels, referencing 5.1) must be targeted for loop unrolling using the appropriate compiler pragma (`#pragma unroll` or C++ attributes). This is mandatory to reduce loop management overhead and expose more instruction-level parallelism, directly impacting the latency targets (NFR-02)

**NFR-24 - JSF C++ Compliance:** The entire C++ codebase (Host and Device) shall strictly adhere to the JSF (Joint Strike Fighter) C++ Coding Standard to ensure high integrity, deterministic behavior, and robustness required for critical infrastructure components. This includes mandatory use of `[[nodiscard]]`, scoped enumerations, explicit constructors, and avoidance of exceptions and RTTI

### 4.4 Build and Delivery (Platform Integration)

**NFR-19 - Build Environment:** The engine source code and project configuration must be maintained as a Visual Studio CUDA project. This ensures compatibility with the Windows development and deployment environment, and enables direct use of NVIDIA's Nsight tools for profiling

**NFR-20 - Delivery Artifact:** The final output of the build process shall be a Dynamically Linked Library (DLL). This library must expose a well-defined C-compatible API (Application Programming Interface) for function calls and data interchange with external Host applications

---

## 5. Technical Implementation Guidelines

### 5.1 Solver Strategy (Critical for Scale)

Given the massive size of the Gain Matrix $G$ ($N \approx 200,000 \times 200,000$), the implementation must aggressively prioritize memory and computation efficiency on the single CUDA device:

- **Hierarchical Sparse Storage:** Use blocked or tile-based formats (in addition to CSR/CSC) for large matrices to optimize memory access patterns and enable efficient parallel processing
- **Graph Partitioning:** Implement graph partitioning algorithms (e.g., using METIS or similar GPU-accelerated methods) to decompose the network into smaller, weakly coupled subproblems. This can optimize memory locality and processing efficiency on the single CUDA device
- **Reordering:** Apply advanced sparse matrix reordering algorithms (e.g., Minimum Degree or Nested Dissection) to minimize the fill-in during Cholesky factorization, thereby managing memory consumption and improving factorization speed
- **Kernel Optimization (Shared Memory & Tiling):** All performance-critical CUDA kernels, especially those for Sparse Matrix-Vector Multiplication (SpMV) and Sparse Matrix-Matrix Multiplication (SpGEMM), must utilize GPU Shared Memory and Tiling (blocking) techniques. This is mandatory to reduce costly global memory access, enhance data reuse, and achieve the stringent NFR-02 incremental latency target

### 5.2 Host-Side Data Structures (Mandatory)

For all critical Host-side data management tasks, the C++ implementation utilizes `std::unordered_map` with custom hashers for high-speed lookup. This map serves as the required lookup layer to translate external, human-readable string identifiers (e.g., device names, bus names) provided via the API into internal, contiguous integer indices used for all GPU-based mathematical operations, ensuring optimal computational efficiency.

**Note:** The implementation uses standard library containers to avoid external dependencies and ensure compatibility with NVCC compilation. Performance is maintained through pre-reserved capacity and efficient hashing.

**Language:** C++17 / C++20

**CUDA Toolkit:** 11.0+

**Libraries:**

- **cuSPARSE:** For sparse matrix operations (SpMV, SpGEMM)
- **cuSOLVER:** For Cholesky factorization (`cusolverSpScsrlsvchol`)
- **Thrust:** For fast vector operations (reductions, sorting)
- **Boost Library:** Specifically `boost::stable_vector` - Mandatory for core Host-side network data storage (NFR-18)

**Architecture:** Component-based (TopologyProcessor, MeasurementHandler, WLSSolver)

---

## 6. Future Scope

- **Dynamic State Estimation:** Incorporating time-domain models (e.g., synchronous machine models) and using filtering techniques (like Extended Kalman Filter) to track state changes dynamically
- **Cybersecurity Anomaly Detection:** Real-time monitoring of measurement patterns and WLS residual behavior to identify potential cyber attacks (e.g., false data injection attacks)
