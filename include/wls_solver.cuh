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
 * @file wls_solver.cuh
 * @brief Weighted Least Squares (WLS) solver for state estimation
 * 
 * Implements the core WLS algorithm using GPU acceleration:
 * 1. Jacobian matrix H(x) computation
 * 2. Gain matrix G = H^T W H formation via SpGEMM
 * 3. Cholesky factorization G = L L^T
 * 4. Forward/backward substitution to solve for state update
 * 
 * Supports:
 * - Real-time mode with gain matrix reuse (FR-11, FR-16)
 * - Precision mode with full refactorization (FR-12)
 * - Robust estimation with Huber M-estimator (FR-17)
 * - CUDA Streams/Graphs for efficient execution (FR-09)
 * 
 * @note Uses cuSPARSE and cuSOLVER for sparse linear algebra.
 */

#ifndef WLS_SOLVER_CUH
#define WLS_SOLVER_CUH

#include "sle_types.cuh"
#include "sparse_matrix.cuh"
#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace sle {

//=============================================================================
// SECTION 1: Solver State Structure
//=============================================================================

/**
 * @brief Internal solver state maintained between iterations
 * 
 * Contains all GPU-allocated buffers needed for WLS iteration.
 * Allocated once during initialization and reused.
 */
struct SolverState {
    // State dimensions
    int32_t n_buses;                ///< Number of buses
    int32_t n_states;               ///< Number of state variables (2*n_buses - 1)
    int32_t n_measurements;         ///< Number of active measurements
    
    // Jacobian matrix H [n_measurements x n_states]
    DeviceCSRMatrix H;
    bool H_pattern_valid;           ///< True if Jacobian pattern is current
    
    // Gain matrix G = H^T W H [n_states x n_states]
    DeviceCSRMatrix G;
    bool G_factor_valid;            ///< True if Cholesky factor is current (FR-16)
    
    // RHS vector b = H^T W r [n_states]
    Real* d_rhs;
    
    // Solution vector delta_x [n_states]
    Real* d_delta_x;
    
    // Temporary vectors for computation
    Real* d_temp_meas;              ///< Temporary measurement-sized vector [n_meas]
    Real* d_temp_state;             ///< Temporary state-sized vector [n_states]
    
    // Huber weights (for robust estimation)
    Real* d_huber_weights;          ///< Modified weights when using Huber
    bool using_huber_weights;       ///< Flag indicating Huber weights are active
    
    // Convergence tracking
    Real* d_max_mismatch;           ///< Device-side max mismatch
    Real* d_objective;              ///< Device-side objective value
    int32_t* d_validity_flag;       ///< NaN/Inf detection flag
    
    // Host-side scalars for result retrieval
    Real h_max_mismatch;
    Real h_objective;
    int32_t h_validity_flag;
    
    /// Default constructor
    SolverState() :
        n_buses(0), n_states(0), n_measurements(0),
        H_pattern_valid(false), G_factor_valid(false),
        d_rhs(nullptr), d_delta_x(nullptr),
        d_temp_meas(nullptr), d_temp_state(nullptr),
        d_huber_weights(nullptr), using_huber_weights(false),
        d_max_mismatch(nullptr), d_objective(nullptr),
        d_validity_flag(nullptr),
        h_max_mismatch(0), h_objective(0), h_validity_flag(0) {
        H = {};
        G = {};
    }
};

//=============================================================================
// SECTION 2: WLS Solver Class
//=============================================================================

/**
 * @class WLSSolver
 * @brief GPU-accelerated Weighted Least Squares solver
 * 
 * Implements the core state estimation algorithm entirely on GPU (FR-08).
 * Manages solver state, sparse matrices, and CUDA resources.
 * 
 * Usage pattern:
 * 1. Create solver with device data
 * 2. Call initialize() once after model upload
 * 3. Call solve() for each estimation cycle
 * 4. Use getResult() to retrieve solution
 * 
 * The solver automatically manages:
 * - Jacobian sparsity pattern analysis
 * - Gain matrix computation and factorization
 * - Hot start vs flat start initialization
 * - Convergence monitoring
 * 
 * Thread-safety: NOT thread-safe. Use one instance per estimation thread.
 */
class WLSSolver {
public:
    /**
     * @brief Constructor
     * 
     * @param stream CUDA stream for asynchronous execution
     */
    explicit WLSSolver(cudaStream_t stream = nullptr);
    
    /**
     * @brief Destructor - releases all GPU resources
     */
    ~WLSSolver();
    
    // Disable copy (large GPU resources)
    WLSSolver(const WLSSolver&) = delete;
    WLSSolver& operator=(const WLSSolver&) = delete;
    
    // Enable move
    WLSSolver(WLSSolver&& other) noexcept;
    WLSSolver& operator=(WLSSolver&& other) noexcept;

    //=========================================================================
    // Initialization
    //=========================================================================
    
    /**
     * @brief Initialize solver for given network size
     * 
     * Allocates all GPU buffers and analyzes Jacobian sparsity pattern.
     * Must be called before first solve() and after any topology change.
     * 
     * @param buses Device bus data
     * @param branches Device branch data
     * @param measurements Device measurement data
     * @param ybus Device Ybus matrix
     * @return cudaSuccess on success
     */
    [[nodiscard]] cudaError_t initialize(
        const DeviceBusData& buses,
        const DeviceBranchData& branches,
        const DeviceMeasurementData& measurements,
        const DeviceYbusMatrix& ybus);
    
    /**
     * @brief Check if solver is initialized
     */
    [[nodiscard]] bool isInitialized() const { return initialized_; }
    
    /**
     * @brief Force reinitialization on next solve
     * 
     * Call this after topology changes to rebuild Jacobian pattern.
     */
    void invalidate();
    
    /**
     * @brief Force Gain matrix refactorization on next solve
     * 
     * Call this after topology changes (FR-05) or for precision mode (FR-12).
     */
    void invalidateGainMatrix() { 
        if (state_) state_->G_factor_valid = false; 
    }

    //=========================================================================
    // Solving
    //=========================================================================
    
    /**
     * @brief Execute WLS estimation cycle (FR-08)
     * 
     * Performs iterative WLS until convergence or limit reached:
     * 1. Compute power flows and measurement function h(x)
     * 2. Compute residuals r = z - h(x)
     * 3. Update Jacobian H(x)
     * 4. If needed: Form G = H^T W H and factorize
     * 5. Solve G * delta_x = H^T W r
     * 6. Update state: x = x + delta_x
     * 7. Check convergence
     * 
     * @param buses Device bus data (state updated in place)
     * @param branches Device branch data (flows updated)
     * @param measurements Device measurement data (residuals updated)
     * @param ybus Device Ybus matrix
     * @param config Solver configuration
     * @return Estimation result with convergence status
     */
    [[nodiscard]] EstimationResult solve(
        DeviceBusData& buses,
        DeviceBranchData& branches,
        DeviceMeasurementData& measurements,
        const DeviceYbusMatrix& ybus,
        const SolverConfig& config);
    
    /**
     * @brief Execute single WLS iteration
     * 
     * Useful for debugging or custom iteration control.
     * 
     * @param buses Device bus data
     * @param branches Device branch data
     * @param measurements Device measurement data
     * @param ybus Device Ybus matrix
     * @param use_huber Use Huber weights
     * @param huber_gamma Huber threshold
     * @return cudaSuccess on success
     */
    [[nodiscard]] cudaError_t singleIteration(
        DeviceBusData& buses,
        DeviceBranchData& branches,
        DeviceMeasurementData& measurements,
        const DeviceYbusMatrix& ybus,
        bool use_huber = false,
        Real huber_gamma = DEFAULT_HUBER_GAMMA);

    //=========================================================================
    // Result Access
    //=========================================================================
    
    /**
     * @brief Get current objective function value J(x)
     */
    [[nodiscard]] Real getObjectiveValue() const;
    
    /**
     * @brief Get maximum state mismatch
     */
    [[nodiscard]] Real getMaxMismatch() const;
    
    /**
     * @brief Get number of iterations in last solve
     */
    [[nodiscard]] int32_t getIterationCount() const { return last_iterations_; }
    
    /**
     * @brief Check if last solve detected invalid values
     */
    [[nodiscard]] bool hasInvalidValues() const;

    //=========================================================================
    // Configuration
    //=========================================================================
    
    /**
     * @brief Set CUDA stream for all operations
     */
    void setStream(cudaStream_t stream);
    
    /**
     * @brief Get CUDA stream
     */
    [[nodiscard]] cudaStream_t getStream() const { return stream_; }
    
    /**
     * @brief Get sparse matrix manager for advanced operations
     */
    [[nodiscard]] SparseMatrixManager& getMatrixManager() { return *matrix_mgr_; }

private:
    // CUDA resources
    cudaStream_t stream_;
    std::unique_ptr<SparseMatrixManager> matrix_mgr_;
    
    // Solver state
    std::unique_ptr<SolverState> state_;
    bool initialized_;
    
    // Iteration tracking
    int32_t last_iterations_;
    ConvergenceStatus last_status_;
    EstimationResult last_result_;
    
    // Timing (for performance monitoring)
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    
    // Internal methods
    [[nodiscard]] cudaError_t allocateSolverState(
        int32_t n_buses,
        int32_t n_measurements);
    
    void freeSolverState();
    
    [[nodiscard]] cudaError_t computeJacobian(
        const DeviceBusData& buses,
        const DeviceBranchData& branches,
        const DeviceMeasurementData& measurements,
        const DeviceYbusMatrix& ybus);
    
    [[nodiscard]] cudaError_t formGainMatrix(
        const DeviceMeasurementData& measurements);
    
    [[nodiscard]] cudaError_t solveNormalEquations(
        const DeviceMeasurementData& measurements);
    
    [[nodiscard]] cudaError_t checkConvergence(
        Real tolerance,
        bool& converged,
        bool& diverged);
    
    [[nodiscard]] cudaError_t applyHuberWeights(
        const DeviceMeasurementData& measurements,
        Real gamma);
};

//=============================================================================
// SECTION 3: CUDA Graph-Based Solver (FR-09)
//=============================================================================

/**
 * @class WLSSolverGraph
 * @brief CUDA Graph-optimized WLS solver for maximum performance
 * 
 * Uses CUDA Graphs to capture and replay the WLS iteration sequence,
 * eliminating kernel launch overhead for repeated executions.
 * 
 * Best suited for real-time mode where:
 * - Network topology is stable (FR-16)
 * - Only measurement values change
 * - Gain matrix factorization is reused
 * 
 * @note Requires CUDA 10.0+ and Compute Capability 7.0+
 */
class WLSSolverGraph {
public:
    /**
     * @brief Constructor
     */
    explicit WLSSolverGraph(cudaStream_t stream = nullptr);
    
    /**
     * @brief Destructor
     */
    ~WLSSolverGraph();
    
    // Disable copy
    WLSSolverGraph(const WLSSolverGraph&) = delete;
    WLSSolverGraph& operator=(const WLSSolverGraph&) = delete;

    /**
     * @brief Capture WLS iteration as CUDA graph
     * 
     * Must be called once after initialization and whenever
     * the iteration structure changes (topology, measurement count).
     * 
     * @param buses Device bus data
     * @param branches Device branch data
     * @param measurements Device measurement data
     * @param ybus Device Ybus matrix
     * @return cudaSuccess on success
     */
    [[nodiscard]] cudaError_t captureGraph(
        DeviceBusData& buses,
        DeviceBranchData& branches,
        DeviceMeasurementData& measurements,
        const DeviceYbusMatrix& ybus);
    
    /**
     * @brief Execute captured graph for one WLS iteration
     * 
     * Much faster than individual kernel launches for repeated calls.
     * 
     * @return cudaSuccess on success
     */
    [[nodiscard]] cudaError_t executeGraph();
    
    /**
     * @brief Check if graph is captured and valid
     */
    [[nodiscard]] bool isGraphValid() const { return graph_valid_; }
    
    /**
     * @brief Invalidate captured graph (forces recapture)
     */
    void invalidateGraph();

private:
    cudaStream_t stream_;
    cudaGraph_t graph_;
    cudaGraphExec_t graph_exec_;
    bool graph_valid_;
    
    std::unique_ptr<WLSSolver> base_solver_;
};

//=============================================================================
// SECTION 4: Observability Analysis (FR-13)
//=============================================================================

/**
 * @class ObservabilityAnalyzer
 * @brief GPU-accelerated observability analysis
 * 
 * Performs fast graph-based check to determine if network is
 * mathematically solvable given current topology and measurements.
 * 
 * Uses parallel BFS/connectivity analysis on GPU.
 */
class ObservabilityAnalyzer {
public:
    explicit ObservabilityAnalyzer(cudaStream_t stream = nullptr);
    ~ObservabilityAnalyzer();
    
    /**
     * @brief Analyze network observability
     * 
     * Checks if the measurement configuration is sufficient for
     * unique state estimation.
     * 
     * @param buses Device bus data
     * @param branches Device branch data
     * @param measurements Device measurement data
     * @param ybus Device Ybus matrix
     * @return true if network is observable
     */
    [[nodiscard]] bool analyze(
        const DeviceBusData& buses,
        const DeviceBranchData& branches,
        const DeviceMeasurementData& measurements,
        const DeviceYbusMatrix& ybus);
    
    /**
     * @brief Get number of connected islands
     */
    [[nodiscard]] int32_t getIslandCount() const { return island_count_; }
    
    /**
     * @brief Get indices of unobservable buses
     */
    [[nodiscard]] const std::vector<int32_t>& getUnobservableBuses() const {
        return unobservable_buses_;
    }

private:
    cudaStream_t stream_;
    int32_t island_count_;
    std::vector<int32_t> unobservable_buses_;
    
    // Device buffers for BFS
    int32_t* d_visited_;
    int32_t* d_island_id_;
    int32_t* d_frontier_;
    int32_t max_buses_;
};

} // namespace sle

#endif // WLS_SOLVER_CUH

