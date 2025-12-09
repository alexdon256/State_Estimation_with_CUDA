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
 * @file wls_solver.cu
 * @brief WLS Solver implementation
 * 
 * Implements the Weighted Least Squares solver using GPU acceleration.
 * Uses cuSPARSE for sparse matrix operations and cuSOLVER for Cholesky factorization.
 */

#include "../include/wls_solver.cuh"
#include "../include/kernels.cuh"
#include "../include/jsf_compliance.h"
#include <cuda_runtime.h>
#include <chrono>

namespace sle {

//=============================================================================
// WLSSolver Implementation
//=============================================================================

WLSSolver::WLSSolver(cudaStream_t stream)
    : stream_(stream)
    , matrix_mgr_(std::make_unique<SparseMatrixManager>(stream))
    , state_(nullptr)
    , initialized_(false)
    , last_iterations_(0)
    , last_status_(ConvergenceStatus::IN_PROGRESS)
{
    // Create timing events
    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
}

WLSSolver::~WLSSolver() {
    freeSolverState();
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
}

WLSSolver::WLSSolver(WLSSolver&& other) noexcept
    : stream_(other.stream_)
    , matrix_mgr_(std::move(other.matrix_mgr_))
    , state_(std::move(other.state_))
    , initialized_(other.initialized_)
    , last_iterations_(other.last_iterations_)
    , last_status_(other.last_status_)
    , start_event_(other.start_event_)
    , stop_event_(other.stop_event_)
{
    other.stream_ = nullptr;
    other.start_event_ = nullptr;
    other.stop_event_ = nullptr;
    other.initialized_ = false;
}

WLSSolver& WLSSolver::operator=(WLSSolver&& other) noexcept {
    if (this != &other) {
        freeSolverState();
        if (start_event_) cudaEventDestroy(start_event_);
        if (stop_event_) cudaEventDestroy(stop_event_);
        
        stream_ = other.stream_;
        matrix_mgr_ = std::move(other.matrix_mgr_);
        state_ = std::move(other.state_);
        initialized_ = other.initialized_;
        last_iterations_ = other.last_iterations_;
        last_status_ = other.last_status_;
        start_event_ = other.start_event_;
        stop_event_ = other.stop_event_;
        
        other.stream_ = nullptr;
        other.start_event_ = nullptr;
        other.stop_event_ = nullptr;
        other.initialized_ = false;
    }
    return *this;
}

//=============================================================================
// Initialization
//=============================================================================

cudaError_t WLSSolver::initialize(
    const DeviceBusData& buses,
    const DeviceBranchData& branches,
    const DeviceMeasurementData& measurements,
    const DeviceYbusMatrix& ybus)
{
    // Allocate solver state buffers
    cudaError_t err = allocateSolverState(buses.count, measurements.count);
    if (err != cudaSuccess) return err;
    
    state_->n_buses = buses.count;
    state_->n_states = 2 * buses.count - 1;  // Angles (except slack) + all magnitudes
    state_->n_measurements = measurements.count;
    
    // Analyze Jacobian sparsity pattern
    err = matrix_mgr_->analyzeJacobianPattern(measurements, buses, branches, state_->H);
    if (err != cudaSuccess) return err;
    
    state_->H_pattern_valid = true;
    state_->G_factor_valid = false;
    
    initialized_ = true;
    return cudaSuccess;
}

void WLSSolver::invalidate() {
    if (state_) {
        state_->H_pattern_valid = false;
        state_->G_factor_valid = false;
    }
    initialized_ = false;
}

//=============================================================================
// Solver State Management
//=============================================================================

cudaError_t WLSSolver::allocateSolverState(int32_t n_buses, int32_t n_measurements) {
    if (state_) {
        freeSolverState();
    }
    
    state_ = std::make_unique<SolverState>();
    
    int32_t n_states = 2 * n_buses - 1;
    
    cudaError_t err;
    
    // Allocate RHS vector
    err = cudaMalloc(&state_->d_rhs, n_states * sizeof(Real));
    if (err != cudaSuccess) return err;
    
    // Allocate solution vector
    err = cudaMalloc(&state_->d_delta_x, n_states * sizeof(Real));
    if (err != cudaSuccess) return err;
    
    // Allocate temporary vectors
    err = cudaMalloc(&state_->d_temp_meas, n_measurements * sizeof(Real));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&state_->d_temp_state, n_states * sizeof(Real));
    if (err != cudaSuccess) return err;
    
    // Allocate Huber weights
    err = cudaMalloc(&state_->d_huber_weights, n_measurements * sizeof(Real));
    if (err != cudaSuccess) return err;
    
    // Allocate scalars
    err = cudaMalloc(&state_->d_max_mismatch, sizeof(Real));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&state_->d_objective, sizeof(Real));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&state_->d_validity_flag, sizeof(int32_t));
    if (err != cudaSuccess) return err;
    
    // Initialize scalars
    cudaMemsetAsync(state_->d_max_mismatch, 0, sizeof(Real), stream_);
    cudaMemsetAsync(state_->d_objective, 0, sizeof(Real), stream_);
    cudaMemsetAsync(state_->d_validity_flag, 0, sizeof(int32_t), stream_);
    
    return cudaSuccess;
}

void WLSSolver::freeSolverState() {
    if (!state_) return;
    
    if (state_->d_rhs) cudaFree(state_->d_rhs);
    if (state_->d_delta_x) cudaFree(state_->d_delta_x);
    if (state_->d_temp_meas) cudaFree(state_->d_temp_meas);
    if (state_->d_temp_state) cudaFree(state_->d_temp_state);
    if (state_->d_huber_weights) cudaFree(state_->d_huber_weights);
    if (state_->d_max_mismatch) cudaFree(state_->d_max_mismatch);
    if (state_->d_objective) cudaFree(state_->d_objective);
    if (state_->d_validity_flag) cudaFree(state_->d_validity_flag);
    
    matrix_mgr_->freeCSR(state_->H);
    matrix_mgr_->freeCSR(state_->G);
    
    state_.reset();
}

//=============================================================================
// Main Solve Function
//=============================================================================

EstimationResult WLSSolver::solve(
    DeviceBusData& buses,
    DeviceBranchData& branches,
    DeviceMeasurementData& measurements,
    const DeviceYbusMatrix& ybus,
    const SolverConfig& config)
{
    EstimationResult result;
    result.status = ConvergenceStatus::IN_PROGRESS;
    
    if (!initialized_ || !state_) {
        result.status = ConvergenceStatus::NOT_OBSERVABLE;
        return result;
    }
    
    // Start timing
    cudaEventRecord(start_event_, stream_);
    
    // Apply flat start if requested
    if (config.use_flat_start) {
        dim3 block(BLOCK_SIZE_STANDARD);
        dim3 grid = compute_grid_size(buses.count, BLOCK_SIZE_STANDARD);
        
        flatStartKernel<<<grid, block, 0, stream_>>>(
            buses.d_v_mag,
            buses.d_v_angle,
            buses.d_v_setpoint,
            buses.d_bus_type,
            buses.count);
    }
    
    // Determine max iterations based on mode
    int32_t max_iter = (config.mode == EstimationMode::REALTIME) ?
        config.max_iterations : MAX_PRECISION_ITERATIONS;
    
    Real tolerance = config.convergence_tolerance;
    bool converged = false;
    bool diverged = false;
    
    // NFR-05: Track retry attempts for divergence handling
    constexpr int32_t MAX_RETRY_ATTEMPTS = 2;
    int32_t retry_count = 0;
    bool retry_with_flat_start = false;
    
    do {
        // NFR-05: If retrying after divergence, apply flat start
        if (retry_with_flat_start) {
            dim3 block(BLOCK_SIZE_STANDARD);
            dim3 grid = compute_grid_size(buses.count, BLOCK_SIZE_STANDARD);
            
            flatStartKernel<<<grid, block, 0, stream_>>>(
                buses.d_v_mag,
                buses.d_v_angle,
                buses.d_v_setpoint,
                buses.d_bus_type,
                buses.count);
            
            // Reset state for fresh start
            diverged = false;
            converged = false;
            retry_with_flat_start = false;
            
            // Invalidate G factor to force recomputation
            if (state_) {
                state_->G_factor_valid = false;
            }
        }
        
        // WLS iteration loop
        for (int32_t iter = 0; iter < max_iter && !converged && !diverged; ++iter) {
            // Single WLS iteration
            cudaError_t err = singleIteration(buses, branches, measurements, ybus,
                                              config.use_robust_estimation,
                                              config.huber_gamma);
            
            if (err != cudaSuccess) {
                result.status = ConvergenceStatus::SINGULAR_MATRIX;
                diverged = true;
                break;
            }
            
            // Check convergence
            err = checkConvergence(tolerance, converged, diverged);
            if (err != cudaSuccess) {
                result.status = ConvergenceStatus::DIVERGED;
                diverged = true;
                break;
            }
            
            result.iterations = iter + 1;
            
            // Check time limit for real-time mode
            if (config.mode == EstimationMode::REALTIME && config.time_limit_ms > 0) {
                cudaEventRecord(stop_event_, stream_);
                cudaEventSynchronize(stop_event_);
                float elapsed_ms;
                cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_);
                if (elapsed_ms > config.time_limit_ms) {
                    // No time for retry in real-time mode
                    result.status = ConvergenceStatus::MAX_ITERATIONS;
                    break;
                }
            }
        }
        
        // NFR-05: If diverged and have retry attempts left, reset to flat start
        if (diverged && retry_count < MAX_RETRY_ATTEMPTS && 
            config.mode != EstimationMode::REALTIME) {
            retry_with_flat_start = true;
            retry_count++;
            result.iterations = 0;  // Reset iteration count for retry
        }
        
    } while (retry_with_flat_start);
    
    // Determine final status
    if (result.status == ConvergenceStatus::IN_PROGRESS) {
        if (converged) {
            result.status = ConvergenceStatus::CONVERGED;
        } else if (diverged) {
            result.status = ConvergenceStatus::DIVERGED;
        } else {
            result.status = ConvergenceStatus::MAX_ITERATIONS;
        }
    }
    
    // Stop timing
    cudaEventRecord(stop_event_, stream_);
    cudaEventSynchronize(stop_event_);
    cudaEventElapsedTime(&result.computation_time_ms, start_event_, stop_event_);
    
    // Copy results to host
    cudaMemcpy(&result.max_mismatch, state_->d_max_mismatch, 
               sizeof(Real), cudaMemcpyDeviceToHost);
    
    // Compute final objective
    Real obj_val = 0.0f;
    (void)launchResidualKernels(measurements, &obj_val, stream_);
    result.objective_value = obj_val;
    
    // Find largest residual for bad data detection
    // (simplified - would use findLargestResidualKernel in production)
    result.largest_residual = 0.0f;
    result.largest_residual_idx = -1;
    result.bad_data_count = 0;
    
    last_iterations_ = result.iterations;
    last_status_ = result.status;
    last_result_ = result;
    
    return result;
}

//=============================================================================
// Single Iteration
//=============================================================================

cudaError_t WLSSolver::singleIteration(
    DeviceBusData& buses,
    DeviceBranchData& branches,
    DeviceMeasurementData& measurements,
    const DeviceYbusMatrix& ybus,
    bool use_huber,
    Real huber_gamma)
{
    cudaError_t err;
    
    // Step 1: Compute power flows and measurement function h(x)
    err = launchPowerFlowKernels(buses, branches, ybus, stream_);
    if (err != cudaSuccess) return err;
    
    err = launchMeasurementFunctionKernel(measurements, buses, branches, stream_);
    if (err != cudaSuccess) return err;
    
    // Step 2: Compute residuals r = z - h(x)
    Real obj_value;
    err = launchResidualKernels(measurements, &obj_value, stream_);
    if (err != cudaSuccess) return err;
    
    // Step 3: Apply Huber weights if using robust estimation
    if (use_huber) {
        err = applyHuberWeights(measurements, huber_gamma);
        if (err != cudaSuccess) return err;
        state_->using_huber_weights = true;
    } else {
        state_->using_huber_weights = false;
    }
    
    // Step 4: Update Jacobian matrix H(x)
    err = computeJacobian(buses, branches, measurements, ybus);
    if (err != cudaSuccess) return err;
    
    // Step 5: Form and factorize Gain matrix if needed
    // NOTE: When using Huber weights, we must recompute G each iteration
    // because the weights change with the residuals
    if (!state_->G_factor_valid || state_->using_huber_weights) {
        err = formGainMatrix(measurements);
        if (err != cudaSuccess) return err;
    }
    
    // Step 6: Solve normal equations G * delta_x = H^T * W * r
    err = solveNormalEquations(measurements);
    if (err != cudaSuccess) return err;
    
    // Step 7: Update state x = x + delta_x
    err = launchStateUpdateKernels(buses, state_->d_delta_x, 
                                   state_->n_states, state_->d_max_mismatch,
                                   stream_);
    
    return err;
}

//=============================================================================
// Internal Methods
//=============================================================================

cudaError_t WLSSolver::computeJacobian(
    const DeviceBusData& buses,
    const DeviceBranchData& branches,
    const DeviceMeasurementData& measurements,
    const DeviceYbusMatrix& ybus)
{
    // Recompute Jacobian values using current state
    return matrix_mgr_->computeJacobianValues(measurements, buses, branches, 
                                              ybus, state_->H);
}

cudaError_t WLSSolver::formGainMatrix(const DeviceMeasurementData& measurements) {
    // Use Huber weights if active, otherwise use base weights
    const Real* weights = state_->using_huber_weights ? 
        state_->d_huber_weights : measurements.d_weight;
    
    // Compute G = H^T * W * H
    cudaError_t err = matrix_mgr_->computeGainMatrix(
        state_->H, weights, measurements.count, state_->G);
    if (err != cudaSuccess) return err;
    
    // Factorize G = L * L^T
    err = matrix_mgr_->factorizeCholesky(state_->G);
    if (err != cudaSuccess) return err;
    
    state_->G_factor_valid = true;
    return cudaSuccess;
}

cudaError_t WLSSolver::solveNormalEquations(const DeviceMeasurementData& measurements) {
    // Use Huber weights if active, otherwise use base weights
    const Real* weights = state_->using_huber_weights ? 
        state_->d_huber_weights : measurements.d_weight;
    
    // Compute RHS: b = H^T * W * r
    dim3 block(BLOCK_SIZE_STANDARD);
    dim3 grid = compute_grid_size(state_->n_states, BLOCK_SIZE_STANDARD);
    
    // Initialize RHS to zero
    cudaMemsetAsync(state_->d_rhs, 0, state_->n_states * sizeof(Real), stream_);
    
    // Compute H^T * W * r
    computeRHSVectorKernel<<<grid, block, 0, stream_>>>(
        state_->d_rhs,
        state_->H.d_row_ptr,
        state_->H.d_col_ind,
        state_->H.d_values,
        weights,
        measurements.d_residual,
        measurements.count,
        state_->n_states);
    
    // Solve G * delta_x = b
    return matrix_mgr_->solveCholesky(state_->d_rhs, state_->d_delta_x, state_->n_states);
}

cudaError_t WLSSolver::checkConvergence(Real tolerance, bool& converged, bool& diverged) {
    // Copy max mismatch to host
    cudaMemcpy(&state_->h_max_mismatch, state_->d_max_mismatch,
               sizeof(Real), cudaMemcpyDeviceToHost);
    
    converged = (state_->h_max_mismatch < tolerance);
    
    // Check for NaN/Inf
    cudaMemsetAsync(state_->d_validity_flag, 0, sizeof(int32_t), stream_);
    
    dim3 block(BLOCK_SIZE_STANDARD);
    dim3 grid = compute_grid_size(state_->n_states, BLOCK_SIZE_STANDARD);
    
    checkValidityKernel<<<grid, block, 0, stream_>>>(
        state_->d_validity_flag,
        state_->d_delta_x,
        state_->n_states);
    
    cudaMemcpy(&state_->h_validity_flag, state_->d_validity_flag,
               sizeof(int32_t), cudaMemcpyDeviceToHost);
    
    diverged = (state_->h_validity_flag != 0);
    
    return cudaGetLastError();
}

cudaError_t WLSSolver::applyHuberWeights(
    const DeviceMeasurementData& measurements,
    Real gamma)
{
    dim3 block(BLOCK_SIZE_STANDARD);
    dim3 grid = compute_grid_size(measurements.count, BLOCK_SIZE_STANDARD);
    
    // Use actual sigma values from measurement data for Huber M-estimator
    computeHuberWeightsKernel<<<grid, block, 0, stream_>>>(
        state_->d_huber_weights,
        measurements.d_weight,
        measurements.d_residual,
        measurements.d_sigma,  // Proper sigma values
        gamma,
        measurements.d_is_active,
        measurements.count);
    
    return cudaGetLastError();
}

//=============================================================================
// Result Access
//=============================================================================

Real WLSSolver::getObjectiveValue() const {
    return last_result_.objective_value;
}

Real WLSSolver::getMaxMismatch() const {
    return state_ ? state_->h_max_mismatch : 0.0f;
}

bool WLSSolver::hasInvalidValues() const {
    return state_ ? (state_->h_validity_flag != 0) : false;
}

void WLSSolver::setStream(cudaStream_t stream) {
    stream_ = stream;
    // Update matrix manager stream too
    // (would need to recreate or add setStream method)
}

//=============================================================================
// WLSSolverGraph Implementation (Placeholder)
//=============================================================================

WLSSolverGraph::WLSSolverGraph(cudaStream_t stream)
    : stream_(stream)
    , graph_(nullptr)
    , graph_exec_(nullptr)
    , graph_valid_(false)
    , base_solver_(std::make_unique<WLSSolver>(stream))
{
}

WLSSolverGraph::~WLSSolverGraph() {
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (graph_) cudaGraphDestroy(graph_);
}

cudaError_t WLSSolverGraph::captureGraph(
    DeviceBusData& buses,
    DeviceBranchData& branches,
    DeviceMeasurementData& measurements,
    const DeviceYbusMatrix& ybus)
{
    // Start capture
    cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
    
    // Record one iteration
    (void)base_solver_->singleIteration(buses, branches, measurements, ybus, false, 1.5f);
    
    // End capture
    cudaStreamEndCapture(stream_, &graph_);
    
    // Create executable graph
    cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0);
    
    graph_valid_ = true;
    return cudaGetLastError();
}

cudaError_t WLSSolverGraph::executeGraph() {
    if (!graph_valid_ || !graph_exec_) {
        return cudaErrorNotReady;
    }
    
    return cudaGraphLaunch(graph_exec_, stream_);
}

void WLSSolverGraph::invalidateGraph() {
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
    graph_valid_ = false;
}

//=============================================================================
// ObservabilityAnalyzer Implementation (Placeholder)
//=============================================================================

ObservabilityAnalyzer::ObservabilityAnalyzer(cudaStream_t stream)
    : stream_(stream)
    , island_count_(0)
    , d_visited_(nullptr)
    , d_island_id_(nullptr)
    , d_frontier_(nullptr)
    , max_buses_(0)
{
}

ObservabilityAnalyzer::~ObservabilityAnalyzer() {
    if (d_visited_) cudaFree(d_visited_);
    if (d_island_id_) cudaFree(d_island_id_);
    if (d_frontier_) cudaFree(d_frontier_);
}

bool ObservabilityAnalyzer::analyze(
    const DeviceBusData& buses,
    const DeviceBranchData& branches,
    const DeviceMeasurementData& measurements,
    const DeviceYbusMatrix& ybus)
{
    // Placeholder implementation
    // Full implementation would use parallel BFS for connectivity analysis
    
    island_count_ = 1;  // Assume single island
    unobservable_buses_.clear();
    
    // Check basic observability conditions
    // - Need at least n_buses - 1 independent measurements
    // - Network must be connected
    
    int32_t n_buses = buses.count;
    int32_t n_meas = measurements.active_count;
    
    // Simple check: need enough measurements
    if (n_meas < n_buses - 1) {
        return false;
    }
    
    return true;
}

} // namespace sle

