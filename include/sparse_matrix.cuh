/**
 * SLE Engine - CUDA-Accelerated State Load Estimator
 * Copyright (c) 2025, Oleksandr Don
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @file sparse_matrix.cuh
 * @brief GPU-accelerated sparse matrix operations for WLS solver
 * 
 * Implements CSR/CSC format operations using cuSPARSE library.
 * Provides efficient SpMV, SpGEMM, and matrix assembly routines
 * optimized for the state estimation problem.
 * 
 * Key features:
 * - Ybus admittance matrix construction and updates (FR-05)
 * - Jacobian matrix H assembly with parallel computation (FR-08)
 * - Gain matrix G = H^T W H formation via SpGEMM (FR-08)
 * - Matrix reordering for reduced fill-in (Section 5.1)
 * 
 * @note All operations use shared memory tiling where beneficial (Section 5.1)
 */

#ifndef SPARSE_MATRIX_CUH
#define SPARSE_MATRIX_CUH

#include "sle_types.cuh"
#include <cusparse.h>
#include <cusolverSp.h>

namespace sle {

//=============================================================================
// SECTION 1: Sparse Matrix Manager Class
//=============================================================================

/**
 * @class SparseMatrixManager
 * @brief Manages sparse matrix operations on GPU with cuSPARSE/cuSOLVER
 * 
 * This class encapsulates all sparse matrix functionality including:
 * - Memory allocation and deallocation for CSR matrices
 * - Ybus matrix construction from branch data
 * - Jacobian matrix H parallel construction
 * - Gain matrix G = H^T W H computation
 * - Cholesky factorization and triangular solves
 * 
 * Thread-safety: NOT thread-safe. Use one instance per CUDA stream.
 */
class SparseMatrixManager {
public:
    /**
     * @brief Constructor - initializes cuSPARSE and cuSOLVER handles
     * @param stream CUDA stream for asynchronous operations
     */
    explicit SparseMatrixManager(cudaStream_t stream = nullptr);
    
    /**
     * @brief Destructor - releases all GPU resources and library handles
     */
    ~SparseMatrixManager();
    
    // Disable copy (JSF compliance)
    SparseMatrixManager(const SparseMatrixManager&) = delete;
    SparseMatrixManager& operator=(const SparseMatrixManager&) = delete;
    
    // Enable move
    SparseMatrixManager(SparseMatrixManager&& other) noexcept;
    SparseMatrixManager& operator=(SparseMatrixManager&& other) noexcept;

    //=========================================================================
    // Ybus Matrix Operations
    //=========================================================================
    
    /**
     * @brief Build Ybus admittance matrix from branch data
     * 
     * Constructs the nodal admittance matrix in CSR format.
     * Y_ij = -y_ij (off-diagonal)
     * Y_ii = sum(y_ij) + y_shunt_i (diagonal)
     * 
     * @param buses Device bus data
     * @param branches Device branch data (with computed admittances)
     * @param ybus Output Ybus matrix structure
     * @return cudaSuccess on success
     */
    [[nodiscard]] cudaError_t buildYbus(
        const DeviceBusData& buses,
        const DeviceBranchData& branches,
        DeviceYbusMatrix& ybus);
    
    /**
     * @brief Update Ybus for topology change (FR-05)
     * 
     * Performs partial update of Ybus when switch status changes.
     * Only modifies affected rows/columns without full rebuild.
     * 
     * @param branches Device branch data
     * @param changed_branches Array of branch indices that changed
     * @param num_changed Number of changed branches
     * @param ybus Ybus matrix to update in-place
     * @return cudaSuccess on success
     */
    [[nodiscard]] cudaError_t updateYbusTopology(
        const DeviceBranchData& branches,
        const int32_t* changed_branches,
        int32_t num_changed,
        DeviceYbusMatrix& ybus);
    
    /**
     * @brief Invalidate Ybus (marks for rebuild)
     */
    void invalidateYbus(DeviceYbusMatrix& ybus) { ybus.is_valid = false; }

    //=========================================================================
    // Jacobian Matrix Operations
    //=========================================================================
    
    /**
     * @brief Analyze Jacobian sparsity pattern
     * 
     * Called once during initialization or after topology change.
     * Determines non-zero structure based on measurement-bus connectivity.
     * 
     * @param measurements Measurement data with associations
     * @param buses Bus data for state dimension
     * @param branches Branch data for connectivity
     * @param H Output Jacobian matrix (pattern only, values zeroed)
     * @return cudaSuccess on success
     */
    [[nodiscard]] cudaError_t analyzeJacobianPattern(
        const DeviceMeasurementData& measurements,
        const DeviceBusData& buses,
        const DeviceBranchData& branches,
        DeviceCSRMatrix& H);
    
    /**
     * @brief Compute Jacobian matrix values (FR-08)
     * 
     * Parallel computation of H(x) elements based on current state.
     * Reuses existing sparsity pattern from analyzeJacobianPattern.
     * 
     * @param measurements Measurement associations
     * @param buses Current bus state (V, theta)
     * @param branches Branch parameters
     * @param ybus Ybus matrix for admittance lookups
     * @param H Jacobian matrix (values updated in-place)
     * @return cudaSuccess on success
     */
    [[nodiscard]] cudaError_t computeJacobianValues(
        const DeviceMeasurementData& measurements,
        const DeviceBusData& buses,
        const DeviceBranchData& branches,
        const DeviceYbusMatrix& ybus,
        DeviceCSRMatrix& H);

    //=========================================================================
    // Gain Matrix Operations (FR-08)
    //=========================================================================
    
    /**
     * @brief Compute Gain matrix G = H^T W H using SpGEMM
     * 
     * Uses cuSPARSE SpGEMM for efficient sparse-sparse multiplication.
     * W is a diagonal weight matrix applied during computation.
     * 
     * @param H Jacobian matrix
     * @param weights Measurement weights (diagonal of W)
     * @param num_weights Number of weights
     * @param G Output Gain matrix
     * @return cudaSuccess on success
     */
    [[nodiscard]] cudaError_t computeGainMatrix(
        const DeviceCSRMatrix& H,
        const Real* weights,
        int32_t num_weights,
        DeviceCSRMatrix& G);
    
    /**
     * @brief Apply matrix reordering for reduced fill-in (Section 5.1)
     * 
     * Uses AMD (Approximate Minimum Degree) ordering to minimize
     * fill-in during Cholesky factorization.
     * 
     * @param G Gain matrix
     * @param perm Output permutation vector
     * @param perm_inv Output inverse permutation vector
     * @return cudaSuccess on success
     */
    [[nodiscard]] cudaError_t computeReordering(
        const DeviceCSRMatrix& G,
        int32_t* perm,
        int32_t* perm_inv);

    //=========================================================================
    // Cholesky Factorization and Solve (FR-08)
    //=========================================================================
    
    /**
     * @brief Perform Cholesky factorization G = L L^T
     * 
     * Uses cuSOLVER for sparse Cholesky decomposition.
     * Factor is stored for subsequent solves.
     * 
     * @param G Gain matrix (must be SPD)
     * @return cudaSuccess on success, error if singular
     */
    [[nodiscard]] cudaError_t factorizeCholesky(const DeviceCSRMatrix& G);
    
    /**
     * @brief Solve system using factored matrix (FR-08)
     * 
     * Performs forward substitution: Ly = b
     * Followed by backward substitution: L^T x = y
     * 
     * @param b Right-hand side vector
     * @param x Output solution vector
     * @param n Vector dimension
     * @return cudaSuccess on success
     */
    [[nodiscard]] cudaError_t solveCholesky(
        const Real* b,
        Real* x,
        int32_t n);
    
    /**
     * @brief Check if Cholesky factor is current
     */
    [[nodiscard]] bool hasValidFactor() const { return factor_valid_; }
    
    /**
     * @brief Invalidate Cholesky factor (triggers refactorization)
     */
    void invalidateFactor() { factor_valid_ = false; }

    //=========================================================================
    // Utility Operations
    //=========================================================================
    
    /**
     * @brief Sparse matrix-vector multiply: y = alpha * A * x + beta * y
     */
    [[nodiscard]] cudaError_t spmv(
        const DeviceCSRMatrix& A,
        const Real* x,
        Real* y,
        Real alpha = 1.0f,
        Real beta = 0.0f);
    
    /**
     * @brief Allocate CSR matrix on device
     */
    [[nodiscard]] cudaError_t allocateCSR(
        DeviceCSRMatrix& mat,
        int32_t rows,
        int32_t cols,
        int32_t nnz);
    
    /**
     * @brief Free CSR matrix memory
     */
    void freeCSR(DeviceCSRMatrix& mat);
    
    /**
     * @brief Allocate Ybus matrix on device
     */
    [[nodiscard]] cudaError_t allocateYbus(
        DeviceYbusMatrix& ybus,
        int32_t n_buses,
        int32_t nnz);
    
    /**
     * @brief Free Ybus matrix memory
     */
    void freeYbus(DeviceYbusMatrix& ybus);

    /**
     * @brief Get library handles for external use
     */
    cusparseHandle_t getCusparseHandle() const { return cusparse_handle_; }
    cusolverSpHandle_t getCusolverHandle() const { return cusolver_handle_; }

private:
    // Library handles
    cusparseHandle_t cusparse_handle_;
    cusolverSpHandle_t cusolver_handle_;
    cudaStream_t stream_;
    
    // Cholesky factorization workspace
    void* chol_workspace_;
    size_t chol_workspace_size_;
    bool factor_valid_;
    
    // Cached matrix data for high-level Cholesky API (CUDA 12.x compatible)
    // Note: CUDA 12.x removed low-level csrcholInfo_t API
    int32_t cached_chol_n_;          ///< Cached matrix dimension
    int32_t cached_chol_nnz_;        ///< Cached number of non-zeros
    Real* cached_chol_values_;        ///< Cached matrix values for solve
    int32_t* cached_chol_row_ptr_;    ///< Cached row pointers
    int32_t* cached_chol_col_ind_;    ///< Cached column indices
    cusparseMatDescr_t chol_mat_descr_; ///< Matrix descriptor for Cholesky
    
    // Temporary buffers for SpGEMM
    void* spgemm_buffer_;
    size_t spgemm_buffer_size_;
    
    // Reordering data
    int32_t* d_perm_;
    int32_t* d_perm_inv_;
    int32_t reorder_size_;
    
    // Helper methods
    [[nodiscard]] cudaError_t ensureCholeskyWorkspace(size_t required);
    [[nodiscard]] cudaError_t ensureSpGEMMBuffer(size_t required);
};

//=============================================================================
// SECTION 2: CUDA Kernels for Matrix Operations
//=============================================================================

/**
 * @brief Kernel: Compute branch admittance values
 * 
 * Calculates series and shunt admittance from R, X, B parameters.
 * y_series = 1/(R + jX) = G + jB
 * 
 * @param branches Branch data with R, X, B values
 * @param n Number of branches
 */
__global__ void computeBranchAdmittanceKernel(
    Real* __restrict__ g_series,
    Real* __restrict__ b_series,
    Real* __restrict__ b_shunt_from,
    Real* __restrict__ b_shunt_to,
    const Real* __restrict__ resistance,
    const Real* __restrict__ reactance,
    const Real* __restrict__ susceptance,
    const Real* __restrict__ tap_ratio,
    int32_t n);

/**
 * @brief Kernel: Count non-zeros per row for Ybus
 */
__global__ void countYbusNonzerosKernel(
    int32_t* __restrict__ row_counts,
    const int32_t* __restrict__ from_bus,
    const int32_t* __restrict__ to_bus,
    const SwitchStatus* __restrict__ status,
    int32_t n_branches,
    int32_t n_buses);

/**
 * @brief Kernel: Fill Ybus values
 * 
 * Populates the Ybus matrix entries from branch admittances.
 * Handles transformer tap ratios and phase shifts.
 */
__global__ void fillYbusKernel(
    Real* __restrict__ g_values,
    Real* __restrict__ b_values,
    int32_t* __restrict__ col_ind,
    const int32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ from_bus,
    const int32_t* __restrict__ to_bus,
    const Real* __restrict__ g_series,
    const Real* __restrict__ b_series,
    const Real* __restrict__ b_shunt_from,
    const Real* __restrict__ b_shunt_to,
    const Real* __restrict__ tap_ratio,
    const Real* __restrict__ phase_shift,
    const SwitchStatus* __restrict__ status,
    int32_t n_branches,
    int32_t n_buses);

// Note: Jacobian computation kernels are now in cuda_optimizations.cuh (opt namespace)
// with shared memory optimizations:
//   - opt::jacobianVmagOptimizedKernel
//   - opt::jacobianPinjOptimizedKernel

//=============================================================================
// SECTION 3: Host-Side Helper Functions
//=============================================================================

/**
 * @brief Compute number of non-zeros in Jacobian based on measurement types
 * 
 * Different measurement types have different numbers of non-zeros:
 * - V_mag: 1 (only dh/dV_i)
 * - P_inj, Q_inj: 2*n_connected (derivatives w.r.t. all connected buses)
 * - P_flow, Q_flow: 4 (derivatives w.r.t. from/to V and theta)
 * 
 * @param meas Measurement data
 * @param ybus Ybus for connectivity information
 * @return Estimated number of non-zeros
 */
int32_t estimateJacobianNNZ(
    const DeviceMeasurementData& meas,
    const DeviceYbusMatrix& ybus);

/**
 * @brief Create cuSPARSE matrix descriptor
 */
[[nodiscard]] cusparseStatus_t createMatrixDescriptor(
    cusparseSpMatDescr_t* descr,
    const DeviceCSRMatrix& mat);

/**
 * @brief Destroy cuSPARSE matrix descriptor
 */
void destroyMatrixDescriptor(cusparseSpMatDescr_t descr);

} // namespace sle

#endif // SPARSE_MATRIX_CUH

