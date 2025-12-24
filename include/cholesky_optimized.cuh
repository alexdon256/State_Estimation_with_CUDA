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
 * @file cholesky_optimized.cuh
 * @brief Optimized Cholesky Factorization for State Estimation
 * 
 * This module provides highly optimized sparse Cholesky factorization
 * specifically tuned for the power system state estimation problem.
 * 
 * Key optimizations (Section 5.1):
 * - Symbolic analysis caching for repeated factorizations
 * - AMD (Approximate Minimum Degree) reordering for fill-in reduction
 * - Nested dissection for large networks
 * - Workspace reuse to avoid repeated allocations
 * - Supernodal factorization for dense subblocks
 * 
 * The Gain matrix G = H^T W H in state estimation typically has:
 * - Symmetric positive definite structure
 * - Sparsity pattern determined by network topology
 * - Pattern that remains constant when topology is unchanged (FR-16)
 * 
 * This allows significant optimizations by caching symbolic analysis.
 */

#ifndef CHOLESKY_OPTIMIZED_CUH
#define CHOLESKY_OPTIMIZED_CUH

#include "sle_types.cuh"
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <memory>
#include <vector>

namespace sle {

//=============================================================================
// SECTION 1: Configuration and Constants
//=============================================================================

/**
 * @brief Reordering algorithm selection
 */
enum class ReorderingAlgorithm {
    NONE,           ///< No reordering (use original ordering)
    AMD,            ///< Approximate Minimum Degree
    RCM,            ///< Reverse Cuthill-McKee
    METIS,          ///< Graph partitioning (external library)
    NESTED_DISSECTION  ///< Nested dissection for large problems
};

/**
 * @brief Configuration for Cholesky factorization
 */
struct CholeskyConfig {
    ReorderingAlgorithm reordering = ReorderingAlgorithm::AMD;
    
    /// Minimum matrix size to apply reordering
    int32_t min_size_for_reorder = 100;
    
    /// Use supernodal factorization for dense blocks
    bool use_supernodal = true;
    
    /// Minimum supernode size
    int32_t min_supernode_size = 16;
    
    /// Cache symbolic analysis results
    bool cache_symbolic = true;
    
    /// Tolerance for detecting singularity
    Real singularity_tol = 1e-12f;
    
    /// Use mixed precision (fp32 factorization, fp64 refinement)
    bool mixed_precision = false;
};

//=============================================================================
// SECTION 2: Symbolic Analysis Cache
//=============================================================================

/**
 * @brief Cached symbolic analysis results
 * 
 * Stores the sparsity pattern of L factor and permutation vectors.
 * Can be reused as long as the matrix structure doesn't change.
 */
struct SymbolicAnalysisCache {
    /// Original matrix dimension
    int32_t n;
    
    /// Number of non-zeros in original matrix
    int32_t nnz_original;
    
    /// Estimated non-zeros in L factor (including fill-in)
    int64_t nnz_factor;
    
    /// Row pointers for L factor pattern
    std::vector<int32_t> L_row_ptr;
    
    /// Column indices for L factor pattern
    std::vector<int32_t> L_col_ind;
    
    /// Permutation vector (original -> reordered)
    std::vector<int32_t> perm;
    
    /// Inverse permutation (reordered -> original)
    std::vector<int32_t> perm_inv;
    
    /// Elimination tree for parallel factorization
    std::vector<int32_t> etree;
    
    /// Level structure for parallel scheduling
    std::vector<int32_t> level_ptr;
    std::vector<int32_t> level_set;
    
    /// Supernode structure
    std::vector<int32_t> supernode_ptr;
    int32_t n_supernodes;
    
    /// Flag indicating cache is valid
    bool is_valid = false;
    
    /// Hash of the sparsity pattern for validation
    uint64_t pattern_hash = 0;
    
    /**
     * @brief Compute hash of matrix sparsity pattern
     */
    static uint64_t computePatternHash(const int32_t* row_ptr, 
                                       const int32_t* col_ind, 
                                       int32_t n);
    
    /**
     * @brief Check if cache matches current matrix pattern
     */
    [[nodiscard]] bool matchesPattern(const int32_t* row_ptr,
                                      const int32_t* col_ind,
                                      int32_t n) const;
};

//=============================================================================
// SECTION 3: Optimized Cholesky Solver Class
//=============================================================================

/**
 * @brief High-performance Cholesky solver for state estimation
 * 
 * This class wraps cuSOLVER's sparse Cholesky with additional
 * optimizations specifically for the WLS state estimation problem.
 */
class OptimizedCholeskySolver {
public:
    /**
     * @brief Constructor
     * @param stream CUDA stream for all operations
     * @param config Solver configuration
     */
    explicit OptimizedCholeskySolver(cudaStream_t stream = nullptr,
                                     const CholeskyConfig& config = CholeskyConfig());
    
    /**
     * @brief Destructor - releases all GPU resources
     */
    ~OptimizedCholeskySolver();
    
    // Disable copy
    OptimizedCholeskySolver(const OptimizedCholeskySolver&) = delete;
    OptimizedCholeskySolver& operator=(const OptimizedCholeskySolver&) = delete;
    
    // Enable move
    OptimizedCholeskySolver(OptimizedCholeskySolver&& other) noexcept;
    OptimizedCholeskySolver& operator=(OptimizedCholeskySolver&& other) noexcept;

    //=========================================================================
    // Main Interface
    //=========================================================================
    
    /**
     * @brief Perform symbolic analysis of matrix structure
     * 
     * This should be called once when the matrix structure changes.
     * Results are cached for reuse during numerical factorization.
     * 
     * @param n Matrix dimension
     * @param nnz Number of non-zeros
     * @param d_row_ptr Device row pointers
     * @param d_col_ind Device column indices
     * @return cudaSuccess on success
     */
    [[nodiscard]] cudaError_t analyzePattern(
        int32_t n,
        int32_t nnz,
        const int32_t* d_row_ptr,
        const int32_t* d_col_ind);
    
    /**
     * @brief Perform numerical factorization
     * 
     * Computes L such that P * A * P^T = L * L^T
     * Uses cached symbolic analysis if available.
     * 
     * @param d_values Device array of matrix values
     * @return cudaSuccess on success, error if singular
     */
    [[nodiscard]] cudaError_t factorize(const Real* d_values);
    
    /**
     * @brief Solve system using computed factorization
     * 
     * Solves A * x = b using: x = P^T * L^(-T) * L^(-1) * P * b
     * 
     * @param d_b Device RHS vector (input)
     * @param d_x Device solution vector (output)
     * @return cudaSuccess on success
     */
    [[nodiscard]] cudaError_t solve(const Real* d_b, Real* d_x);
    
    /**
     * @brief Combined factorize and solve
     * 
     * More efficient than separate calls when solving single system.
     * 
     * @param d_values Matrix values
     * @param d_b RHS vector
     * @param d_x Solution vector
     * @return cudaSuccess on success
     */
    [[nodiscard]] cudaError_t factorizeAndSolve(
        const Real* d_values,
        const Real* d_b,
        Real* d_x);
    
    /**
     * @brief Solve multiple RHS vectors
     * 
     * Efficiently solves A * X = B for multiple RHS vectors.
     * 
     * @param d_B Device RHS matrix (n x nrhs, column-major)
     * @param d_X Device solution matrix (output)
     * @param nrhs Number of RHS vectors
     * @return cudaSuccess on success
     */
    [[nodiscard]] cudaError_t solveMultiple(
        const Real* d_B,
        Real* d_X,
        int32_t nrhs);

    //=========================================================================
    // State Management
    //=========================================================================
    
    /**
     * @brief Check if symbolic analysis is cached
     */
    [[nodiscard]] bool hasSymbolicAnalysis() const { return cache_.is_valid; }
    
    /**
     * @brief Check if numerical factorization is current
     */
    [[nodiscard]] bool hasNumericalFactor() const { return factor_valid_; }
    
    /**
     * @brief Invalidate symbolic cache (call after structure change)
     */
    void invalidateSymbolic() { cache_.is_valid = false; }
    
    /**
     * @brief Invalidate numerical factor (call after value change)
     */
    void invalidateNumerical() { factor_valid_ = false; }
    
    /**
     * @brief Get cuSPARSE handle for external use
     */
    [[nodiscard]] cusparseHandle_t getCusparseHandle() const { return cusparse_handle_; }
    
    /**
     * @brief Get estimated fill-in from symbolic analysis
     */
    [[nodiscard]] int64_t getFactorNNZ() const { return cache_.nnz_factor; }
    
    /**
     * @brief Get number of supernodes
     */
    [[nodiscard]] int32_t getNumSupernodes() const { return cache_.n_supernodes; }

    //=========================================================================
    // Configuration
    //=========================================================================
    
    /**
     * @brief Update configuration
     * 
     * Note: Changing reordering algorithm invalidates cached analysis.
     */
    void setConfig(const CholeskyConfig& config);
    
    /**
     * @brief Get current configuration
     */
    [[nodiscard]] const CholeskyConfig& getConfig() const { return config_; }
    
    /**
     * @brief Set CUDA stream
     */
    void setStream(cudaStream_t stream);

private:
    //=========================================================================
    // Internal Methods
    //=========================================================================
    
    /**
     * @brief Compute fill-reducing ordering
     */
    [[nodiscard]] cudaError_t computeReordering(
        int32_t n,
        const int32_t* d_row_ptr,
        const int32_t* d_col_ind);
    
    /**
     * @brief Build elimination tree
     */
    void buildEliminationTree();
    
    /**
     * @brief Identify supernodes for blocked factorization
     */
    void identifySupernodes();
    
    /**
     * @brief Compute level structure for parallel scheduling
     */
    void computeLevelStructure();
    
    /**
     * @brief Allocate workspace for factorization
     */
    [[nodiscard]] cudaError_t allocateWorkspace(size_t required);
    
    /**
     * @brief Apply permutation to vector
     */
    [[nodiscard]] cudaError_t applyPermutation(
        const Real* d_in,
        Real* d_out,
        const int32_t* d_perm,
        int32_t n);
    
    /**
     * @brief Apply inverse permutation to vector
     */
    [[nodiscard]] cudaError_t applyInversePermutation(
        const Real* d_in,
        Real* d_out,
        const int32_t* d_perm_inv,
        int32_t n);

    //=========================================================================
    // Member Variables
    //=========================================================================
    
    // CUDA handles
    cudaStream_t stream_;
    cusolverSpHandle_t cusolver_handle_;
    cusparseHandle_t cusparse_handle_;
    
    // Configuration
    CholeskyConfig config_;
    
    // Symbolic analysis cache
    SymbolicAnalysisCache cache_;
    
    // Cached matrix data for high-level Cholesky API (CUDA 12.x compatible)
    // Note: CUDA 12.x removed low-level csrcholInfo_t API
    Real* cached_values_;
    int32_t* cached_row_ptr_;
    int32_t* cached_col_ind_;
    
    // Matrix descriptor
    cusparseMatDescr_t mat_descr_;
    
    // Workspace buffer
    void* d_workspace_;
    size_t workspace_size_;
    
    // Device permutation vectors
    int32_t* d_perm_;
    int32_t* d_perm_inv_;
    
    // Device temporary vectors for permuted data
    Real* d_temp_vec_;
    Real* d_permuted_values_;
    
    // Reordered matrix (stored for reuse)
    int32_t* d_reordered_row_ptr_;
    int32_t* d_reordered_col_ind_;
    Real* d_reordered_values_;
    
    // Current matrix info
    int32_t current_n_;
    int32_t current_nnz_;
    
    // State flags
    bool factor_valid_;
    bool symbolic_on_device_;
};

//=============================================================================
// SECTION 4: CUDA Kernels for Cholesky Operations
//=============================================================================

/**
 * @brief Apply permutation to CSR matrix values
 * 
 * Reorders matrix values according to computed permutation.
 * Uses shared memory for efficient gather operations.
 */
__global__ void applyMatrixPermutationKernel(
    Real* __restrict__ reordered_values,
    int32_t* __restrict__ reordered_col_ind,
    const Real* __restrict__ original_values,
    const int32_t* __restrict__ original_col_ind,
    const int32_t* __restrict__ original_row_ptr,
    const int32_t* __restrict__ reordered_row_ptr,
    const int32_t* __restrict__ perm,
    const int32_t* __restrict__ perm_inv,
    int32_t n);

/**
 * @brief Apply permutation to dense vector
 */
__global__ void applyVectorPermutationKernel(
    Real* __restrict__ y,
    const Real* __restrict__ x,
    const int32_t* __restrict__ perm,
    int32_t n);

/**
 * @brief Apply inverse permutation to dense vector
 */
__global__ void applyInverseVectorPermutationKernel(
    Real* __restrict__ y,
    const Real* __restrict__ x,
    const int32_t* __restrict__ perm_inv,
    int32_t n);

/**
 * @brief Diagonal scaling for improved conditioning
 * 
 * Computes D such that D * A * D has unit diagonal.
 */
__global__ void computeDiagonalScalingKernel(
    Real* __restrict__ d_diag,
    const int32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_ind,
    const Real* __restrict__ values,
    int32_t n);

/**
 * @brief Apply diagonal scaling to matrix
 */
__global__ void applyDiagonalScalingKernel(
    Real* __restrict__ scaled_values,
    const Real* __restrict__ values,
    const int32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_ind,
    const Real* __restrict__ d_left,
    const Real* __restrict__ d_right,
    int32_t n);

//=============================================================================
// SECTION 5: Iterative Refinement
//=============================================================================

/**
 * @brief Iterative refinement configuration
 */
struct RefinementConfig {
    /// Maximum refinement iterations
    int32_t max_iterations = 2;
    
    /// Convergence tolerance
    Real tolerance = 1e-10f;
    
    /// Use mixed precision refinement
    bool mixed_precision = false;
};

/**
 * @brief Apply iterative refinement to improve solution accuracy
 * 
 * Uses the factorization to refine the solution:
 * r = b - A*x
 * solve: A*e = r
 * x = x + e
 * 
 * @param solver The Cholesky solver with computed factorization
 * @param d_A_row_ptr Original matrix row pointers
 * @param d_A_col_ind Original matrix column indices
 * @param d_A_values Original matrix values
 * @param d_b RHS vector
 * @param d_x Solution vector (input/output)
 * @param n System dimension
 * @param config Refinement configuration
 * @param stream CUDA stream
 * @return Number of refinement iterations performed
 */
[[nodiscard]] int32_t applyIterativeRefinement(
    OptimizedCholeskySolver& solver,
    const int32_t* d_A_row_ptr,
    const int32_t* d_A_col_ind,
    const Real* d_A_values,
    const Real* d_b,
    Real* d_x,
    int32_t n,
    const RefinementConfig& config,
    cudaStream_t stream);

//=============================================================================
// SECTION 6: Utility Functions
//=============================================================================

/**
 * @brief Compute AMD ordering on host
 * 
 * @param n Matrix dimension
 * @param row_ptr CSR row pointers (host)
 * @param col_ind CSR column indices (host)
 * @param perm Output permutation vector (host)
 * @param perm_inv Output inverse permutation (host)
 */
void computeAMDOrdering(
    int32_t n,
    const int32_t* row_ptr,
    const int32_t* col_ind,
    int32_t* perm,
    int32_t* perm_inv);

/**
 * @brief Compute RCM ordering on host
 */
void computeRCMOrdering(
    int32_t n,
    const int32_t* row_ptr,
    const int32_t* col_ind,
    int32_t* perm,
    int32_t* perm_inv);

/**
 * @brief Estimate fill-in after factorization
 * 
 * Uses symbolic factorization to count non-zeros in L factor.
 * 
 * @param n Matrix dimension
 * @param row_ptr CSR row pointers
 * @param col_ind CSR column indices
 * @param perm Permutation (can be identity)
 * @return Estimated number of non-zeros in L
 */
[[nodiscard]] int64_t estimateFactorFillIn(
    int32_t n,
    const int32_t* row_ptr,
    const int32_t* col_ind,
    const int32_t* perm);

/**
 * @brief Build elimination tree
 * 
 * @param n Matrix dimension
 * @param row_ptr CSR row pointers
 * @param col_ind CSR column indices
 * @param perm Permutation
 * @param parent Output parent array for elimination tree
 */
void buildEliminationTree(
    int32_t n,
    const int32_t* row_ptr,
    const int32_t* col_ind,
    const int32_t* perm,
    int32_t* parent);

} // namespace sle

#endif // CHOLESKY_OPTIMIZED_CUH

