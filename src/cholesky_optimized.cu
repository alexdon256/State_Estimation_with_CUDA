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
 * @file cholesky_optimized.cu
 * @brief Implementation of optimized Cholesky factorization
 * 
 * High-performance sparse Cholesky with:
 * - Symbolic analysis caching
 * - Fill-reducing reordering
 * - Efficient workspace management
 */

#include "../include/cholesky_optimized.cuh"
#include "../include/cuda_optimizations.cuh"
#include "../include/jsf_compliance.h"
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <algorithm>
#include <numeric>
#include <queue>
#include <cstring>
#include <vector>
#include <omp.h>

namespace sle {

//=============================================================================
// SymbolicAnalysisCache Implementation
//=============================================================================

uint64_t SymbolicAnalysisCache::computePatternHash(
    const int32_t* row_ptr,
    const int32_t* col_ind,
    int32_t n)
{
    // Simple hash combining row pointer pattern
    uint64_t hash = static_cast<uint64_t>(n) * 2654435761ULL;
    
    // Hash row pointers
    for (int32_t i = 0; i <= n; ++i) {
        hash ^= (static_cast<uint64_t>(row_ptr[i]) * 2654435761ULL) >> (i % 32);
    }
    
    // Hash column indices (sample for efficiency)
    int32_t nnz = row_ptr[n];
    int32_t step = (nnz > 1000) ? nnz / 1000 : 1;
    for (int32_t i = 0; i < nnz; i += step) {
        hash ^= (static_cast<uint64_t>(col_ind[i]) * 2654435761ULL) << (i % 32);
    }
    
    return hash;
}

bool SymbolicAnalysisCache::matchesPattern(
    const int32_t* row_ptr,
    const int32_t* col_ind,
    int32_t n) const
{
    if (!is_valid || n != this->n) return false;
    if (row_ptr[n] != nnz_original) return false;
    
    return computePatternHash(row_ptr, col_ind, n) == pattern_hash;
}

//=============================================================================
// OptimizedCholeskySolver Implementation
//=============================================================================

OptimizedCholeskySolver::OptimizedCholeskySolver(
    cudaStream_t stream,
    const CholeskyConfig& config)
    : stream_(stream)
    , cusolver_handle_(nullptr)
    , cusparse_handle_(nullptr)
    , config_(config)
    , cached_values_(nullptr)
    , cached_row_ptr_(nullptr)
    , cached_col_ind_(nullptr)
    , mat_descr_(nullptr)
    , d_workspace_(nullptr)
    , workspace_size_(0)
    , d_perm_(nullptr)
    , d_perm_inv_(nullptr)
    , d_temp_vec_(nullptr)
    , d_permuted_values_(nullptr)
    , d_reordered_row_ptr_(nullptr)
    , d_reordered_col_ind_(nullptr)
    , d_reordered_values_(nullptr)
    , current_n_(0)
    , current_nnz_(0)
    , factor_valid_(false)
    , symbolic_on_device_(false)
{
    // Initialize cuSOLVER handle
    cusolverSpCreate(&cusolver_handle_);
    if (stream_) {
        cusolverSpSetStream(cusolver_handle_, stream_);
    }
    
    // Initialize cuSPARSE handle
    cusparseCreate(&cusparse_handle_);
    if (stream_) {
        cusparseSetStream(cusparse_handle_, stream_);
    }
    
    // Initialize cached matrix pointers (CUDA 12.x compatible - no csrcholInfo_t)
    cached_values_ = nullptr;
    cached_row_ptr_ = nullptr;
    cached_col_ind_ = nullptr;
    
    // Create matrix descriptor
    cusparseCreateMatDescr(&mat_descr_);
    cusparseSetMatType(mat_descr_, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
    cusparseSetMatFillMode(mat_descr_, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatIndexBase(mat_descr_, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatDiagType(mat_descr_, CUSPARSE_DIAG_TYPE_NON_UNIT);
}

OptimizedCholeskySolver::~OptimizedCholeskySolver() {
    // Free device memory
    if (d_workspace_) cudaFree(d_workspace_);
    if (d_perm_) cudaFree(d_perm_);
    if (d_perm_inv_) cudaFree(d_perm_inv_);
    if (d_temp_vec_) cudaFree(d_temp_vec_);
    if (d_permuted_values_) cudaFree(d_permuted_values_);
    if (d_reordered_row_ptr_) cudaFree(d_reordered_row_ptr_);
    if (d_reordered_col_ind_) cudaFree(d_reordered_col_ind_);
    if (d_reordered_values_) cudaFree(d_reordered_values_);
    
    // Free cached matrix data (CUDA 12.x compatible)
    if (cached_values_) cudaFree(cached_values_);
    if (cached_row_ptr_) cudaFree(cached_row_ptr_);
    if (cached_col_ind_) cudaFree(cached_col_ind_);
    
    // Destroy structures
    if (mat_descr_) cusparseDestroyMatDescr(mat_descr_);
    
    // Destroy handles
    if (cusolver_handle_) cusolverSpDestroy(cusolver_handle_);
    if (cusparse_handle_) cusparseDestroy(cusparse_handle_);
}

OptimizedCholeskySolver::OptimizedCholeskySolver(
    OptimizedCholeskySolver&& other) noexcept
    : stream_(other.stream_)
    , cusolver_handle_(other.cusolver_handle_)
    , cusparse_handle_(other.cusparse_handle_)
    , config_(other.config_)
    , cache_(std::move(other.cache_))
    , cached_values_(other.cached_values_)
    , cached_row_ptr_(other.cached_row_ptr_)
    , cached_col_ind_(other.cached_col_ind_)
    , mat_descr_(other.mat_descr_)
    , d_workspace_(other.d_workspace_)
    , workspace_size_(other.workspace_size_)
    , d_perm_(other.d_perm_)
    , d_perm_inv_(other.d_perm_inv_)
    , d_temp_vec_(other.d_temp_vec_)
    , d_permuted_values_(other.d_permuted_values_)
    , d_reordered_row_ptr_(other.d_reordered_row_ptr_)
    , d_reordered_col_ind_(other.d_reordered_col_ind_)
    , d_reordered_values_(other.d_reordered_values_)
    , current_n_(other.current_n_)
    , current_nnz_(other.current_nnz_)
    , factor_valid_(other.factor_valid_)
    , symbolic_on_device_(other.symbolic_on_device_)
{
    other.cusolver_handle_ = nullptr;
    other.cusparse_handle_ = nullptr;
    other.cached_values_ = nullptr;
    other.cached_row_ptr_ = nullptr;
    other.cached_col_ind_ = nullptr;
    other.mat_descr_ = nullptr;
    other.d_workspace_ = nullptr;
    other.d_perm_ = nullptr;
    other.d_perm_inv_ = nullptr;
    other.d_temp_vec_ = nullptr;
    other.d_permuted_values_ = nullptr;
    other.d_reordered_row_ptr_ = nullptr;
    other.d_reordered_col_ind_ = nullptr;
    other.d_reordered_values_ = nullptr;
}

OptimizedCholeskySolver& OptimizedCholeskySolver::operator=(
    OptimizedCholeskySolver&& other) noexcept
{
    if (this != &other) {
        // Release current resources
        if (d_workspace_) cudaFree(d_workspace_);
        if (d_perm_) cudaFree(d_perm_);
        if (d_perm_inv_) cudaFree(d_perm_inv_);
        if (d_temp_vec_) cudaFree(d_temp_vec_);
        if (d_permuted_values_) cudaFree(d_permuted_values_);
        if (d_reordered_row_ptr_) cudaFree(d_reordered_row_ptr_);
        if (d_reordered_col_ind_) cudaFree(d_reordered_col_ind_);
        if (d_reordered_values_) cudaFree(d_reordered_values_);
        if (cached_values_) cudaFree(cached_values_);
        if (cached_row_ptr_) cudaFree(cached_row_ptr_);
        if (cached_col_ind_) cudaFree(cached_col_ind_);
        if (mat_descr_) cusparseDestroyMatDescr(mat_descr_);
        if (cusolver_handle_) cusolverSpDestroy(cusolver_handle_);
        if (cusparse_handle_) cusparseDestroy(cusparse_handle_);
        
        // Move from other
        stream_ = other.stream_;
        cusolver_handle_ = other.cusolver_handle_;
        cusparse_handle_ = other.cusparse_handle_;
        config_ = other.config_;
        cache_ = std::move(other.cache_);
        cached_values_ = other.cached_values_;
        cached_row_ptr_ = other.cached_row_ptr_;
        cached_col_ind_ = other.cached_col_ind_;
        mat_descr_ = other.mat_descr_;
        d_workspace_ = other.d_workspace_;
        workspace_size_ = other.workspace_size_;
        d_perm_ = other.d_perm_;
        d_perm_inv_ = other.d_perm_inv_;
        d_temp_vec_ = other.d_temp_vec_;
        d_permuted_values_ = other.d_permuted_values_;
        d_reordered_row_ptr_ = other.d_reordered_row_ptr_;
        d_reordered_col_ind_ = other.d_reordered_col_ind_;
        d_reordered_values_ = other.d_reordered_values_;
        current_n_ = other.current_n_;
        current_nnz_ = other.current_nnz_;
        factor_valid_ = other.factor_valid_;
        symbolic_on_device_ = other.symbolic_on_device_;
        
        // Null out other
        other.cusolver_handle_ = nullptr;
        other.cusparse_handle_ = nullptr;
        other.cached_values_ = nullptr;
        other.cached_row_ptr_ = nullptr;
        other.cached_col_ind_ = nullptr;
        other.mat_descr_ = nullptr;
        other.d_workspace_ = nullptr;
        other.d_perm_ = nullptr;
        other.d_perm_inv_ = nullptr;
        other.d_temp_vec_ = nullptr;
        other.d_permuted_values_ = nullptr;
        other.d_reordered_row_ptr_ = nullptr;
        other.d_reordered_col_ind_ = nullptr;
        other.d_reordered_values_ = nullptr;
    }
    return *this;
}

//=============================================================================
// Pattern Analysis
//=============================================================================

cudaError_t OptimizedCholeskySolver::analyzePattern(
    int32_t n_param,
    int32_t nnz,
    const int32_t* d_row_ptr,
    const int32_t* d_col_ind)
{
    // Check if we can reuse existing analysis
    if (config_.cache_symbolic && cache_.is_valid && 
        cache_.n == n_param && cache_.nnz_original == nnz) {
        // Quick hash check
        std::vector<int32_t> h_row_ptr(n_param + 1);
        std::vector<int32_t> h_col_ind(nnz);
        
        cudaMemcpy(h_row_ptr.data(), d_row_ptr, (n_param + 1) * sizeof(int32_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_col_ind.data(), d_col_ind, nnz * sizeof(int32_t),
                   cudaMemcpyDeviceToHost);
        
        if (cache_.matchesPattern(h_row_ptr.data(), h_col_ind.data(), n_param)) {
            // Pattern matches, can reuse symbolic analysis
            return cudaSuccess;
        }
    }
    
    current_n_ = n_param;
    current_nnz_ = nnz;
    
    // Copy matrix structure to host for reordering computation
    std::vector<int32_t> h_row_ptr(n_param + 1);
    std::vector<int32_t> h_col_ind(nnz);
    
    cudaMemcpy(h_row_ptr.data(), d_row_ptr, (n_param + 1) * sizeof(int32_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_col_ind.data(), d_col_ind, nnz * sizeof(int32_t),
               cudaMemcpyDeviceToHost);
    
    // Initialize cache
    cache_.n = n_param;
    cache_.nnz_original = nnz;
    cache_.perm.resize(n_param);
    cache_.perm_inv.resize(n_param);
    
    // Compute fill-reducing ordering
    if (config_.reordering != ReorderingAlgorithm::NONE && 
        n_param >= config_.min_size_for_reorder) {
        
        switch (config_.reordering) {
            case ReorderingAlgorithm::AMD:
                computeAMDOrdering(n_param, h_row_ptr.data(), h_col_ind.data(),
                                  cache_.perm.data(), cache_.perm_inv.data());
                break;
                
            case ReorderingAlgorithm::RCM:
                computeRCMOrdering(n_param, h_row_ptr.data(), h_col_ind.data(),
                                  cache_.perm.data(), cache_.perm_inv.data());
                break;
                
            default:
                // Identity permutation
                std::iota(cache_.perm.begin(), cache_.perm.end(), 0);
                std::iota(cache_.perm_inv.begin(), cache_.perm_inv.end(), 0);
                break;
        }
    } else {
        // Identity permutation
        std::iota(cache_.perm.begin(), cache_.perm.end(), 0);
        std::iota(cache_.perm_inv.begin(), cache_.perm_inv.end(), 0);
    }
    
    // Estimate fill-in
    cache_.nnz_factor = estimateFactorFillIn(n_param, h_row_ptr.data(), 
                                              h_col_ind.data(),
                                              cache_.perm.data());
    
    // Build elimination tree
    cache_.etree.resize(n_param);
    sle::buildEliminationTree(n_param, h_row_ptr.data(), h_col_ind.data(),
                        cache_.perm.data(), cache_.etree.data());
    
    // Compute pattern hash
    cache_.pattern_hash = SymbolicAnalysisCache::computePatternHash(
        h_row_ptr.data(), h_col_ind.data(), n_param);
    
    // Allocate device permutation vectors
    if (d_perm_) cudaFree(d_perm_);
    if (d_perm_inv_) cudaFree(d_perm_inv_);
    
    cudaMalloc(&d_perm_, n_param * sizeof(int32_t));
    cudaMalloc(&d_perm_inv_, n_param * sizeof(int32_t));
    
    cudaMemcpy(d_perm_, cache_.perm.data(), n_param * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_perm_inv_, cache_.perm_inv.data(), n_param * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    
    // Allocate temporary vectors
    if (d_temp_vec_) cudaFree(d_temp_vec_);
    cudaMalloc(&d_temp_vec_, n_param * sizeof(Real));
    
    // CUDA 12.x compatible: No low-level symbolic analysis API
    // Just mark cache as valid - actual work happens in solve()
    cache_.is_valid = true;
    factor_valid_ = false;
    symbolic_on_device_ = true;
    
    return cudaGetLastError();
}

//=============================================================================
// Numerical Factorization
//=============================================================================

cudaError_t OptimizedCholeskySolver::factorize(const Real* d_values) {
    if (!cache_.is_valid || current_n_ == 0 || current_nnz_ == 0) {
        return cudaErrorNotReady;
    }
    
    // CUDA 12.x compatible: Cache matrix values for use in solve()
    // The high-level API does analysis + factorization + solve together
    
    // Reallocate cached arrays if needed
    bool need_realloc = !cached_values_ || !cached_row_ptr_ || !cached_col_ind_;
    
    if (need_realloc) {
        if (cached_values_) cudaFree(cached_values_);
        if (cached_row_ptr_) cudaFree(cached_row_ptr_);
        if (cached_col_ind_) cudaFree(cached_col_ind_);
        
        cudaError_t err;
        err = cudaMalloc(&cached_values_, current_nnz_ * sizeof(Real));
        if (err != cudaSuccess) return err;
        
        err = cudaMalloc(&cached_row_ptr_, (current_n_ + 1) * sizeof(int32_t));
    if (err != cudaSuccess) return err;
    
        err = cudaMalloc(&cached_col_ind_, current_nnz_ * sizeof(int32_t));
        if (err != cudaSuccess) return err;
    }
    
    // Copy matrix values (values change each iteration)
    cudaMemcpyAsync(cached_values_, d_values, current_nnz_ * sizeof(Real),
                    cudaMemcpyDeviceToDevice, stream_);
    
    // Copy structure if reordering arrays exist, otherwise the structure was
    // already set up during analyzePattern
    if (d_reordered_row_ptr_ && d_reordered_col_ind_) {
        cudaMemcpyAsync(cached_row_ptr_, d_reordered_row_ptr_, (current_n_ + 1) * sizeof(int32_t),
                        cudaMemcpyDeviceToDevice, stream_);
        cudaMemcpyAsync(cached_col_ind_, d_reordered_col_ind_, current_nnz_ * sizeof(int32_t),
                        cudaMemcpyDeviceToDevice, stream_);
    }
    
    cudaStreamSynchronize(stream_);
    
    factor_valid_ = true;
    return cudaSuccess;
}

//=============================================================================
// Solve
//=============================================================================

cudaError_t OptimizedCholeskySolver::solve(const Real* d_b, Real* d_x) {
    if (!factor_valid_) {
        return cudaErrorNotReady;
    }
    
    // Apply permutation to RHS: b_perm = P * b
    (void)applyPermutation(d_b, d_temp_vec_, d_perm_, current_n_);
    
    // CUDA 12.x compatible: Use high-level API that does everything
    int singularity = -1;
    
#ifdef SLE_USE_DOUBLE
    cusolverStatus_t status = cusolverSpDcsrlsvchol(
        cusolver_handle_,
        current_n_,
        current_nnz_,
        mat_descr_,
        cached_values_,
        cached_row_ptr_,
        cached_col_ind_,
        d_temp_vec_,
        config_.singularity_tol,
        0,  // reorder = 0
        d_x,
        &singularity);
#else
    cusolverStatus_t status = cusolverSpScsrlsvchol(
        cusolver_handle_,
        current_n_,
        current_nnz_,
        mat_descr_,
        cached_values_,
        cached_row_ptr_,
        cached_col_ind_,
        d_temp_vec_,
        config_.singularity_tol,
        0,  // reorder = 0
        d_x,
        &singularity);
#endif
    
    if (status != CUSOLVER_STATUS_SUCCESS || singularity >= 0) {
        return cudaErrorUnknown;
    }
    
    // Apply inverse permutation to solution: x = P^T * x_perm
    // Copy x to temp, then permute back
    cudaMemcpyAsync(d_temp_vec_, d_x, current_n_ * sizeof(Real),
                    cudaMemcpyDeviceToDevice, stream_);
    (void)applyInversePermutation(d_temp_vec_, d_x, d_perm_inv_, current_n_);
    
    return cudaGetLastError();
}

cudaError_t OptimizedCholeskySolver::factorizeAndSolve(
    const Real* d_values,
    const Real* d_b,
    Real* d_x)
{
    cudaError_t err = factorize(d_values);
    if (err != cudaSuccess) return err;
    
    return solve(d_b, d_x);
}

cudaError_t OptimizedCholeskySolver::solveMultiple(
    const Real* d_B,
    Real* d_X,
    int32_t nrhs)
{
    if (!factor_valid_) {
        return cudaErrorNotReady;
    }
    
    // For small number of RHS or small systems, sequential is efficient
    // For larger problems, we use batched approach with streams
    constexpr int32_t BATCH_THRESHOLD_NRHS = 4;
    constexpr int32_t BATCH_THRESHOLD_N = 1000;
    
    if (nrhs < BATCH_THRESHOLD_NRHS || current_n_ < BATCH_THRESHOLD_N) {
        // Sequential solve - simple and efficient for small problems
    for (int32_t i = 0; i < nrhs; ++i) {
        const Real* d_bi = d_B + i * current_n_;
        Real* d_xi = d_X + i * current_n_;
        
        cudaError_t err = solve(d_bi, d_xi);
        if (err != cudaSuccess) return err;
        }
    } else {
        // Batched solve using multiple streams for better GPU utilization
        constexpr int32_t MAX_CONCURRENT = 4;
        int32_t num_streams = std::min(nrhs, MAX_CONCURRENT);
        
        // Create temporary streams for parallel execution
        std::vector<cudaStream_t> streams(num_streams, nullptr);
        for (int32_t s = 0; s < num_streams; ++s) {
            cudaError_t err = cudaStreamCreate(&streams[s]);
            if (err != cudaSuccess) {
                // Cleanup already created streams
                for (int32_t j = 0; j < s; ++j) {
                    cudaStreamDestroy(streams[j]);
                }
                return err;
            }
        }
        
        // Allocate temporary buffers for each stream
        std::vector<Real*> d_temp_buffers(num_streams, nullptr);
        for (int32_t s = 0; s < num_streams; ++s) {
            cudaError_t err = cudaMalloc(&d_temp_buffers[s], current_n_ * sizeof(Real));
            if (err != cudaSuccess) {
                // Cleanup already allocated buffers and all streams
                for (int32_t j = 0; j < s; ++j) {
                    cudaFree(d_temp_buffers[j]);
                }
                for (int32_t j = 0; j < num_streams; ++j) {
                    cudaStreamDestroy(streams[j]);
                }
                return err;
            }
        }
        
        // Process RHS in batches across streams
        cudaError_t last_err = cudaSuccess;
        for (int32_t i = 0; i < nrhs; ++i) {
            int32_t stream_idx = i % num_streams;
            cudaStream_t curr_stream = streams[stream_idx];
            
            const Real* d_bi = d_B + i * current_n_;
            Real* d_xi = d_X + i * current_n_;
            
            // Apply permutation to RHS
            dim3 block(BLOCK_SIZE_STANDARD);
            dim3 grid = compute_grid_size(current_n_, BLOCK_SIZE_STANDARD);
            
            applyVectorPermutationKernel<<<grid, block, 0, curr_stream>>>(
                d_temp_buffers[stream_idx], d_bi, d_perm_, current_n_);
            
            // Solve using cuSOLVER (note: cusolverSpScsrlsvchol is not thread-safe
            // so we synchronize before each solve)
            cudaStreamSynchronize(curr_stream);
            
            int singularity = -1;
#ifdef SLE_USE_DOUBLE
            cusolverStatus_t status = cusolverSpDcsrlsvchol(
                cusolver_handle_, current_n_, current_nnz_, mat_descr_,
                cached_values_, cached_row_ptr_, cached_col_ind_,
                d_temp_buffers[stream_idx], config_.singularity_tol, 0,
                d_xi, &singularity);
#else
            cusolverStatus_t status = cusolverSpScsrlsvchol(
                cusolver_handle_, current_n_, current_nnz_, mat_descr_,
                cached_values_, cached_row_ptr_, cached_col_ind_,
                d_temp_buffers[stream_idx], config_.singularity_tol, 0,
                d_xi, &singularity);
#endif
            
            if (status != CUSOLVER_STATUS_SUCCESS || singularity >= 0) {
                last_err = cudaErrorUnknown;
            }
            
            // Apply inverse permutation
            cudaMemcpyAsync(d_temp_buffers[stream_idx], d_xi, 
                           current_n_ * sizeof(Real), cudaMemcpyDeviceToDevice, curr_stream);
            applyInverseVectorPermutationKernel<<<grid, block, 0, curr_stream>>>(
                d_xi, d_temp_buffers[stream_idx], d_perm_inv_, current_n_);
        }
        
        // Synchronize all streams
        for (int32_t s = 0; s < num_streams; ++s) {
            cudaStreamSynchronize(streams[s]);
        }
        
        // Cleanup
        for (int32_t s = 0; s < num_streams; ++s) {
            cudaFree(d_temp_buffers[s]);
            cudaStreamDestroy(streams[s]);
        }
        
        if (last_err != cudaSuccess) return last_err;
    }
    
    return cudaSuccess;
}

//=============================================================================
// Helper Methods
//=============================================================================

cudaError_t OptimizedCholeskySolver::allocateWorkspace(size_t required) {
    if (workspace_size_ >= required) {
        return cudaSuccess;
    }
    
    if (d_workspace_) {
        cudaFree(d_workspace_);
    }
    
    cudaError_t err = cudaMalloc(&d_workspace_, required);
    if (err == cudaSuccess) {
        workspace_size_ = required;
    }
    return err;
}

cudaError_t OptimizedCholeskySolver::applyPermutation(
    const Real* d_in,
    Real* d_out,
    const int32_t* d_perm,
    int32_t n)
{
    dim3 block(BLOCK_SIZE_STANDARD);
    dim3 grid = compute_grid_size(n, BLOCK_SIZE_STANDARD);
    
    applyVectorPermutationKernel<<<grid, block, 0, stream_>>>(
        d_out, d_in, d_perm, n);
    
    return cudaGetLastError();
}

cudaError_t OptimizedCholeskySolver::applyInversePermutation(
    const Real* d_in,
    Real* d_out,
    const int32_t* d_perm_inv,
    int32_t n)
{
    dim3 block(BLOCK_SIZE_STANDARD);
    dim3 grid = compute_grid_size(n, BLOCK_SIZE_STANDARD);
    
    applyInverseVectorPermutationKernel<<<grid, block, 0, stream_>>>(
        d_out, d_in, d_perm_inv, n);
    
    return cudaGetLastError();
}

void OptimizedCholeskySolver::setConfig(const CholeskyConfig& config) {
    bool reorder_changed = (config.reordering != config_.reordering);
    config_ = config;
    
    if (reorder_changed) {
        cache_.is_valid = false;
        factor_valid_ = false;
    }
}

void OptimizedCholeskySolver::setStream(cudaStream_t stream) {
    stream_ = stream;
    if (cusolver_handle_) {
        cusolverSpSetStream(cusolver_handle_, stream);
    }
    if (cusparse_handle_) {
        cusparseSetStream(cusparse_handle_, stream);
    }
}

//=============================================================================
// Permutation Kernels
//=============================================================================

__global__ void applyVectorPermutationKernel(
    Real* __restrict__ y,
    const Real* __restrict__ x,
    const int32_t* __restrict__ perm,
    int32_t n)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[perm[i]];
    }
}

__global__ void applyInverseVectorPermutationKernel(
    Real* __restrict__ y,
    const Real* __restrict__ x,
    const int32_t* __restrict__ perm_inv,
    int32_t n)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[perm_inv[i]] = x[i];
    }
}

__global__ void applyMatrixPermutationKernel(
    Real* __restrict__ reordered_values,
    int32_t* __restrict__ reordered_col_ind,
    const Real* __restrict__ original_values,
    const int32_t* __restrict__ original_col_ind,
    const int32_t* __restrict__ original_row_ptr,
    const int32_t* __restrict__ reordered_row_ptr,
    const int32_t* __restrict__ perm,
    const int32_t* __restrict__ perm_inv,
    int32_t n)
{
    int32_t new_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (new_row >= n) return;
    
    int32_t old_row = perm[new_row];
    int32_t old_start = original_row_ptr[old_row];
    int32_t old_end = original_row_ptr[old_row + 1];
    int32_t new_start = reordered_row_ptr[new_row];
    
    for (int32_t i = old_start; i < old_end; ++i) {
        int32_t old_col = original_col_ind[i];
        int32_t new_col = perm_inv[old_col];
        
        int32_t new_idx = new_start + (i - old_start);
        reordered_col_ind[new_idx] = new_col;
        reordered_values[new_idx] = original_values[i];
    }
}

__global__ void computeDiagonalScalingKernel(
    Real* __restrict__ d_diag,
    const int32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_ind,
    const Real* __restrict__ values,
    int32_t n)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Find diagonal element
    int32_t row_start = row_ptr[i];
    int32_t row_end = row_ptr[i + 1];
    
    Real diag_val = 0.0f;
    for (int32_t k = row_start; k < row_end; ++k) {
        if (col_ind[k] == i) {
            diag_val = values[k];
            break;
        }
    }
    
    // Compute scaling factor: 1/sqrt(|a_ii|)
    d_diag[i] = (diag_val > SLE_REAL_EPSILON) ? 
                rsqrtf(fabsf(diag_val)) : 1.0f;
}

__global__ void applyDiagonalScalingKernel(
    Real* __restrict__ scaled_values,
    const Real* __restrict__ values,
    const int32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_ind,
    const Real* __restrict__ d_left,
    const Real* __restrict__ d_right,
    int32_t n)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    int32_t row_start = row_ptr[i];
    int32_t row_end = row_ptr[i + 1];
    Real di = d_left[i];
    
    for (int32_t k = row_start; k < row_end; ++k) {
        int32_t j = col_ind[k];
        Real dj = d_right[j];
        scaled_values[k] = values[k] * di * dj;
    }
}

//=============================================================================
// Reordering Algorithms (Host-side)
//=============================================================================

void computeAMDOrdering(
    int32_t n,
    const int32_t* row_ptr,
    const int32_t* col_ind,
    int32_t* perm,
    int32_t* perm_inv)
{
    // Simplified AMD implementation
    // For production, use CHOLMOD's AMD or similar library
    
    // Compute node degrees
    std::vector<int32_t> degree(n);
    #pragma omp parallel for schedule(static) if(n > 500)
    for (int32_t i = 0; i < n; ++i) {
        degree[i] = row_ptr[i + 1] - row_ptr[i];
    }
    
    // Create priority queue (min-heap by degree)
    std::vector<bool> eliminated(n, false);
    std::vector<std::pair<int32_t, int32_t>> nodes(n);  // (degree, index)
    #pragma omp parallel for schedule(static) if(n > 500)
    for (int32_t i = 0; i < n; ++i) {
        nodes[i] = {degree[i], i};
    }
    
    // Sort by degree (using parallel sort if available)
    std::sort(nodes.begin(), nodes.end());
    
    // Simple minimum degree ordering
    for (int32_t k = 0; k < n; ++k) {
        // Find minimum degree non-eliminated node
        int32_t min_node = -1;
        for (const auto& [deg, node] : nodes) {
            if (!eliminated[node]) {
                min_node = node;
                break;
            }
        }
        
        if (min_node < 0) break;
        
        perm[k] = min_node;
        perm_inv[min_node] = k;
        eliminated[min_node] = true;
        
        // Update degrees (simplified - no fill-in update)
    }
}

void computeRCMOrdering(
    int32_t n,
    const int32_t* row_ptr,
    const int32_t* col_ind,
    int32_t* perm,
    int32_t* perm_inv)
{
    // Reverse Cuthill-McKee implementation
    
    if (n == 0) return;
    
    // Find pseudo-peripheral node using BFS
    std::vector<bool> visited(n, false);
    std::vector<int32_t> level(n, -1);
    std::queue<int32_t> queue;
    
    // Start from node 0 (or could find peripheral node first)
    int32_t start = 0;
    
    // BFS to find RCM ordering
    std::vector<int32_t> rcm_order;
    rcm_order.reserve(n);
    
    queue.push(start);
    visited[start] = true;
    level[start] = 0;
    
    while (!queue.empty()) {
        int32_t node = queue.front();
        queue.pop();
        rcm_order.push_back(node);
        
        // Get neighbors sorted by degree
        std::vector<std::pair<int32_t, int32_t>> neighbors;
        for (int32_t k = row_ptr[node]; k < row_ptr[node + 1]; ++k) {
            int32_t neighbor = col_ind[k];
            if (!visited[neighbor]) {
                int32_t deg = row_ptr[neighbor + 1] - row_ptr[neighbor];
                neighbors.push_back({deg, neighbor});
            }
        }
        
        // Sort by degree (ascending)
        std::sort(neighbors.begin(), neighbors.end());
        
        // Add to queue
        for (const auto& [deg, neighbor] : neighbors) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                level[neighbor] = level[node] + 1;
                queue.push(neighbor);
            }
        }
    }
    
    // Handle disconnected components
    for (int32_t i = 0; i < n; ++i) {
        if (!visited[i]) {
            rcm_order.push_back(i);
        }
    }
    
    // Reverse the order (Reverse Cuthill-McKee)
    for (int32_t i = 0; i < n; ++i) {
        perm[i] = rcm_order[n - 1 - i];
        perm_inv[rcm_order[n - 1 - i]] = i;
    }
}

int64_t estimateFactorFillIn(
    int32_t n,
    const int32_t* row_ptr,
    const int32_t* col_ind,
    const int32_t* perm)
{
    // Symbolic factorization to count non-zeros in L
    // Uses elimination tree traversal
    
    std::vector<int32_t> parent(n, -1);
    std::vector<int32_t> level(n, 0);
    std::vector<int32_t> col_counts(n, 0);
    
    // Build elimination tree
    buildEliminationTree(n, row_ptr, col_ind, perm, parent.data());
    
    // Post-order traversal to count column entries
    std::vector<int32_t> post_order;
    post_order.reserve(n);
    
    std::vector<bool> visited(n, false);
    std::function<void(int32_t)> dfs = [&](int32_t node) {
        if (node < 0 || visited[node]) return;
        visited[node] = true;
        
        // Find children
        for (int32_t i = 0; i < n; ++i) {
            if (parent[i] == node) {
                dfs(i);
            }
        }
        post_order.push_back(node);
    };
    
    // Start from roots
    for (int32_t i = 0; i < n; ++i) {
        if (parent[i] < 0) {
            dfs(i);
        }
    }
    
    // Simple estimate: original nnz + fill-in estimate
    int64_t nnz_original = row_ptr[n];
    
    // Estimate fill-in as fraction of dense factor
    // More sophisticated: use symbolic factorization
    int64_t estimated_fill = nnz_original / 3;  // Rough estimate
    
    return nnz_original + estimated_fill;
}

void buildEliminationTree(
    int32_t n,
    const int32_t* row_ptr,
    const int32_t* col_ind,
    const int32_t* perm,
    int32_t* parent)
{
    // Build elimination tree using first-descendant method
    
    std::vector<int32_t> ancestor(n, -1);
    
    for (int32_t k = 0; k < n; ++k) {
        parent[k] = -1;
        ancestor[k] = -1;
        
        int32_t pk = perm[k];  // Original node
        
        // Process row pk (in permuted matrix)
        for (int32_t p = row_ptr[pk]; p < row_ptr[pk + 1]; ++p) {
            int32_t j = col_ind[p];
            
            // Find permuted column index
            int32_t pj = -1;
            for (int32_t i = 0; i < n; ++i) {
                if (perm[i] == j) {
                    pj = i;
                    break;
                }
            }
            
            if (pj >= 0 && pj < k) {
                // Path compression
                int32_t i = pj;
                while (ancestor[i] >= 0 && ancestor[i] != k) {
                    int32_t next = ancestor[i];
                    ancestor[i] = k;
                    i = next;
                }
                
                if (ancestor[i] < 0) {
                    parent[i] = k;
                }
                ancestor[i] = k;
            }
        }
    }
}

//=============================================================================
// Iterative Refinement
//=============================================================================

int32_t applyIterativeRefinement(
    OptimizedCholeskySolver& solver,
    const int32_t* d_A_row_ptr,
    const int32_t* d_A_col_ind,
    const Real* d_A_values,
    const Real* d_b,
    Real* d_x,
    int32_t n,
    const RefinementConfig& config,
    cudaStream_t stream)
{
    // Allocate temporary vectors
    Real* d_r;  // Residual
    Real* d_e;  // Error correction
    cudaMalloc(&d_r, n * sizeof(Real));
    cudaMalloc(&d_e, n * sizeof(Real));
    
    int32_t iter = 0;
    
    for (iter = 0; iter < config.max_iterations; ++iter) {
        // Compute residual: r = b - A*x
        // Using cuSPARSE SpMV
        cusparseHandle_t cusparse_handle = solver.getCusparseHandle();
        
        // First: r = b
        cudaMemcpyAsync(d_r, d_b, n * sizeof(Real),
                        cudaMemcpyDeviceToDevice, stream);
        
        // Then: r = r - A*x (beta = 1, alpha = -1)
        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecR;
        
#ifdef SLE_USE_DOUBLE
        cudaDataType data_type = CUDA_R_64F;
#else
        cudaDataType data_type = CUDA_R_32F;
#endif
        
        // Create descriptors
        int32_t nnz;
        cudaMemcpy(&nnz, d_A_row_ptr + n, sizeof(int32_t), cudaMemcpyDeviceToHost);
        
        cusparseCreateCsr(&matA, n, n, nnz,
                         const_cast<int32_t*>(d_A_row_ptr),
                         const_cast<int32_t*>(d_A_col_ind),
                         const_cast<Real*>(d_A_values),
                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                         CUSPARSE_INDEX_BASE_ZERO, data_type);
        
        cusparseCreateDnVec(&vecX, n, d_x, data_type);
        cusparseCreateDnVec(&vecR, n, d_r, data_type);
        
        // SpMV: r = -1*A*x + 1*r
        Real alpha = -1.0f;
        Real beta = 1.0f;
        
        size_t bufferSize;
        cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecX, &beta, vecR,
                               data_type, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
        
        void* d_buffer;
        cudaMalloc(&d_buffer, bufferSize);
        
        cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, vecX, &beta, vecR,
                    data_type, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        
        cudaFree(d_buffer);
        cusparseDestroySpMat(matA);
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecR);
        
        // Check residual norm for early termination
        // Compute ||r||_2^2 = sum(r_i^2) using Thrust transform_reduce
        thrust::device_ptr<Real> r_ptr(d_r);
        Real r_norm_sq = thrust::transform_reduce(
            thrust::cuda::par.on(stream),
            r_ptr, r_ptr + n,
            [] __host__ __device__ (Real x) { return x * x; },
            0.0f,
            thrust::plus<Real>());
        
        Real r_norm = sqrtf(r_norm_sq);
        
        // Early termination if residual is small enough
        if (r_norm < config.tolerance) {
            cudaFree(d_r);
            cudaFree(d_e);
            return iter + 1;  // Return number of iterations performed
        }
        
        // Solve A*e = r
        (void)solver.solve(d_r, d_e);
        
        // Update: x = x + e
        // Using Thrust for vector addition
        thrust::device_ptr<Real> x_ptr(d_x);
        thrust::device_ptr<Real> e_ptr(d_e);
        thrust::transform(thrust::cuda::par.on(stream),
                         x_ptr, x_ptr + n, e_ptr, x_ptr,
                         thrust::plus<Real>());
    }
    
    cudaFree(d_r);
    cudaFree(d_e);
    
    return iter;
}

} // namespace sle

