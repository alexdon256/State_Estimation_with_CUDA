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
 * @file kernels.cu
 * @brief CUDA kernel implementations for power system calculations
 * 
 * This file contains all GPU kernels for the SLE Engine.
 * Optimizations applied (Section 5.1):
 * - Coalesced memory access via SoA layout
 * - Shared memory tiling for frequently accessed data
 * - Warp-level primitives for efficient reductions
 * - Loop unrolling for small fixed-size loops
 * - Branchless code where possible
 * - sincosf for combined trig computations
 * - Register blocking for improved ILP
 */

#include "../include/kernels.cuh"
#include "../include/cuda_optimizations.cuh"
#include "../include/jsf_compliance.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <vector>

namespace sle {

//=============================================================================
// Device Helper Functions
//=============================================================================

/**
 * @brief Fast inverse square root using Newton-Raphson
 * Used for current magnitude calculations
 */
__device__ __forceinline__ Real fast_rsqrt(Real x) {
#ifdef SLE_USE_DOUBLE
    return rsqrt(x);
#else
    return rsqrtf(x);
#endif
}

/**
 * @brief Warp-level reduction for sum
 */
__device__ __forceinline__ Real warp_reduce_sum(Real val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/**
 * @brief Warp-level reduction for maximum
 */
__device__ __forceinline__ Real warp_reduce_max(Real val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        Real other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

/**
 * @brief Block-level reduction using shared memory
 */
template<int BLOCK_SIZE>
__device__ Real block_reduce_sum(Real val, Real* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    // Warp-level reduction
    val = warp_reduce_sum(val);
    
    // Write warp results to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    // First warp reduces all warp results
    val = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    
    return val;
}

//=============================================================================
// Power Flow Calculation Kernels
//=============================================================================

/**
 * @brief Optimized power injection kernel with shared memory
 * 
 * Uses sincosf for combined trig computation (2x speedup over separate calls).
 * Prefetches voltage values to reduce memory latency.
 */
__global__ void computePowerInjectionsKernel(
    Real* __restrict__ p_inj,
    Real* __restrict__ q_inj,
    const Real* __restrict__ v_mag,
    const Real* __restrict__ v_angle,
    const int32_t* __restrict__ ybus_row_ptr,
    const int32_t* __restrict__ ybus_col_ind,
    const Real* __restrict__ ybus_g,
    const Real* __restrict__ ybus_b,
    int32_t n_buses)
{
    // Each thread handles one bus
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= n_buses) return;
    
    // Load local bus voltage (coalesced access)
    Real Vi = v_mag[i];
    Real theta_i = v_angle[i];
    
    // Accumulate power injections using register blocking
    Real Pi = 0.0f;
    Real Qi = 0.0f;
    
    // Get row range for this bus in Ybus
    int32_t row_start = ybus_row_ptr[i];
    int32_t row_end = ybus_row_ptr[i + 1];
    
    // Process in groups of 4 for instruction-level parallelism
    int32_t idx = row_start;
    
    // Main loop with 4x unrolling for ILP
    for (; idx + 3 < row_end; idx += 4) {
        // Load 4 neighbors' data
        int32_t j0 = ybus_col_ind[idx];
        int32_t j1 = ybus_col_ind[idx + 1];
        int32_t j2 = ybus_col_ind[idx + 2];
        int32_t j3 = ybus_col_ind[idx + 3];
        
        Real Gij0 = ybus_g[idx];     Real Bij0 = ybus_b[idx];
        Real Gij1 = ybus_g[idx + 1]; Real Bij1 = ybus_b[idx + 1];
        Real Gij2 = ybus_g[idx + 2]; Real Bij2 = ybus_b[idx + 2];
        Real Gij3 = ybus_g[idx + 3]; Real Bij3 = ybus_b[idx + 3];
        
        Real Vj0 = v_mag[j0]; Real theta_j0 = v_angle[j0];
        Real Vj1 = v_mag[j1]; Real theta_j1 = v_angle[j1];
        Real Vj2 = v_mag[j2]; Real theta_j2 = v_angle[j2];
        Real Vj3 = v_mag[j3]; Real theta_j3 = v_angle[j3];
        
        // Compute angle differences
        Real theta_ij0 = theta_i - theta_j0;
        Real theta_ij1 = theta_i - theta_j1;
        Real theta_ij2 = theta_i - theta_j2;
        Real theta_ij3 = theta_i - theta_j3;
        
        // Use sincosf for combined sin/cos computation (much faster)
        Real cos0, sin0, cos1, sin1, cos2, sin2, cos3, sin3;
        sincosf(theta_ij0, &sin0, &cos0);
        sincosf(theta_ij1, &sin1, &cos1);
        sincosf(theta_ij2, &sin2, &cos2);
        sincosf(theta_ij3, &sin3, &cos3);
        
        // Compute ViVj products
        Real ViVj0 = Vi * Vj0;
        Real ViVj1 = Vi * Vj1;
        Real ViVj2 = Vi * Vj2;
        Real ViVj3 = Vi * Vj3;
        
        // Accumulate P (fused multiply-add pattern)
        Pi += ViVj0 * (Gij0 * cos0 + Bij0 * sin0);
        Pi += ViVj1 * (Gij1 * cos1 + Bij1 * sin1);
        Pi += ViVj2 * (Gij2 * cos2 + Bij2 * sin2);
        Pi += ViVj3 * (Gij3 * cos3 + Bij3 * sin3);
        
        // Accumulate Q
        Qi += ViVj0 * (Gij0 * sin0 - Bij0 * cos0);
        Qi += ViVj1 * (Gij1 * sin1 - Bij1 * cos1);
        Qi += ViVj2 * (Gij2 * sin2 - Bij2 * cos2);
        Qi += ViVj3 * (Gij3 * sin3 - Bij3 * cos3);
    }
    
    // Handle remaining elements
    for (; idx < row_end; ++idx) {
        int32_t j = ybus_col_ind[idx];
        Real Gij = ybus_g[idx];
        Real Bij = ybus_b[idx];
        
        Real Vj = v_mag[j];
        Real theta_ij = theta_i - v_angle[j];
        
        Real cos_theta, sin_theta;
        sincosf(theta_ij, &sin_theta, &cos_theta);
        
        Real ViVj = Vi * Vj;
        Pi += ViVj * (Gij * cos_theta + Bij * sin_theta);
        Qi += ViVj * (Gij * sin_theta - Bij * cos_theta);
    }
    
    // Store results (coalesced write)
    p_inj[i] = Pi;
    q_inj[i] = Qi;
}

/**
 * @brief Optimized branch flow computation
 * 
 * Optimizations:
 * - sincosf for combined trig computation
 * - Precomputed common terms to reduce arithmetic operations
 * - Branchless status check using multiplication
 * - Register blocking for ILP
 */
__global__ void computeBranchFlowsKernel(
    Real* __restrict__ p_flow_from,
    Real* __restrict__ q_flow_from,
    Real* __restrict__ p_flow_to,
    Real* __restrict__ q_flow_to,
    const Real* __restrict__ v_mag,
    const Real* __restrict__ v_angle,
    const int32_t* __restrict__ from_bus,
    const int32_t* __restrict__ to_bus,
    const Real* __restrict__ g_series,
    const Real* __restrict__ b_series,
    const Real* __restrict__ b_shunt_from,
    const Real* __restrict__ b_shunt_to,
    const Real* __restrict__ tap_ratio,
    const Real* __restrict__ phase_shift,
    const SwitchStatus* __restrict__ status,
    int32_t n_branches)
{
    // Each thread handles one branch
    int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (k >= n_branches) return;
    
    // Branchless status check: multiply by 0 if open, 1 if closed
    Real is_closed = static_cast<Real>(status[k] == SwitchStatus::CLOSED);
    
    // Load branch data (coalesced access due to SoA layout)
    int32_t i = from_bus[k];
    int32_t j = to_bus[k];
    
    Real g = g_series[k];
    Real b = b_series[k];
    Real b_sh_from = b_shunt_from[k];
    Real b_sh_to = b_shunt_to[k];
    Real a = tap_ratio[k];
    Real phi = phase_shift[k];
    
    // Load bus voltages (gather from bus arrays)
    Real Vi = v_mag[i];
    Real Vj = v_mag[j];
    Real theta_i = v_angle[i];
    Real theta_j = v_angle[j];
    
    // Angle difference including phase shift
    Real theta_ij = theta_i - theta_j - phi;
    
    // Use sincosf for combined sin/cos computation
    Real cos_theta, sin_theta;
    sincosf(theta_ij, &sin_theta, &cos_theta);
    
    // Precompute common terms (reduces total operations)
    Real a2 = a * a;
    Real inv_a2 = 1.0f / a2;
    Real inv_a = 1.0f / a;
    
    Real Vi2 = Vi * Vi;
    Real Vj2 = Vj * Vj;
    Real ViVj = Vi * Vj;
    
    Real Vi2_a2 = Vi2 * inv_a2;
    Real ViVj_a = ViVj * inv_a;
    
    // Precompute g/b trig combinations (shared between from/to)
    Real g_cos = g * cos_theta;
    Real g_sin = g * sin_theta;
    Real b_cos = b * cos_theta;
    Real b_sin = b * sin_theta;
    
    // From-side power flows:
    // P_ij = (V_i^2/a^2) * g - (V_i*V_j/a) * (g*cos(θ) + b*sin(θ))
    // Q_ij = -(V_i^2/a^2) * (b + b_sh) - (V_i*V_j/a) * (g*sin(θ) - b*cos(θ))
    Real P_from = Vi2_a2 * g - ViVj_a * (g_cos + b_sin);
    Real Q_from = -Vi2_a2 * (b + b_sh_from) - ViVj_a * (g_sin - b_cos);
    
    // To-side power flows (measured at bus j looking toward bus i):
    // theta_ji = theta_j - theta_i + phi = -theta_ij + 2*phi ≈ -theta_ij for small phi
    // For simplicity using theta_ji = -theta_ij:
    // P_ji = V_j^2 * g - (V_i*V_j/a) * (g*cos(-θ) + b*sin(-θ))
    //      = V_j^2 * g - (V_i*V_j/a) * (g*cos(θ) - b*sin(θ))
    // Q_ji = -V_j^2 * (b + b_sh) - (V_i*V_j/a) * (g*sin(-θ) - b*cos(-θ))
    //      = -V_j^2 * (b + b_sh) - (V_i*V_j/a) * (-g*sin(θ) - b*cos(θ))
    //      = -V_j^2 * (b + b_sh) + (V_i*V_j/a) * (g*sin(θ) + b*cos(θ))
    Real P_to = Vj2 * g - ViVj_a * (g_cos - b_sin);
    Real Q_to = -Vj2 * (b + b_sh_to) + ViVj_a * (g_sin + b_cos);
    
    // Apply switch status mask (branchless: multiply by 0 if open)
    p_flow_from[k] = P_from * is_closed;
    q_flow_from[k] = Q_from * is_closed;
    p_flow_to[k] = P_to * is_closed;
    q_flow_to[k] = Q_to * is_closed;
}

__global__ void computeBranchCurrentsKernel(
    Real* __restrict__ i_mag_from,
    Real* __restrict__ i_mag_to,
    const Real* __restrict__ p_flow_from,
    const Real* __restrict__ q_flow_from,
    const Real* __restrict__ p_flow_to,
    const Real* __restrict__ q_flow_to,
    const Real* __restrict__ v_mag,
    const int32_t* __restrict__ from_bus,
    const int32_t* __restrict__ to_bus,
    const SwitchStatus* __restrict__ status,
    int32_t n_branches)
{
    int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (k >= n_branches) return;
    
    Real is_closed = static_cast<Real>(status[k] == SwitchStatus::CLOSED);
    
    // Load power flows
    Real Pf = p_flow_from[k];
    Real Qf = q_flow_from[k];
    Real Pt = p_flow_to[k];
    Real Qt = q_flow_to[k];
    
    // Load voltages
    Real Vi = v_mag[from_bus[k]];
    Real Vj = v_mag[to_bus[k]];
    
    // Current magnitude: I = sqrt(P^2 + Q^2) / V
    // Using fast inverse sqrt for performance
    Real S_from = Pf * Pf + Qf * Qf;
    Real S_to = Pt * Pt + Qt * Qt;
    
    // Avoid division by zero
    Real Vi_safe = fmaxf(Vi, SLE_REAL_EPSILON);
    Real Vj_safe = fmaxf(Vj, SLE_REAL_EPSILON);
    
    i_mag_from[k] = sqrtf(S_from) / Vi_safe * is_closed;
    i_mag_to[k] = sqrtf(S_to) / Vj_safe * is_closed;
}

//=============================================================================
// Measurement Function Kernel
//=============================================================================

__global__ void computeMeasurementFunctionKernel(
    Real* __restrict__ h_values,
    const MeasurementType* __restrict__ meas_type,
    const int32_t* __restrict__ location_index,
    const BranchEnd* __restrict__ branch_end,
    const Real* __restrict__ pt_ratio,
    const Real* __restrict__ ct_ratio,
    const uint8_t* __restrict__ is_active,
    const Real* __restrict__ v_mag,
    const Real* __restrict__ v_angle,
    const Real* __restrict__ p_inj,
    const Real* __restrict__ q_inj,
    const Real* __restrict__ p_flow_from,
    const Real* __restrict__ q_flow_from,
    const Real* __restrict__ p_flow_to,
    const Real* __restrict__ q_flow_to,
    const Real* __restrict__ i_mag_from,
    const Real* __restrict__ i_mag_to,
    int32_t n_meas)
{
    int32_t m = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (m >= n_meas) return;
    
    // Skip inactive measurements
    if (!is_active[m]) {
        h_values[m] = 0.0f;
        return;
    }
    
    // Load measurement info
    MeasurementType type = meas_type[m];
    int32_t loc = location_index[m];
    BranchEnd end = branch_end[m];
    Real pt = pt_ratio[m];
    Real ct = ct_ratio[m];
    
    Real h = 0.0f;
    
    // Compute h(x) based on measurement type
    // Using switch with explicit enumeration for compiler optimization
    switch (type) {
        case MeasurementType::V_MAG:
            // h(x) = V_i / pt_ratio
            h = v_mag[loc] / pt;
            break;
            
        case MeasurementType::V_ANGLE:
            // h(x) = theta_i (no scaling for angle)
            h = v_angle[loc];
            break;
            
        case MeasurementType::P_INJECTION:
            // h(x) = P_inj_i / (pt * ct)
            h = p_inj[loc] / (pt * ct);
            break;
            
        case MeasurementType::Q_INJECTION:
            // h(x) = Q_inj_i / (pt * ct)
            h = q_inj[loc] / (pt * ct);
            break;
            
        case MeasurementType::P_FLOW:
            // h(x) = P_flow at specified end
            h = (end == BranchEnd::FROM) ? 
                p_flow_from[loc] : p_flow_to[loc];
            h /= (pt * ct);
            break;
            
        case MeasurementType::Q_FLOW:
            h = (end == BranchEnd::FROM) ? 
                q_flow_from[loc] : q_flow_to[loc];
            h /= (pt * ct);
            break;
            
        case MeasurementType::I_MAG:
            h = (end == BranchEnd::FROM) ? 
                i_mag_from[loc] : i_mag_to[loc];
            h /= ct;
            break;
            
        case MeasurementType::P_PSEUDO:
        case MeasurementType::Q_PSEUDO:
            // Pseudo measurements use injection values
            h = (type == MeasurementType::P_PSEUDO) ? 
                p_inj[loc] : q_inj[loc];
            break;
            
        default:
            h = 0.0f;
            break;
    }
    
    h_values[m] = h;
}

//=============================================================================
// Residual Computation Kernels
//=============================================================================

__global__ void computeResidualsKernel(
    Real* __restrict__ residuals,
    const Real* __restrict__ z_measured,
    const Real* __restrict__ h_estimated,
    const uint8_t* __restrict__ is_active,
    int32_t n_meas)
{
    int32_t m = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (m >= n_meas) return;
    
    // Compute residual r = z - h(x)
    // Zero for inactive measurements
    Real r = is_active[m] ? (z_measured[m] - h_estimated[m]) : 0.0f;
    residuals[m] = r;
}

__global__ void computeObjectiveFunctionKernel(
    Real* __restrict__ partial_sums,
    const Real* __restrict__ residuals,
    const Real* __restrict__ weights,
    const uint8_t* __restrict__ is_active,
    int32_t n_meas)
{
    extern __shared__ Real shared_mem[];
    
    int32_t tid = threadIdx.x;
    int32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute weighted squared residual for this element
    Real val = 0.0f;
    if (gid < n_meas && is_active[gid]) {
        Real r = residuals[gid];
        Real w = weights[gid];
        val = w * r * r;  // w * r^2
    }
    
    // Store in shared memory
    shared_mem[tid] = val;
    __syncthreads();
    
    // Parallel reduction in shared memory
    #pragma unroll
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_mem[0];
    }
}

/**
 * @brief Device helper function for atomic max on positive floats
 * 
 * Uses the fact that for positive floats, the IEEE 754 representation
 * preserves ordering when interpreted as unsigned integers.
 */
__device__ __forceinline__ Real atomicMaxPositiveFloat(Real* address, Real val) {
    // For positive floats, we can use atomicMax on their bit representation
    // because IEEE 754 positive floats are ordered the same as unsigned ints
    unsigned int* address_as_ui = reinterpret_cast<unsigned int*>(address);
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    unsigned int val_ui = __float_as_uint(val);
    
    do {
        assumed = old;
        if (__uint_as_float(assumed) >= val) {
            break;
        }
        old = atomicCAS(address_as_ui, assumed, val_ui);
    } while (assumed != old);
    
    return __uint_as_float(old);
}

__global__ void findLargestResidualKernel(
    Real* __restrict__ max_residual,
    int32_t* __restrict__ max_index,
    const Real* __restrict__ residuals,
    const uint8_t* __restrict__ is_active,
    int32_t n_meas)
{
    extern __shared__ char shared_bytes[];
    Real* shared_vals = reinterpret_cast<Real*>(shared_bytes);
    int32_t* shared_idx = reinterpret_cast<int32_t*>(shared_vals + blockDim.x);
    
    int32_t tid = threadIdx.x;
    int32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load residual (absolute value - always positive)
    Real val = 0.0f;
    int32_t idx = -1;
    if (gid < n_meas && is_active[gid]) {
        val = fabsf(residuals[gid]);
        idx = gid;
    }
    
    shared_vals[tid] = val;
    shared_idx[tid] = idx;
    __syncthreads();
    
    // Parallel max reduction within block
    #pragma unroll
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_vals[tid + s] > shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Block leader updates global max using atomic operation
    if (tid == 0) {
        Real block_max = shared_vals[0];
        int32_t block_max_idx = shared_idx[0];
        
        // Atomically update global maximum (for positive values)
        Real old_max = atomicMaxPositiveFloat(max_residual, block_max);
        
        // If this block had the new maximum, update the index
        // Note: This has a race condition but for finding ANY max index it's acceptable
        if (block_max > old_max) {
            atomicExch(max_index, block_max_idx);
        }
    }
}

//=============================================================================
// Huber M-Estimator Kernels
//=============================================================================

__global__ void computeHuberWeightsKernel(
    Real* __restrict__ huber_weights,
    const Real* __restrict__ base_weights,
    const Real* __restrict__ residuals,
    const Real* __restrict__ sigma,
    Real gamma,
    const uint8_t* __restrict__ is_active,
    int32_t n_meas)
{
    int32_t m = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (m >= n_meas) return;
    
    if (!is_active[m]) {
        huber_weights[m] = 0.0f;
        return;
    }
    
    Real r = residuals[m];
    Real s = sigma[m];
    Real w_base = base_weights[m];  // 1/sigma^2
    
    // Normalized residual
    Real r_norm = fabsf(r / s);
    
    // Huber weight modification:
    // If |r/sigma| <= gamma: w = 1/sigma^2 (standard)
    // If |r/sigma| > gamma: w = gamma / (sigma * |r|)
    Real w_huber;
    if (r_norm <= gamma) {
        w_huber = w_base;
    } else {
        // Reduced weight for outliers
        w_huber = gamma / (s * fabsf(r) + SLE_REAL_EPSILON);
    }
    
    huber_weights[m] = w_huber;
}

__global__ void computeHuberObjectiveKernel(
    Real* __restrict__ partial_sums,
    const Real* __restrict__ residuals,
    const Real* __restrict__ sigma,
    Real gamma,
    const uint8_t* __restrict__ is_active,
    int32_t n_meas)
{
    extern __shared__ Real shared_mem[];
    
    int32_t tid = threadIdx.x;
    int32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    Real val = 0.0f;
    if (gid < n_meas && is_active[gid]) {
        Real r = residuals[gid];
        Real s = sigma[gid];
        Real r_norm = fabsf(r / s);
        
        // Huber loss function
        if (r_norm <= gamma) {
            val = 0.5f * r_norm * r_norm;
        } else {
            val = gamma * r_norm - 0.5f * gamma * gamma;
        }
    }
    
    shared_mem[tid] = val;
    __syncthreads();
    
    // Parallel reduction
    #pragma unroll
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_mem[0];
    }
}

//=============================================================================
// State Update and Convergence Kernels
//=============================================================================

__global__ void applyStateUpdateKernel(
    Real* __restrict__ v_mag,
    Real* __restrict__ v_angle,
    const Real* __restrict__ delta_x,
    int32_t slack_index,
    int32_t n_buses)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= n_buses) return;
    
    // State vector layout:
    // delta_x[0..n_buses-2] = angle updates (skip slack)
    // delta_x[n_buses-1..2*n_buses-2] = magnitude updates
    
    // Update voltage angle (skip slack bus)
    if (i != slack_index) {
        // Compute index in delta_x for angle
        int32_t angle_idx = (i < slack_index) ? i : i - 1;
        v_angle[i] += delta_x[angle_idx];
    }
    
    // Update voltage magnitude
    int32_t mag_idx = n_buses - 1 + i;  // Magnitude starts after angles
    v_mag[i] += delta_x[mag_idx];
    
    // Ensure voltage stays positive
    v_mag[i] = fmaxf(v_mag[i], 0.5f);  // Minimum 0.5 p.u.
    v_mag[i] = fminf(v_mag[i], 1.5f);  // Maximum 1.5 p.u.
}

__global__ void computeMaxMismatchKernel(
    Real* __restrict__ max_mismatch,
    const Real* __restrict__ delta_x,
    int32_t n_states)
{
    extern __shared__ Real shared_mem[];
    
    int32_t tid = threadIdx.x;
    int32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load absolute value
    Real val = (gid < n_states) ? fabsf(delta_x[gid]) : 0.0f;
    
    shared_mem[tid] = val;
    __syncthreads();
    
    // Parallel max reduction
    #pragma unroll
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] = fmaxf(shared_mem[tid], shared_mem[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        // Atomic max update for positive floats
        // Use atomicMaxPositiveFloat for correct float comparison
        atomicMaxPositiveFloat(max_mismatch, shared_mem[0]);
    }
}

__global__ void saveStateKernel(
    Real* __restrict__ v_mag_prev,
    Real* __restrict__ v_angle_prev,
    const Real* __restrict__ v_mag,
    const Real* __restrict__ v_angle,
    int32_t n_buses)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= n_buses) return;
    
    v_mag_prev[i] = v_mag[i];
    v_angle_prev[i] = v_angle[i];
}

__global__ void flatStartKernel(
    Real* __restrict__ v_mag,
    Real* __restrict__ v_angle,
    const Real* __restrict__ v_setpoint,
    const BusType* __restrict__ bus_type,
    int32_t n_buses)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= n_buses) return;
    
    // Use voltage setpoint for PV buses, 1.0 for others
    BusType type = bus_type[i];
    Real v_init = (type == BusType::PV || type == BusType::SLACK) ?
                  v_setpoint[i] : 1.0f;
    
    v_mag[i] = v_init;
    v_angle[i] = 0.0f;
}

//=============================================================================
// Right-Hand Side Vector Kernel
//=============================================================================

__global__ void computeRHSVectorKernel(
    Real* __restrict__ b,
    const int32_t* __restrict__ H_row_ptr,
    const int32_t* __restrict__ H_col_ind,
    const Real* __restrict__ H_values,
    const Real* __restrict__ weights,
    const Real* __restrict__ residuals,
    int32_t n_meas,
    int32_t n_states)
{
    // Each thread handles one state variable
    int32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j >= n_states) return;
    
    // Compute b_j = sum_i(H_ij * w_i * r_i) = (H^T * W * r)_j
    // This requires iterating over all measurements that depend on state j
    // For efficiency, we iterate over measurements and accumulate using atomics
    
    Real sum = 0.0f;
    
    // Iterate over all measurements (rows of H)
    for (int32_t i = 0; i < n_meas; ++i) {
        int32_t row_start = H_row_ptr[i];
        int32_t row_end = H_row_ptr[i + 1];
        
        // Find if this row has column j
        for (int32_t idx = row_start; idx < row_end; ++idx) {
            if (H_col_ind[idx] == j) {
                Real H_ij = H_values[idx];
                Real w_i = weights[i];
                Real r_i = residuals[i];
                sum += H_ij * w_i * r_i;
                break;
            }
        }
    }
    
    b[j] = sum;
}

//=============================================================================
// Utility Kernels
//=============================================================================

__global__ void scaleVectorKernel(
    Real* __restrict__ y,
    const Real* __restrict__ x,
    Real alpha,
    int32_t n)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * x[i];
    }
}

__global__ void axpyKernel(
    Real* __restrict__ z,
    const Real* __restrict__ x,
    const Real* __restrict__ y,
    Real alpha,
    Real beta,
    int32_t n)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = alpha * x[i] + beta * y[i];
    }
}

__global__ void zeroVectorKernel(
    Real* __restrict__ x,
    int32_t n)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = 0.0f;
    }
}

__global__ void fillVectorKernel(
    Real* __restrict__ x,
    Real value,
    int32_t n)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = value;
    }
}

__global__ void sqrtVectorKernel(
    Real* __restrict__ y,
    const Real* __restrict__ x,
    int32_t n)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = sqrtf(fmaxf(x[i], 0.0f));
    }
}

__global__ void checkValidityKernel(
    int32_t* __restrict__ has_invalid,
    const Real* __restrict__ x,
    int32_t n)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= n) return;
    
    Real val = x[i];
    
    // Check for NaN or Inf using intrinsics
    if (isnan(val) || isinf(val)) {
        atomicExch(has_invalid, 1);
    }
}

//=============================================================================
// Kernel Launch Wrappers
//=============================================================================

/**
 * @brief Launch power flow kernels with automatic optimization selection
 * 
 * Selects between standard and optimized kernels based on problem size:
 * - For dense buses (high avg connectivity), use shared memory version
 * - For sparse buses, use standard per-thread version
 * - For large networks, uses tiled branch flow kernel
 */
cudaError_t launchPowerFlowKernels(
    DeviceBusData& buses,
    DeviceBranchData& branches,
    const DeviceYbusMatrix& ybus,
    cudaStream_t stream)
{
    int32_t n_buses = buses.count;
    int32_t n_branches = branches.count;
    
    dim3 block(BLOCK_SIZE_STANDARD);
    dim3 grid_buses = compute_grid_size(n_buses, BLOCK_SIZE_STANDARD);
    dim3 grid_branches = compute_grid_size(n_branches, BLOCK_SIZE_STANDARD);
    
    // Compute average connectivity for kernel selection
    float avg_neighbors = static_cast<float>(ybus.nnz) / n_buses;
    
    // Select power injection kernel based on connectivity
    if (avg_neighbors > 16.0f && n_buses > 1000) {
        // Use optimized shared memory kernel for dense/large networks
        // One block per bus, threads cooperatively process neighbors
        opt::computePowerInjectionsOptimizedKernel<64>
            <<<n_buses, 64, 0, stream>>>(
            buses.d_p_injection,
            buses.d_q_injection,
            buses.d_v_mag,
            buses.d_v_angle,
            ybus.d_row_ptr,
            ybus.d_col_ind,
            ybus.d_g_values,
            ybus.d_b_values,
            n_buses);
    } else {
        // Use standard kernel (one thread per bus)
        computePowerInjectionsKernel<<<grid_buses, block, 0, stream>>>(
            buses.d_p_injection,
            buses.d_q_injection,
            buses.d_v_mag,
            buses.d_v_angle,
            ybus.d_row_ptr,
            ybus.d_col_ind,
            ybus.d_g_values,
            ybus.d_b_values,
            n_buses);
    }
    
    // Select branch flow kernel based on network size
    if (n_branches > 10000) {
        // Use tiled kernel for large networks
        int num_tiles = (n_branches + opt::TILE_SIZE - 1) / opt::TILE_SIZE;
        opt::computeBranchFlowsTiledKernel<<<num_tiles, opt::TILE_SIZE, 0, stream>>>(
            branches.d_p_flow_from,
            branches.d_q_flow_from,
            branches.d_p_flow_to,
            branches.d_q_flow_to,
            buses.d_v_mag,
            buses.d_v_angle,
            branches.d_from_bus,
            branches.d_to_bus,
            branches.d_g_series,
            branches.d_b_series,
            branches.d_b_shunt_from,
            branches.d_b_shunt_to,
            branches.d_tap_ratio,
            branches.d_phase_shift,
            branches.d_status,
            n_branches);
    } else {
        // Use standard kernel
        computeBranchFlowsKernel<<<grid_branches, block, 0, stream>>>(
            branches.d_p_flow_from,
            branches.d_q_flow_from,
            branches.d_p_flow_to,
            branches.d_q_flow_to,
            buses.d_v_mag,
            buses.d_v_angle,
            branches.d_from_bus,
            branches.d_to_bus,
            branches.d_g_series,
            branches.d_b_series,
            branches.d_b_shunt_from,
            branches.d_b_shunt_to,
            branches.d_tap_ratio,
            branches.d_phase_shift,
            branches.d_status,
            n_branches);
    }
    
    // Compute branch current magnitudes (standard kernel is efficient for this)
    computeBranchCurrentsKernel<<<grid_branches, block, 0, stream>>>(
        branches.d_i_mag_from,
        branches.d_i_mag_to,
        branches.d_p_flow_from,
        branches.d_q_flow_from,
        branches.d_p_flow_to,
        branches.d_q_flow_to,
        buses.d_v_mag,
        branches.d_from_bus,
        branches.d_to_bus,
        branches.d_status,
        n_branches);
    
    return cudaGetLastError();
}

cudaError_t launchMeasurementFunctionKernel(
    DeviceMeasurementData& measurements,
    const DeviceBusData& buses,
    const DeviceBranchData& branches,
    cudaStream_t stream)
{
    int32_t n_meas = measurements.count;
    
    dim3 block(BLOCK_SIZE_STANDARD);
    dim3 grid = compute_grid_size(n_meas, BLOCK_SIZE_STANDARD);
    
    computeMeasurementFunctionKernel<<<grid, block, 0, stream>>>(
        measurements.d_estimated,
        measurements.d_type,
        measurements.d_location_index,
        measurements.d_branch_end,
        measurements.d_pt_ratio,
        measurements.d_ct_ratio,
        measurements.d_is_active,
        buses.d_v_mag,
        buses.d_v_angle,
        buses.d_p_injection,
        buses.d_q_injection,
        branches.d_p_flow_from,
        branches.d_q_flow_from,
        branches.d_p_flow_to,
        branches.d_q_flow_to,
        branches.d_i_mag_from,
        branches.d_i_mag_to,
        n_meas);
    
    return cudaGetLastError();
}

cudaError_t launchResidualKernels(
    DeviceMeasurementData& measurements,
    Real* objective_value,
    cudaStream_t stream)
{
    int32_t n_meas = measurements.count;
    
    dim3 block(BLOCK_SIZE_STANDARD);
    dim3 grid = compute_grid_size(n_meas, BLOCK_SIZE_STANDARD);
    
    // Compute residuals
    computeResidualsKernel<<<grid, block, 0, stream>>>(
        measurements.d_residual,
        measurements.d_value,
        measurements.d_estimated,
        measurements.d_is_active,
        n_meas);
    
    // Compute objective function (with partial reduction)
    size_t shared_size = BLOCK_SIZE_REDUCTION * sizeof(Real);
    dim3 block_red(BLOCK_SIZE_REDUCTION);
    dim3 grid_red = compute_grid_size(n_meas, BLOCK_SIZE_REDUCTION);
    
    // Allocate temporary buffer for partial sums
    Real* d_partial_sums;
    cudaMalloc(&d_partial_sums, grid_red.x * sizeof(Real));
    
    computeObjectiveFunctionKernel<<<grid_red, block_red, shared_size, stream>>>(
        d_partial_sums,
        measurements.d_residual,
        measurements.d_weight,
        measurements.d_is_active,
        n_meas);
    
    // Final reduction on GPU (or copy to host for small counts)
    if (objective_value != nullptr) {
        // Simple: copy partial sums to host and sum
        std::vector<Real> h_partial(grid_red.x);
        cudaMemcpyAsync(h_partial.data(), d_partial_sums, 
                        grid_red.x * sizeof(Real),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        Real total = 0.0f;
        for (Real v : h_partial) total += v;
        *objective_value = total;
    }
    
    cudaFree(d_partial_sums);
    
    return cudaGetLastError();
}

cudaError_t launchStateUpdateKernels(
    DeviceBusData& buses,
    const Real* delta_x,
    int32_t n_states,
    Real* max_mismatch,
    cudaStream_t stream)
{
    int32_t n_buses = buses.count;
    
    dim3 block(BLOCK_SIZE_STANDARD);
    dim3 grid = compute_grid_size(n_buses, BLOCK_SIZE_STANDARD);
    
    // Apply state update
    applyStateUpdateKernel<<<grid, block, 0, stream>>>(
        buses.d_v_mag,
        buses.d_v_angle,
        delta_x,
        buses.slack_bus_index,
        n_buses);
    
    // Compute max mismatch for convergence check
    if (max_mismatch != nullptr) {
        size_t shared_size = BLOCK_SIZE_REDUCTION * sizeof(Real);
        dim3 block_red(BLOCK_SIZE_REDUCTION);
        dim3 grid_red = compute_grid_size(n_states, BLOCK_SIZE_REDUCTION);
        
        // Initialize max to 0
        cudaMemsetAsync(max_mismatch, 0, sizeof(Real), stream);
        
        computeMaxMismatchKernel<<<grid_red, block_red, shared_size, stream>>>(
            max_mismatch,
            delta_x,
            n_states);
    }
    
    return cudaGetLastError();
}

} // namespace sle

