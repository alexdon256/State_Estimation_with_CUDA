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
 * @file kernels.cuh
 * @brief CUDA kernels for power system calculations
 * 
 * This header defines all GPU kernels for the SLE Engine including:
 * - Power flow equations (P, Q, I calculations)
 * - Measurement function h(x) evaluation
 * - Residual computation and weighting
 * - Jacobian matrix element computation
 * - Convergence checking
 * 
 * All kernels are optimized for:
 * - Coalesced memory access via SoA layout (NFR-03)
 * - Shared memory utilization (Section 5.1)
 * - Loop unrolling (NFR-23)
 * - Warp-level primitives for reductions
 * 
 * @note Kernels use template parameters for precision (float/double)
 *       to support both real-time (fp32) and precision (fp64) modes.
 */

#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "sle_types.cuh"
#include <cuda_runtime.h>

namespace sle {

//=============================================================================
// SECTION 1: Power Flow Calculation Kernels (FR-10)
//=============================================================================

/**
 * @brief Compute power injections at all buses
 * 
 * Calculates P_inj and Q_inj using:
 * P_i = V_i * sum_j(V_j * (G_ij*cos(theta_ij) + B_ij*sin(theta_ij)))
 * Q_i = V_i * sum_j(V_j * (G_ij*sin(theta_ij) - B_ij*cos(theta_ij)))
 * 
 * where theta_ij = theta_i - theta_j
 * 
 * @param p_inj Output active power injections [n_buses]
 * @param q_inj Output reactive power injections [n_buses]
 * @param v_mag Bus voltage magnitudes [n_buses]
 * @param v_angle Bus voltage angles [n_buses]
 * @param ybus_row_ptr Ybus CSR row pointers [n_buses+1]
 * @param ybus_col_ind Ybus CSR column indices [nnz]
 * @param ybus_g Ybus conductance values [nnz]
 * @param ybus_b Ybus susceptance values [nnz]
 * @param n_buses Number of buses
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
    int32_t n_buses);

/**
 * @brief Compute power flows on all branches
 * 
 * For branch k from bus i to bus j with tap ratio a and phase shift phi:
 * 
 * From side (i):
 * P_ij = (V_i^2/a^2)*g - (V_i*V_j/a)*(g*cos(theta_ij-phi) + b*sin(theta_ij-phi))
 * Q_ij = -(V_i^2/a^2)*(b+b_sh) - (V_i*V_j/a)*(g*sin(theta_ij-phi) - b*cos(theta_ij-phi))
 * 
 * @param p_flow_from Output P at from end [n_branches]
 * @param q_flow_from Output Q at from end [n_branches]
 * @param p_flow_to Output P at to end [n_branches]
 * @param q_flow_to Output Q at to end [n_branches]
 * @param v_mag Bus voltage magnitudes
 * @param v_angle Bus voltage angles
 * @param from_bus From bus indices
 * @param to_bus To bus indices
 * @param g_series Series conductance
 * @param b_series Series susceptance
 * @param b_shunt_from Shunt susceptance at from end
 * @param b_shunt_to Shunt susceptance at to end
 * @param tap_ratio Tap ratios
 * @param phase_shift Phase shifts
 * @param status Branch status (OPEN/CLOSED)
 * @param n_branches Number of branches
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
    int32_t n_branches);

/**
 * @brief Compute current magnitudes on all branches
 * 
 * I_mag = sqrt(P^2 + Q^2) / V
 * 
 * @param i_mag_from Output current magnitude at from end [n_branches]
 * @param i_mag_to Output current magnitude at to end [n_branches]
 * @param p_flow_from P at from end
 * @param q_flow_from Q at from end
 * @param p_flow_to P at to end
 * @param q_flow_to Q at to end
 * @param v_mag Bus voltage magnitudes
 * @param from_bus From bus indices
 * @param to_bus To bus indices
 * @param status Branch status
 * @param n_branches Number of branches
 */
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
    int32_t n_branches);

//=============================================================================
// SECTION 2: Measurement Function h(x) Evaluation (FR-10)
//=============================================================================

/**
 * @brief Compute measurement function h(x) for all measurements
 * 
 * Evaluates the estimated measurement value based on measurement type.
 * All values are in per-unit (p.u.) - NO PT/CT scaling applied:
 * - V_MAG: h(x) = V_i
 * - V_ANGLE: h(x) = theta_i
 * - P_INJECTION: h(x) = P_inj_i
 * - Q_INJECTION: h(x) = Q_inj_i
 * - P_FLOW: h(x) = P_flow_k
 * - Q_FLOW: h(x) = Q_flow_k
 * - I_MAG: h(x) = I_mag_k
 * - P_PSEUDO, Q_PSEUDO: h(x) = P/Q_inj
 * 
 * Convention:
 * - All measurements and states are in per-unit (p.u.)
 * - User provides meter readings in p.u.
 * - residual = z_measured - h(x)
 * - PT/CT ratios in kernel signature are reserved for future calibration features
 * - For meters on transformers: bus voltage already includes tap ratio effect
 * 
 * @param h_values Output estimated values [n_meas]
 * @param meas_type Measurement types [n_meas]
 * @param location_index Location indices [n_meas]
 * @param branch_end Branch end indicators [n_meas]
 * @param pt_ratio PT ratios [n_meas] (currently unused, reserved for calibration)
 * @param ct_ratio CT ratios [n_meas] (currently unused, reserved for calibration)
 * @param is_active Active flags [n_meas]
 * @param v_mag Bus voltage magnitudes [n_buses] (p.u.)
 * @param v_angle Bus voltage angles [n_buses] (radians)
 * @param p_inj Bus power injections [n_buses] (p.u.)
 * @param q_inj Bus reactive injections [n_buses] (p.u.)
 * @param p_flow_from Branch P flows from end [n_branches] (p.u.)
 * @param q_flow_from Branch Q flows from end [n_branches] (p.u.)
 * @param p_flow_to Branch P flows to end [n_branches] (p.u.)
 * @param q_flow_to Branch Q flows to end [n_branches] (p.u.)
 * @param i_mag_from Branch I magnitudes from end [n_branches] (p.u.)
 * @param i_mag_to Branch I magnitudes to end [n_branches] (p.u.)
 * @param n_meas Number of measurements
 */
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
    int32_t n_meas);

//=============================================================================
// SECTION 3: Residual Computation (FR-10, FR-15)
//=============================================================================

/**
 * @brief Compute measurement residuals r = z - h(x)
 * 
 * @param residuals Output residual vector [n_meas]
 * @param z_measured Measured values [n_meas]
 * @param h_estimated Estimated values [n_meas]
 * @param is_active Active flags [n_meas]
 * @param n_meas Number of measurements
 */
__global__ void computeResidualsKernel(
    Real* __restrict__ residuals,
    const Real* __restrict__ z_measured,
    const Real* __restrict__ h_estimated,
    const uint8_t* __restrict__ is_active,
    int32_t n_meas);

/**
 * @brief Compute weighted sum of squared residuals J(x) = r^T W r
 * 
 * Uses parallel reduction with warp-level primitives for efficiency.
 * 
 * @param partial_sums Output partial sums (one per block)
 * @param residuals Residual vector [n_meas]
 * @param weights Weight vector [n_meas]
 * @param is_active Active flags [n_meas]
 * @param n_meas Number of measurements
 */
__global__ void computeObjectiveFunctionKernel(
    Real* __restrict__ partial_sums,
    const Real* __restrict__ residuals,
    const Real* __restrict__ weights,
    const uint8_t* __restrict__ is_active,
    int32_t n_meas);

/**
 * @brief Find largest residual and its index (FR-15)
 * 
 * Parallel reduction to find maximum |residual| for bad data detection.
 * 
 * @param max_residual Output maximum residual value
 * @param max_index Output index of maximum residual
 * @param residuals Residual vector [n_meas]
 * @param is_active Active flags [n_meas]
 * @param n_meas Number of measurements
 */
__global__ void findLargestResidualKernel(
    Real* __restrict__ max_residual,
    int32_t* __restrict__ max_index,
    const Real* __restrict__ residuals,
    const uint8_t* __restrict__ is_active,
    int32_t n_meas);

//=============================================================================
// SECTION 4: Huber M-Estimator for Robust Estimation (FR-17)
//=============================================================================

/**
 * @brief Compute Huber weights for robust estimation
 * 
 * The Huber function modifies weights based on residual magnitude:
 * - If |r_i/sigma_i| <= gamma: w_i = 1/sigma_i^2 (standard weight)
 * - If |r_i/sigma_i| > gamma: w_i = gamma/(sigma_i * |r_i|) (reduced weight)
 * 
 * This limits the influence of outliers on the estimation.
 * 
 * @param huber_weights Output modified weights [n_meas]
 * @param base_weights Original weights (1/sigma^2) [n_meas]
 * @param residuals Current residuals [n_meas]
 * @param sigma Standard deviations [n_meas]
 * @param gamma Huber threshold parameter
 * @param is_active Active flags [n_meas]
 * @param n_meas Number of measurements
 */
__global__ void computeHuberWeightsKernel(
    Real* __restrict__ huber_weights,
    const Real* __restrict__ base_weights,
    const Real* __restrict__ residuals,
    const Real* __restrict__ sigma,
    Real gamma,
    const uint8_t* __restrict__ is_active,
    int32_t n_meas);

/**
 * @brief Compute Huber objective function value
 * 
 * For Huber loss function:
 * - If |r| <= gamma*sigma: L = 0.5 * (r/sigma)^2
 * - If |r| > gamma*sigma: L = gamma * |r/sigma| - 0.5*gamma^2
 * 
 * @param partial_sums Output partial sums (one per block)
 * @param residuals Residual vector [n_meas]
 * @param sigma Standard deviations [n_meas]
 * @param gamma Huber threshold parameter
 * @param is_active Active flags [n_meas]
 * @param n_meas Number of measurements
 */
__global__ void computeHuberObjectiveKernel(
    Real* __restrict__ partial_sums,
    const Real* __restrict__ residuals,
    const Real* __restrict__ sigma,
    Real gamma,
    const uint8_t* __restrict__ is_active,
    int32_t n_meas);

//=============================================================================
// SECTION 5: State Update and Convergence (FR-08, FR-14)
//=============================================================================

/**
 * @brief Apply state update: x_new = x + delta_x
 * 
 * Updates voltage magnitudes and angles based on WLS solution.
 * 
 * @param v_mag Voltage magnitudes [n_buses], updated in place
 * @param v_angle Voltage angles [n_buses], updated in place
 * @param delta_x State update vector [2*n_buses - 1]
 *               First (n_buses-1) elements are angle updates (skip slack)
 *               Next n_buses elements are magnitude updates
 * @param slack_index Index of slack bus (angle not updated)
 * @param n_buses Number of buses
 */
__global__ void applyStateUpdateKernel(
    Real* __restrict__ v_mag,
    Real* __restrict__ v_angle,
    const Real* __restrict__ delta_x,
    int32_t slack_index,
    int32_t n_buses);

/**
 * @brief Compute maximum state mismatch for convergence check
 * 
 * Calculates max(|delta_x|) for convergence determination.
 * Uses parallel reduction.
 * 
 * @param max_mismatch Output maximum mismatch value
 * @param delta_x State update vector [n_states]
 * @param n_states Number of state variables
 */
__global__ void computeMaxMismatchKernel(
    Real* __restrict__ max_mismatch,
    const Real* __restrict__ delta_x,
    int32_t n_states);

/**
 * @brief Save current state for next iteration comparison
 * 
 * @param v_mag_prev Previous V magnitudes [n_buses]
 * @param v_angle_prev Previous angles [n_buses]
 * @param v_mag Current V magnitudes [n_buses]
 * @param v_angle Current angles [n_buses]
 * @param n_buses Number of buses
 */
__global__ void saveStateKernel(
    Real* __restrict__ v_mag_prev,
    Real* __restrict__ v_angle_prev,
    const Real* __restrict__ v_mag,
    const Real* __restrict__ v_angle,
    int32_t n_buses);

/**
 * @brief Initialize flat start (V=1.0 p.u., theta=0)
 * 
 * @param v_mag Output voltage magnitudes [n_buses]
 * @param v_angle Output voltage angles [n_buses]
 * @param v_setpoint Voltage setpoints for PV buses [n_buses]
 * @param bus_type Bus types [n_buses]
 * @param n_buses Number of buses
 */
__global__ void flatStartKernel(
    Real* __restrict__ v_mag,
    Real* __restrict__ v_angle,
    const Real* __restrict__ v_setpoint,
    const BusType* __restrict__ bus_type,
    int32_t n_buses);

//=============================================================================
// SECTION 6: Right-Hand Side Vector Construction (FR-08)
//=============================================================================

/**
 * @brief Compute right-hand side vector b = H^T * W * r
 * 
 * This is the gradient of the objective function.
 * 
 * @param b Output RHS vector [n_states]
 * @param H_row_ptr Jacobian CSR row pointers
 * @param H_col_ind Jacobian CSR column indices
 * @param H_values Jacobian values
 * @param weights Measurement weights [n_meas]
 * @param residuals Measurement residuals [n_meas]
 * @param n_meas Number of measurements
 * @param n_states Number of state variables
 */
__global__ void computeRHSVectorKernel(
    Real* __restrict__ b,
    const int32_t* __restrict__ H_row_ptr,
    const int32_t* __restrict__ H_col_ind,
    const Real* __restrict__ H_values,
    const Real* __restrict__ weights,
    const Real* __restrict__ residuals,
    int32_t n_meas,
    int32_t n_states);

//=============================================================================
// SECTION 7: Utility Kernels
//=============================================================================

/**
 * @brief Vector scaling: y = alpha * x
 */
__global__ void scaleVectorKernel(
    Real* __restrict__ y,
    const Real* __restrict__ x,
    Real alpha,
    int32_t n);

/**
 * @brief Vector addition: z = alpha*x + beta*y
 */
__global__ void axpyKernel(
    Real* __restrict__ z,
    const Real* __restrict__ x,
    const Real* __restrict__ y,
    Real alpha,
    Real beta,
    int32_t n);

/**
 * @brief Set vector elements to zero
 */
__global__ void zeroVectorKernel(
    Real* __restrict__ x,
    int32_t n);

/**
 * @brief Set vector elements to constant value
 */
__global__ void fillVectorKernel(
    Real* __restrict__ x,
    Real value,
    int32_t n);

/**
 * @brief Compute element-wise square root: y = sqrt(x)
 */
__global__ void sqrtVectorKernel(
    Real* __restrict__ y,
    const Real* __restrict__ x,
    int32_t n);

/**
 * @brief Check for NaN or Inf values (divergence detection)
 * 
 * @param has_invalid Output flag (set to 1 if NaN/Inf found)
 * @param x Vector to check [n]
 * @param n Vector length
 */
__global__ void checkValidityKernel(
    int32_t* __restrict__ has_invalid,
    const Real* __restrict__ x,
    int32_t n);

//=============================================================================
// SECTION 8: Kernel Launch Wrappers
//=============================================================================

/**
 * @brief Launch power flow calculation kernels
 * 
 * Computes injections, flows, and currents in sequence.
 * 
 * @param buses Device bus data
 * @param branches Device branch data
 * @param ybus Device Ybus matrix
 * @param stream CUDA stream
 * @return cudaSuccess on success
 */
[[nodiscard]] cudaError_t launchPowerFlowKernels(
    DeviceBusData& buses,
    DeviceBranchData& branches,
    const DeviceYbusMatrix& ybus,
    cudaStream_t stream = nullptr);

/**
 * @brief Launch measurement function evaluation
 * 
 * @param measurements Device measurement data
 * @param buses Device bus data
 * @param branches Device branch data
 * @param stream CUDA stream
 * @return cudaSuccess on success
 */
[[nodiscard]] cudaError_t launchMeasurementFunctionKernel(
    DeviceMeasurementData& measurements,
    const DeviceBusData& buses,
    const DeviceBranchData& branches,
    cudaStream_t stream = nullptr);

/**
 * @brief Launch residual and objective computation
 * 
 * @param measurements Device measurement data
 * @param objective_value Output objective function value
 * @param stream CUDA stream
 * @return cudaSuccess on success
 */
[[nodiscard]] cudaError_t launchResidualKernels(
    DeviceMeasurementData& measurements,
    Real* objective_value,
    cudaStream_t stream = nullptr);

/**
 * @brief Launch state update and convergence check
 * 
 * @param buses Device bus data
 * @param delta_x State update vector
 * @param n_states Number of state variables
 * @param max_mismatch Output maximum mismatch
 * @param stream CUDA stream
 * @return cudaSuccess on success
 */
[[nodiscard]] cudaError_t launchStateUpdateKernels(
    DeviceBusData& buses,
    const Real* delta_x,
    int32_t n_states,
    Real* max_mismatch,
    cudaStream_t stream = nullptr);

} // namespace sle

#endif // KERNELS_CUH

