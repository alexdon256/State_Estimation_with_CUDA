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
 * @file sle_types.cuh
 * @brief Core type definitions for the CUDA-Accelerated State Load Estimator (SLE) Engine
 * 
 * This header defines all fundamental types, enumerations, and Structure-of-Arrays (SoA)
 * data layouts used throughout the SLE engine. SoA layout is critical for coalesced
 * GPU memory access patterns (NFR-03).
 * 
 * @note All types follow JSF C++ compliance (NFR-25) with explicit constructors,
 *       scoped enumerations, and [[nodiscard]] attributes where appropriate.
 */

#ifndef SLE_TYPES_CUH
#define SLE_TYPES_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

namespace sle {

//=============================================================================
// SECTION 1: Platform and Precision Configuration
//=============================================================================

/**
 * @brief Precision typedef for numerical computations
 * 
 * NFR-04: Support float (fp32) for maximum speed in real-time mode.
 * Define SLE_USE_DOUBLE to switch to double precision for offline analysis.
 */
#ifdef SLE_USE_DOUBLE
    using Real = double;
    #define SLE_REAL_EPSILON 1e-12
    #define SLE_CONVERGENCE_TOL 1e-6
#else
    using Real = float;
    #define SLE_REAL_EPSILON 1e-6f
    #define SLE_CONVERGENCE_TOL 1e-4f
#endif

/// Cache line size for alignment (NFR-11)
constexpr std::size_t CACHE_LINE_SIZE = 64;

/// CUDA warp size for kernel optimization
constexpr int WARP_SIZE = 32;

/// Maximum iterations for real-time mode (FR-11)
constexpr int MAX_REALTIME_ITERATIONS = 5;

/// Maximum iterations for precision mode (FR-12)
constexpr int MAX_PRECISION_ITERATIONS = 100;

/// Invalid index sentinel value
constexpr int32_t INVALID_INDEX = -1;

/// Default Huber threshold for robust estimation (FR-17)
constexpr Real DEFAULT_HUBER_GAMMA = 1.5f;

//=============================================================================
// SECTION 2: Scoped Enumerations (JSF C++ Compliant)
//=============================================================================

/**
 * @brief Bus type classification for power flow
 */
enum class BusType : uint8_t {
    PQ = 0,     ///< Load bus (P and Q specified)
    PV = 1,     ///< Generator bus (P and V specified)
    SLACK = 2   ///< Reference/Swing bus (V and theta specified)
};

/**
 * @brief Switching device status (FR-03)
 */
enum class SwitchStatus : uint8_t {
    OPEN = 0,   ///< Branch disconnected from Ybus
    CLOSED = 1  ///< Branch connected (zero or low impedance)
};

/**
 * @brief Measurement type classification (FR-01, Section 3.7.1)
 */
enum class MeasurementType : uint8_t {
    // Bus measurements
    V_MAG = 0,          ///< Bus voltage magnitude
    V_ANGLE = 1,        ///< Bus voltage angle (PMU)
    P_INJECTION = 2,    ///< Bus active power injection
    Q_INJECTION = 3,    ///< Bus reactive power injection
    
    // Branch measurements (require from/to specification)
    P_FLOW = 4,         ///< Branch active power flow
    Q_FLOW = 5,         ///< Branch reactive power flow
    I_MAG = 6,          ///< Branch current magnitude
    
    // Pseudo measurements for observability
    P_PSEUDO = 7,       ///< Pseudo active power
    Q_PSEUDO = 8,       ///< Pseudo reactive power
    
    INVALID = 255       ///< Sentinel for uninitialized
};

/**
 * @brief Branch end indicator for flow measurements
 */
enum class BranchEnd : uint8_t {
    FROM = 0,   ///< Measurement at "from" bus end
    TO = 1      ///< Measurement at "to" bus end
};

/**
 * @brief Convergence status codes (FR-14)
 */
enum class [[nodiscard]] ConvergenceStatus : uint8_t {
    CONVERGED = 0,          ///< Solution found within tolerance
    MAX_ITERATIONS = 1,     ///< Iteration limit reached
    DIVERGED = 2,           ///< NaN/Inf detected
    SINGULAR_MATRIX = 3,    ///< Gain matrix singular (observability issue)
    NOT_OBSERVABLE = 4,     ///< Network not observable (FR-13)
    IN_PROGRESS = 5         ///< Estimation still running
};

/**
 * @brief Estimation mode selection (FR-11, FR-12)
 */
enum class EstimationMode : uint8_t {
    REALTIME = 0,   ///< Time-constrained with hot start and G reuse
    PRECISION = 1   ///< Full convergence with G refactorization
};

//=============================================================================
// SECTION 3: Host-Side Data Structures (SoA Layout)
//=============================================================================

/**
 * @brief Structure-of-Arrays for Bus data on host (NFR-03)
 * 
 * Each array is independently allocated for optimal memory access.
 * Indices are used consistently across all arrays.
 */
struct alignas(CACHE_LINE_SIZE) HostBusData {
    int32_t count;                  ///< Number of buses
    
    // Static parameters (FR-01)
    Real* base_kv;                  ///< Base voltage [kV]
    BusType* bus_type;              ///< Bus type (PQ, PV, SLACK)
    int32_t* external_id;           ///< External identifier for mapping
    
    // State variables (voltage phasor)
    Real* v_mag;                    ///< Voltage magnitude [p.u.]
    Real* v_angle;                  ///< Voltage angle [radians]
    
    // Derived quantities (FR-10)
    Real* p_injection;              ///< Active power injection [p.u.]
    Real* q_injection;              ///< Reactive power injection [p.u.]
    
    // Scheduled values for PV/PQ buses
    Real* p_scheduled;              ///< Scheduled active power [p.u.]
    Real* q_scheduled;              ///< Scheduled reactive power [p.u.]
    Real* v_setpoint;               ///< Voltage setpoint for PV buses [p.u.]
    
    // Index of slack bus (used for angle reference)
    int32_t slack_bus_index;
};

/**
 * @brief Structure-of-Arrays for Branch data on host (NFR-03)
 * 
 * Includes transmission lines and transformers with tap settings.
 */
struct alignas(CACHE_LINE_SIZE) HostBranchData {
    int32_t count;                  ///< Number of branches
    
    // Topology (FR-01)
    int32_t* from_bus;              ///< From bus index
    int32_t* to_bus;                ///< To bus index
    
    // Impedance parameters [p.u.] (FR-01)
    Real* resistance;               ///< Series resistance R
    Real* reactance;                ///< Series reactance X
    Real* susceptance;              ///< Total line charging susceptance B
    Real* conductance;              ///< Shunt conductance G (usually 0)
    
    // Transformer parameters (FR-01)
    Real* tap_ratio;                ///< Off-nominal tap ratio (1.0 for lines)
    Real* phase_shift;              ///< Phase shift angle [radians]
    bool* is_transformer;           ///< True if branch is a transformer
    
    // Switching device status (FR-03, FR-04)
    SwitchStatus* status;           ///< Current switch status
    int32_t* sd_index;              ///< Associated switching device index (-1 if none)
    
    // Derived admittance components (cached for performance)
    Real* g_series;                 ///< Series conductance
    Real* b_series;                 ///< Series susceptance  
    Real* b_shunt_from;             ///< Shunt susceptance at from end
    Real* b_shunt_to;               ///< Shunt susceptance at to end
    
    // Flow calculations (FR-10)
    Real* p_flow_from;              ///< Active power flow (from end) [p.u.]
    Real* q_flow_from;              ///< Reactive power flow (from end) [p.u.]
    Real* p_flow_to;                ///< Active power flow (to end) [p.u.]
    Real* q_flow_to;                ///< Reactive power flow (to end) [p.u.]
    Real* i_mag_from;               ///< Current magnitude (from end) [p.u.]
    Real* i_mag_to;                 ///< Current magnitude (to end) [p.u.]
};

/**
 * @brief Measurement association structure (FR-02)
 * 
 * Maps each measurement in the telemetry vector z to its network location.
 * Designed for constant-time lookup during parallel Jacobian construction.
 */
struct alignas(CACHE_LINE_SIZE) MeasurementAssociation {
    MeasurementType type;           ///< Type of measurement
    int32_t location_index;         ///< Bus or branch index
    BranchEnd branch_end;           ///< For branch measurements: FROM or TO
    
    // Instrument transformer ratios (FR-01)
    Real pt_ratio;                  ///< Potential transformer ratio
    Real ct_ratio;                  ///< Current transformer ratio
    
    // Measurement quality
    Real weight;                    ///< Measurement weight (1/variance)
    Real sigma;                     ///< Standard deviation
    bool is_active;                 ///< Whether measurement is currently active
    bool is_pseudo;                 ///< True for pseudo/virtual measurements
};

/**
 * @brief Structure-of-Arrays for Measurement data on host (FR-02)
 */
struct alignas(CACHE_LINE_SIZE) HostMeasurementData {
    int32_t count;                  ///< Total number of measurements
    int32_t active_count;           ///< Number of active measurements
    
    // Measurement associations (FR-02)
    MeasurementType* type;          ///< Measurement type
    int32_t* location_index;        ///< Bus or branch index
    BranchEnd* branch_end;          ///< Branch end indicator
    
    // Instrument ratios
    Real* pt_ratio;                 ///< PT ratios
    Real* ct_ratio;                 ///< CT ratios
    
    // Values and weights
    Real* value;                    ///< Measured value z
    Real* weight;                   ///< Weight (1/variance)
    Real* sigma;                    ///< Standard deviation
    
    // State flags
    bool* is_active;                ///< Active flag
    bool* is_pseudo;                ///< Pseudo measurement flag
    
    // Calculated values (FR-10)
    Real* estimated;                ///< Estimated value h(x)
    Real* residual;                 ///< Residual z - h(x)
};

/**
 * @brief Switching device data (FR-03)
 */
struct alignas(CACHE_LINE_SIZE) HostSwitchingDeviceData {
    int32_t count;                  ///< Number of switching devices
    
    int32_t* branch_index;          ///< Associated branch index
    SwitchStatus* status;           ///< Current status
    SwitchStatus* pending_status;   ///< Queued status change (FR-05)
    bool* has_pending_change;       ///< Flag for pending change
};

//=============================================================================
// SECTION 4: Device-Side Data Structures (GPU Memory)
//=============================================================================

/**
 * @brief Device pointers for bus data
 * 
 * Mirrors HostBusData but uses device memory pointers.
 * Allocated once and persisted (FR-06).
 */
struct DeviceBusData {
    int32_t count;
    
    Real* d_base_kv;
    BusType* d_bus_type;
    
    Real* d_v_mag;
    Real* d_v_angle;
    Real* d_v_mag_prev;             ///< Previous iteration for convergence check
    Real* d_v_angle_prev;
    
    Real* d_p_injection;
    Real* d_q_injection;
    Real* d_p_scheduled;
    Real* d_q_scheduled;
    Real* d_v_setpoint;
    
    int32_t slack_bus_index;
};

/**
 * @brief Device pointers for branch data
 */
struct DeviceBranchData {
    int32_t count;
    
    int32_t* d_from_bus;
    int32_t* d_to_bus;
    
    Real* d_resistance;
    Real* d_reactance;
    Real* d_susceptance;
    Real* d_conductance;
    
    Real* d_tap_ratio;
    Real* d_phase_shift;
    
    Real* d_g_series;
    Real* d_b_series;
    Real* d_b_shunt_from;
    Real* d_b_shunt_to;
    
    SwitchStatus* d_status;
    
    Real* d_p_flow_from;
    Real* d_q_flow_from;
    Real* d_p_flow_to;
    Real* d_q_flow_to;
    Real* d_i_mag_from;
    Real* d_i_mag_to;
};

/**
 * @brief Device pointers for measurement data
 */
struct DeviceMeasurementData {
    int32_t count;
    int32_t active_count;
    
    MeasurementType* d_type;
    int32_t* d_location_index;
    BranchEnd* d_branch_end;
    
    Real* d_pt_ratio;
    Real* d_ct_ratio;
    
    Real* d_value;                  ///< Measured values z
    Real* d_weight;                 ///< Weights W diagonal (1/sigma^2)
    Real* d_sigma;                  ///< Standard deviation for Huber M-estimator
    
    uint8_t* d_is_active;           ///< Active flag (0 or 1, not bool to avoid alignment issues)
    
    Real* d_estimated;              ///< h(x)
    Real* d_residual;               ///< z - h(x)
};

//=============================================================================
// SECTION 5: Sparse Matrix Structures (Section 5.1)
//=============================================================================

/**
 * @brief CSR format sparse matrix on device
 * 
 * Used for Jacobian H, Gain matrix G, and Ybus.
 * Optimized for cuSPARSE operations.
 */
struct DeviceCSRMatrix {
    int32_t rows;                   ///< Number of rows
    int32_t cols;                   ///< Number of columns
    int32_t nnz;                    ///< Number of non-zeros
    
    int32_t* d_row_ptr;             ///< Row pointers [rows+1]
    int32_t* d_col_ind;             ///< Column indices [nnz]
    Real* d_values;                 ///< Non-zero values [nnz]
    
    // cuSPARSE descriptor handle (set during initialization)
    void* cusparse_descr;           ///< cusparseSpMatDescr_t
};

/**
 * @brief Ybus admittance matrix data
 * 
 * Complex admittance stored as separate real/imaginary CSR matrices
 * for efficient GPU computation.
 */
struct DeviceYbusMatrix {
    int32_t n_buses;
    int32_t nnz;
    
    int32_t* d_row_ptr;
    int32_t* d_col_ind;
    Real* d_g_values;               ///< Conductance (real part)
    Real* d_b_values;               ///< Susceptance (imaginary part)
    
    bool is_valid;                  ///< Set to false when topology changes
};

//=============================================================================
// SECTION 6: Solver Configuration and Results
//=============================================================================

/**
 * @brief WLS solver configuration parameters
 */
struct SolverConfig {
    EstimationMode mode;            ///< Real-time or precision mode
    int32_t max_iterations;         ///< Maximum WLS iterations
    Real convergence_tolerance;     ///< Mismatch threshold
    Real huber_gamma;               ///< Huber function threshold (FR-17)
    bool use_robust_estimation;     ///< Enable Huber M-estimator
    bool use_flat_start;            ///< Force flat start initialization
    Real time_limit_ms;             ///< Time limit for real-time mode [ms]
    
    /// Default constructor with standard settings
    SolverConfig() :
        mode(EstimationMode::REALTIME),
        max_iterations(MAX_REALTIME_ITERATIONS),
        convergence_tolerance(SLE_CONVERGENCE_TOL),
        huber_gamma(DEFAULT_HUBER_GAMMA),
        use_robust_estimation(false),
        use_flat_start(false),
        time_limit_ms(20.0f) {}
};

/**
 * @brief Estimation result structure (FR-10, FR-14)
 */
struct [[nodiscard]] EstimationResult {
    ConvergenceStatus status;       ///< Convergence outcome
    int32_t iterations;             ///< Number of iterations performed
    Real max_mismatch;              ///< Maximum state mismatch
    Real objective_value;           ///< J(x) = r^T W r
    Real computation_time_ms;       ///< Total GPU computation time
    
    // Statistics
    Real largest_residual;          ///< Largest measurement residual
    int32_t largest_residual_idx;   ///< Index of largest residual
    int32_t bad_data_count;         ///< Number of suspected bad data points
    
    /// Default constructor
    EstimationResult() :
        status(ConvergenceStatus::IN_PROGRESS),
        iterations(0),
        max_mismatch(std::numeric_limits<Real>::max()),
        objective_value(0),
        computation_time_ms(0),
        largest_residual(0),
        largest_residual_idx(-1),
        bad_data_count(0) {}
};

//=============================================================================
// SECTION 7: CUDA Kernel Launch Configuration
//=============================================================================

/**
 * @brief Compute optimal thread block configuration for a kernel
 * 
 * @param n Number of elements to process
 * @param block_size Desired threads per block (default 256)
 * @return Grid dimension (number of blocks)
 */
__host__ __forceinline__ 
dim3 compute_grid_size(int32_t n, int32_t block_size = 256) {
    return dim3((n + block_size - 1) / block_size);
}

/**
 * @brief Standard block size for most kernels
 */
constexpr int BLOCK_SIZE_STANDARD = 256;

/**
 * @brief Block size for reduction operations
 */
constexpr int BLOCK_SIZE_REDUCTION = 512;

/**
 * @brief Block size for sparse matrix operations
 */
constexpr int BLOCK_SIZE_SPARSE = 128;

//=============================================================================
// SECTION 8: Error Checking Macros
//=============================================================================

/**
 * @brief CUDA error checking macro with file/line information
 * 
 * Usage: CUDA_CHECK(cudaMalloc(&ptr, size));
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return err; \
        } \
    } while(0)

/**
 * @brief CUDA error checking macro that returns error code (JSF C++ compliant)
 * 
 * NFR-24: JSF C++ prohibits exceptions. Use return codes for error handling.
 */
#define CUDA_CHECK_RETURN(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return err; \
        } \
    } while(0)

/**
 * @brief CUDA error checking macro with boolean return
 * 
 * Returns false on error, true on success.
 */
#define CUDA_CHECK_BOOL(call) \
    ([&]() -> bool { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return false; \
        } \
        return true; \
    }())

/**
 * @brief cuSPARSE error checking macro
 */
#define CUSPARSE_CHECK(call) \
    do { \
        cusparseStatus_t status = (call); \
        if (status != CUSPARSE_STATUS_SUCCESS) { \
            fprintf(stderr, "cuSPARSE Error at %s:%d - code %d\n", \
                    __FILE__, __LINE__, status); \
            return cudaErrorUnknown; \
        } \
    } while(0)

/**
 * @brief cuSOLVER error checking macro
 */
#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t status = (call); \
        if (status != CUSOLVER_STATUS_SUCCESS) { \
            fprintf(stderr, "cuSOLVER Error at %s:%d - code %d\n", \
                    __FILE__, __LINE__, status); \
            return cudaErrorUnknown; \
        } \
    } while(0)

} // namespace sle

#endif // SLE_TYPES_CUH

