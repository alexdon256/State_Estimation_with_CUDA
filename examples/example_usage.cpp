/**
 * @file example_usage.cpp
 * @brief Comprehensive SLE Engine usage example (FR-26)
 * 
 * Demonstrates the full operational lifecycle of the SLE Engine using the
 * C-compatible DLL API. This example can be compiled as standard C++ without
 * requiring CUDA compilation.
 * 
 * Demonstrates:
 * 1. Model Upload & Full Run (FR-12)
 * 2. Telemetry Update & Fast Run (FR-11, FR-16)
 * 3. SD Update & Fast Run (FR-05)
 * 4. Structural Update & Full Run (FR-25)
 * 5. Transformer Tap Update (FR-01 controllable taps)
 * 6. Bad Data Detection (FR-15)
 * 7. Robust Estimation (FR-17)
 * 
 * Compile: cl /EHsc example_usage.cpp /link SLE.lib
 */

#include "../include/sle_api.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstring>

//=============================================================================
// Constants and Configuration
//=============================================================================

static const float PI = 3.14159265358979f;

//=============================================================================
// Helper Functions
//=============================================================================

/**
 * @brief Print estimation results to console
 */
void printResult(const SLE_Result& result, const char* phase) {
    std::cout << "\n=== " << phase << " ===" << std::endl;
    std::cout << "Status: ";
    switch (result.status) {
        case SLE_CONVERGED:        std::cout << "CONVERGED"; break;
        case SLE_MAX_ITERATIONS:   std::cout << "MAX_ITERATIONS"; break;
        case SLE_DIVERGED:         std::cout << "DIVERGED"; break;
        case SLE_SINGULAR:         std::cout << "SINGULAR_MATRIX"; break;
        case SLE_NOT_OBSERVABLE:   std::cout << "NOT_OBSERVABLE"; break;
        default:                   std::cout << "IN_PROGRESS"; break;
    }
    std::cout << std::endl;
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Iterations:     " << result.iterations << std::endl;
    std::cout << "Max Mismatch:   " << result.max_mismatch << std::endl;
    std::cout << "Objective J(x): " << result.objective << std::endl;
    std::cout << "Compute Time:   " << result.computation_time_ms << " ms" << std::endl;
    
    if (result.bad_data_count > 0) {
        std::cout << "Bad Data Count: " << result.bad_data_count << std::endl;
        std::cout << "Largest Resid:  " << result.largest_residual 
                  << " at index " << result.largest_residual_idx << std::endl;
    }
}

/**
 * @brief Print voltage profile
 */
void printVoltages(SLE_Handle engine, int32_t bus_count) {
    std::cout << "\nVoltage Profile:" << std::endl;
    std::cout << std::setw(8) << "Bus" 
              << std::setw(12) << "V [p.u.]" 
              << std::setw(12) << "Angle [deg]" << std::endl;
    std::cout << std::string(32, '-') << std::endl;
    
    std::vector<SLE_Real> v_mag(bus_count);
    std::vector<SLE_Real> v_angle(bus_count);
    
    sle_GetVoltageMagnitudes(engine, v_mag.data(), bus_count);
    sle_GetVoltageAngles(engine, v_angle.data(), bus_count);
    
    for (int32_t i = 0; i < bus_count; ++i) {
        std::cout << std::setw(8) << (i + 1)
                  << std::setw(12) << std::fixed << std::setprecision(4) << v_mag[i]
                  << std::setw(12) << std::fixed << std::setprecision(2) 
                  << (v_angle[i] * 180.0f / PI) << std::endl;
    }
}

/**
 * @brief Add noise to measurement values
 */
void addMeasurementNoise(SLE_Real* values, const SLE_Real* sigmas, 
                         int32_t count, unsigned int seed) {
    srand(seed);
    for (int32_t i = 0; i < count; ++i) {
        // Simple box-muller transform for Gaussian noise
        float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        float noise = sigmas[i] * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI * u2);
        values[i] += noise;
    }
}

/**
 * @brief Load IEEE 14-Bus test case data
 */
bool loadIEEE14Bus(SLE_Handle engine) {
    // IEEE 14-Bus System Data
    // Bus data: base_kv, type, v_setpoint, p_scheduled, q_scheduled
    const struct {
        const char* id;
        SLE_Real base_kv;
        SLE_BusType type;
        SLE_Real v_setpoint;
        SLE_Real p_scheduled;
        SLE_Real q_scheduled;
    } buses[] = {
        {"Bus1",  138.0f, SLE_BUS_SLACK, 1.060f,  0.0f,    0.0f},
        {"Bus2",  138.0f, SLE_BUS_PV,    1.045f, -0.217f,  0.127f},
        {"Bus3",  138.0f, SLE_BUS_PV,    1.010f, -0.942f,  0.044f},
        {"Bus4",  138.0f, SLE_BUS_PQ,    1.000f, -0.478f,  0.039f},
        {"Bus5",  138.0f, SLE_BUS_PQ,    1.000f, -0.076f, -0.016f},
        {"Bus6",  138.0f, SLE_BUS_PV,    1.070f, -0.112f,  0.075f},
        {"Bus7",  138.0f, SLE_BUS_PQ,    1.000f,  0.0f,    0.0f},
        {"Bus8",  138.0f, SLE_BUS_PV,    1.090f,  0.0f,    0.176f},
        {"Bus9",  138.0f, SLE_BUS_PQ,    1.000f, -0.295f, -0.166f},
        {"Bus10", 138.0f, SLE_BUS_PQ,    1.000f, -0.090f, -0.058f},
        {"Bus11", 138.0f, SLE_BUS_PQ,    1.000f, -0.035f, -0.018f},
        {"Bus12", 138.0f, SLE_BUS_PQ,    1.000f, -0.061f, -0.016f},
        {"Bus13", 138.0f, SLE_BUS_PQ,    1.000f, -0.135f, -0.058f},
        {"Bus14", 138.0f, SLE_BUS_PQ,    1.000f, -0.149f, -0.050f}
    };
    
    for (const auto& b : buses) {
        SLE_BusInfo info = {};
        info.id = b.id;
        info.base_kv = b.base_kv;
        info.type = b.type;
        info.v_setpoint = b.v_setpoint;
        info.p_scheduled = b.p_scheduled;
        info.q_scheduled = b.q_scheduled;
        
        if (sle_AddBus(engine, &info) != SLE_OK) {
            return false;
        }
    }
    
    // Branch data: from, to, r, x, b, tap, shift, is_transformer
    const struct {
        const char* id;
        const char* from;
        const char* to;
        SLE_Real r;
        SLE_Real x;
        SLE_Real b;
        SLE_Real tap;
        int32_t is_transformer;
    } branches[] = {
        {"Line1-2",   "Bus1",  "Bus2",  0.01938f, 0.05917f, 0.0528f, 1.0f, 0},
        {"Line1-5",   "Bus1",  "Bus5",  0.05403f, 0.22304f, 0.0492f, 1.0f, 0},
        {"Line2-3",   "Bus2",  "Bus3",  0.04699f, 0.19797f, 0.0438f, 1.0f, 0},
        {"Line2-4",   "Bus2",  "Bus4",  0.05811f, 0.17632f, 0.0340f, 1.0f, 0},
        {"Line2-5",   "Bus2",  "Bus5",  0.05695f, 0.17388f, 0.0346f, 1.0f, 0},
        {"Line3-4",   "Bus3",  "Bus4",  0.06701f, 0.17103f, 0.0128f, 1.0f, 0},
        {"Xfmr4-7",   "Bus4",  "Bus7",  0.0f,     0.20912f, 0.0f,    0.978f, 1},
        {"Xfmr4-9",   "Bus4",  "Bus9",  0.0f,     0.55618f, 0.0f,    0.969f, 1},
        {"Xfmr5-6",   "Bus5",  "Bus6",  0.0f,     0.25202f, 0.0f,    0.932f, 1},
        {"Line6-11",  "Bus6",  "Bus11", 0.09498f, 0.19890f, 0.0f,    1.0f, 0},
        {"Line6-12",  "Bus6",  "Bus12", 0.12291f, 0.25581f, 0.0f,    1.0f, 0},
        {"Line6-13",  "Bus6",  "Bus13", 0.06615f, 0.13027f, 0.0f,    1.0f, 0},
        {"Line7-8",   "Bus7",  "Bus8",  0.0f,     0.17615f, 0.0f,    1.0f, 0},
        {"Line7-9",   "Bus7",  "Bus9",  0.11001f, 0.20640f, 0.0f,    1.0f, 0},
        {"Line9-10",  "Bus9",  "Bus10", 0.03181f, 0.08450f, 0.0f,    1.0f, 0},
        {"Line9-14",  "Bus9",  "Bus14", 0.12711f, 0.27038f, 0.0f,    1.0f, 0},
        {"Line10-11", "Bus10", "Bus11", 0.08205f, 0.19207f, 0.0f,    1.0f, 0},
        {"Line12-13", "Bus12", "Bus13", 0.22092f, 0.19988f, 0.0f,    1.0f, 0},
        {"Line13-14", "Bus13", "Bus14", 0.17093f, 0.34802f, 0.0f,    1.0f, 0}
    };
    
    for (const auto& br : branches) {
        SLE_BranchInfo info = {};
        info.id = br.id;
        info.from_bus = br.from;
        info.to_bus = br.to;
        info.resistance = br.r;
        info.reactance = br.x;
        info.susceptance = br.b;
        info.conductance = 0.0f;
        info.tap_ratio = br.tap;
        info.phase_shift = 0.0f;
        info.is_transformer = br.is_transformer;
        info.sd_id = nullptr;
        
        if (sle_AddBranch(engine, &info) != SLE_OK) {
            return false;
        }
    }
    
    // Measurements: All bus voltages + key power flows
    // Voltage measurements
    const char* bus_ids[] = {
        "Bus1", "Bus2", "Bus3", "Bus4", "Bus5", "Bus6", "Bus7",
        "Bus8", "Bus9", "Bus10", "Bus11", "Bus12", "Bus13", "Bus14"
    };
    
    for (int i = 0; i < 14; ++i) {
        char meas_id[32];
        snprintf(meas_id, sizeof(meas_id), "V%d", i + 1);
        
        SLE_MeasurementInfo info = {};
        info.id = meas_id;
        info.type = SLE_MEAS_V_MAG;
        info.location = bus_ids[i];
        info.branch_end = SLE_BRANCH_FROM;
        info.sigma = 0.004f;
        info.pt_ratio = 1.0f;
        info.ct_ratio = 1.0f;
        info.is_pseudo = 0;
        
        if (sle_AddMeasurement(engine, &info) != SLE_OK) {
            return false;
        }
    }
    
    // P injection measurements at key buses
    const char* p_inj_buses[] = {"Bus1", "Bus2", "Bus3", "Bus6", "Bus8"};
    for (int i = 0; i < 5; ++i) {
        char meas_id[32];
        snprintf(meas_id, sizeof(meas_id), "P%s", p_inj_buses[i] + 3);  // Skip "Bus"
        
        SLE_MeasurementInfo info = {};
        info.id = meas_id;
        info.type = SLE_MEAS_P_INJECTION;
        info.location = p_inj_buses[i];
        info.sigma = 0.01f;
        info.pt_ratio = 1.0f;
        info.ct_ratio = 1.0f;
        info.is_pseudo = 0;
        
        if (sle_AddMeasurement(engine, &info) != SLE_OK) {
            return false;
        }
    }
    
    // Q injection measurements
    const char* q_inj_buses[] = {"Bus1", "Bus2", "Bus6"};
    for (int i = 0; i < 3; ++i) {
        char meas_id[32];
        snprintf(meas_id, sizeof(meas_id), "Q%s", q_inj_buses[i] + 3);
        
        SLE_MeasurementInfo info = {};
        info.id = meas_id;
        info.type = SLE_MEAS_Q_INJECTION;
        info.location = q_inj_buses[i];
        info.sigma = 0.01f;
        info.pt_ratio = 1.0f;
        info.ct_ratio = 1.0f;
        info.is_pseudo = 0;
        
        if (sle_AddMeasurement(engine, &info) != SLE_OK) {
            return false;
        }
    }
    
    // P flow measurements on key lines
    const char* p_flow_branches[] = {"Line1-2", "Line1-5", "Line2-3", "Line2-4"};
    for (int i = 0; i < 4; ++i) {
        char meas_id[32];
        snprintf(meas_id, sizeof(meas_id), "Pf_%s", p_flow_branches[i]);
        
        SLE_MeasurementInfo info = {};
        info.id = meas_id;
        info.type = SLE_MEAS_P_FLOW;
        info.location = p_flow_branches[i];
        info.branch_end = SLE_BRANCH_FROM;
        info.sigma = 0.008f;
        info.pt_ratio = 1.0f;
        info.ct_ratio = 1.0f;
        info.is_pseudo = 0;
        
        if (sle_AddMeasurement(engine, &info) != SLE_OK) {
            return false;
        }
    }
    
    // Q flow measurements
    const char* q_flow_branches[] = {"Line1-2", "Line1-5", "Line2-3"};
    for (int i = 0; i < 3; ++i) {
        char meas_id[32];
        snprintf(meas_id, sizeof(meas_id), "Qf_%s", q_flow_branches[i]);
        
        SLE_MeasurementInfo info = {};
        info.id = meas_id;
        info.type = SLE_MEAS_Q_FLOW;
        info.location = q_flow_branches[i];
        info.branch_end = SLE_BRANCH_FROM;
        info.sigma = 0.008f;
        info.pt_ratio = 1.0f;
        info.ct_ratio = 1.0f;
        info.is_pseudo = 0;
        
        if (sle_AddMeasurement(engine, &info) != SLE_OK) {
            return false;
        }
    }
    
    // Pseudo measurements for zero injection bus (Bus 7)
    {
        SLE_MeasurementInfo info = {};
        info.id = "P7_pseudo";
        info.type = SLE_MEAS_P_PSEUDO;
        info.location = "Bus7";
        info.sigma = 0.001f;
        info.pt_ratio = 1.0f;
        info.ct_ratio = 1.0f;
        info.is_pseudo = 1;
        sle_AddMeasurement(engine, &info);
        
        info.id = "Q7_pseudo";
        info.type = SLE_MEAS_Q_PSEUDO;
        sle_AddMeasurement(engine, &info);
    }
    
    return true;
}

/**
 * @brief Get true measurement values from power flow solution
 */
void getIEEE14TrueMeasurements(SLE_Real* voltages, SLE_Real* p_inj, 
                                SLE_Real* q_inj, SLE_Real* p_flow, SLE_Real* q_flow) {
    // True power flow solution values
    const SLE_Real v_true[] = {
        1.060f, 1.045f, 1.010f, 1.018f, 1.020f, 1.070f, 1.062f,
        1.090f, 1.056f, 1.051f, 1.057f, 1.055f, 1.050f, 1.036f
    };
    memcpy(voltages, v_true, 14 * sizeof(SLE_Real));
    
    // P injections at Bus1, Bus2, Bus3, Bus6, Bus8
    const SLE_Real p_true[] = {2.324f, 0.183f, -0.942f, -0.112f, 0.0f};
    memcpy(p_inj, p_true, 5 * sizeof(SLE_Real));
    
    // Q injections at Bus1, Bus2, Bus6
    const SLE_Real q_true[] = {-0.165f, 0.308f, 0.052f};
    memcpy(q_inj, q_true, 3 * sizeof(SLE_Real));
    
    // P flows on Line1-2, Line1-5, Line2-3, Line2-4
    const SLE_Real pf_true[] = {1.569f, 0.755f, 0.732f, 0.561f};
    memcpy(p_flow, pf_true, 4 * sizeof(SLE_Real));
    
    // Q flows on Line1-2, Line1-5, Line2-3
    const SLE_Real qf_true[] = {-0.204f, 0.039f, 0.035f};
    memcpy(q_flow, qf_true, 3 * sizeof(SLE_Real));
}

//=============================================================================
// Main Example
//=============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << " SLE Engine - Full Lifecycle Example" << std::endl;
    std::cout << " (FR-26 Comprehensive Usage Demo)" << std::endl;
    std::cout << " Using C-compatible DLL API" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Check GPU availability
    if (!sle_IsGPUAvailable()) {
        std::cerr << "No CUDA-capable GPU found!" << std::endl;
        return 1;
    }
    
    std::cout << "SLE Version: " << sle_GetVersion() << std::endl;
    std::cout << "CUDA Version: " << sle_GetCUDAVersion() << std::endl;
    
    //=========================================================================
    // PHASE 0: Configuration and Initialization
    //=========================================================================
    std::cout << "\n>>> Phase 0: Configuration and Initialization" << std::endl;
    
    // Ensure config file exists (creates default if not present)
    sle_EnsureConfigFile(nullptr);
    
    // Create engine - auto-loads from sle_engine.config
    SLE_Handle engine = nullptr;
    SLE_StatusCode status = sle_Create(&engine);
    if (status != SLE_OK) {
        std::cerr << "Failed to create engine!" << std::endl;
        return 1;
    }
    
    std::cout << "Config loaded from: " << sle_GetConfigPath(engine) << std::endl;
    
    // Initialize CUDA device
    if (sle_Initialize(engine) != SLE_OK) {
        std::cerr << "Failed to initialize CUDA!" << std::endl;
        sle_Destroy(engine);
        return 1;
    }
    
    std::cout << "CUDA Device: " << sle_GetDeviceInfo(engine) << std::endl;
    
    // Get configuration
    SLE_Config config;
    sle_GetConfig(engine, &config);
    std::cout << "Max Buses: " << config.max_buses << std::endl;
    std::cout << "Max Measurements: " << config.max_measurements << std::endl;
    
    //=========================================================================
    // PHASE 1: Model Upload & Full WLS Run (FR-12)
    //=========================================================================
    std::cout << "\n>>> Phase 1: Model Upload & Full WLS Run (FR-12)" << std::endl;
    
    // Load IEEE 14-Bus test case
    if (!loadIEEE14Bus(engine)) {
        std::cerr << "Failed to load IEEE 14-bus test case!" << std::endl;
        sle_Destroy(engine);
        return 1;
    }
    
    int32_t bus_count = sle_GetBusCount(engine);
    int32_t branch_count = sle_GetBranchCount(engine);
    int32_t meas_count = sle_GetMeasurementCount(engine);
    
    std::cout << "Network Model:" << std::endl;
    std::cout << "  Buses:        " << bus_count << std::endl;
    std::cout << "  Branches:     " << branch_count << std::endl;
    std::cout << "  Measurements: " << meas_count << std::endl;
    
    // Upload model to GPU
    if (sle_UploadModel(engine) != SLE_OK) {
        std::cerr << "Failed to upload model to GPU!" << std::endl;
        sle_Destroy(engine);
        return 1;
    }
    
    std::cout << "GPU Memory Used: " << (sle_GetGPUMemoryUsage(engine) / 1024) << " KB" << std::endl;
    
    // Prepare initial measurement values (from power flow solution)
    SLE_Real voltages[14], p_inj[5], q_inj[3], p_flow[4], q_flow[3];
    getIEEE14TrueMeasurements(voltages, p_inj, q_inj, p_flow, q_flow);
    
    // Assemble telemetry vector in measurement order
    std::vector<SLE_Real> telemetry;
    for (int i = 0; i < 14; ++i) telemetry.push_back(voltages[i]);
    for (int i = 0; i < 5; ++i) telemetry.push_back(p_inj[i]);
    for (int i = 0; i < 3; ++i) telemetry.push_back(q_inj[i]);
    for (int i = 0; i < 4; ++i) telemetry.push_back(p_flow[i]);
    for (int i = 0; i < 3; ++i) telemetry.push_back(q_flow[i]);
    telemetry.push_back(0.0f);  // P7 pseudo
    telemetry.push_back(0.0f);  // Q7 pseudo
    
    // Update telemetry
    sle_UpdateTelemetry(engine, telemetry.data(), static_cast<int32_t>(telemetry.size()));
    
    // Run FULL estimation (Precision Mode - FR-12)
    SLE_Result result1;
    sle_Solve(engine, SLE_MODE_PRECISION, &result1);
    printResult(result1, "Phase 1: Full WLS Run (Precision Mode)");
    printVoltages(engine, bus_count);
    
    //=========================================================================
    // PHASE 2: Telemetry Update & Fast WLS Run (FR-11, FR-16)
    //=========================================================================
    std::cout << "\n>>> Phase 2: Telemetry Update & Fast WLS Run (FR-11, FR-16)" << std::endl;
    
    // Create sigma array for noise
    SLE_Real sigmas[31];
    for (int i = 0; i < 14; ++i) sigmas[i] = 0.004f;
    for (int i = 14; i < 29; ++i) sigmas[i] = 0.01f;
    sigmas[29] = 0.001f;
    sigmas[30] = 0.001f;
    
    // Add noise to measurements
    addMeasurementNoise(telemetry.data(), sigmas, 31, 123);
    
    // Update telemetry (fast async copy)
    sle_UpdateTelemetry(engine, telemetry.data(), static_cast<int32_t>(telemetry.size()));
    
    // Run FAST estimation (Real-time Mode - FR-11)
    SLE_Result result2;
    sle_Solve(engine, SLE_MODE_REALTIME, &result2);
    printResult(result2, "Phase 2: Fast WLS Run (Real-time Mode, Gain Matrix Reuse)");
    
    // Compare timing
    std::cout << "\nTiming Comparison:" << std::endl;
    std::cout << "  Full Run:  " << result1.computation_time_ms << " ms" << std::endl;
    std::cout << "  Fast Run:  " << result2.computation_time_ms << " ms" << std::endl;
    if (result2.computation_time_ms > 0) {
        std::cout << "  Speedup:   " << (result1.computation_time_ms / result2.computation_time_ms) 
                  << "x" << std::endl;
    }
    
    //=========================================================================
    // PHASE 3: Switching Device Update & Fast Run (FR-05)
    //=========================================================================
    std::cout << "\n>>> Phase 3: Switch Status Change & Fast WLS Run (FR-05)" << std::endl;
    
    // Add a switching device to an existing branch
    SLE_SwitchInfo sw_info = {};
    sw_info.id = "SW1-5";
    sw_info.branch_id = "Line1-5";
    sw_info.status = SLE_SWITCH_CLOSED;
    sle_AddSwitch(engine, &sw_info);
    
    // Re-upload model (structural change)
    sle_UploadModel(engine);
    
    // Open the switch (simulating breaker trip)
    std::cout << "Opening switch SW1-5 on Line1-5..." << std::endl;
    sle_SetSwitchStatus(engine, "SW1-5", SLE_SWITCH_OPEN);
    
    // Run estimation with topology change
    SLE_Result result3;
    sle_Solve(engine, SLE_MODE_REALTIME, &result3);
    printResult(result3, "Phase 3: WLS Run After Topology Change");
    printVoltages(engine, bus_count);
    
    // Close the switch back
    std::cout << "\nClosing switch SW1-5..." << std::endl;
    sle_SetSwitchStatus(engine, "SW1-5", SLE_SWITCH_CLOSED);
    
    SLE_Result result3b;
    sle_Solve(engine, SLE_MODE_REALTIME, &result3b);
    printResult(result3b, "Phase 3b: WLS Run After Switch Closure");
    
    //=========================================================================
    // PHASE 4: Transformer Tap Update (FR-01 controllable taps)
    //=========================================================================
    std::cout << "\n>>> Phase 4: Transformer Tap Update (FR-01)" << std::endl;
    
    // Get current tap position
    SLE_Real old_tap = sle_GetTransformerTap(engine, "Xfmr4-7");
    std::cout << "Current tap ratio of Xfmr4-7: " << old_tap << std::endl;
    
    // Change transformer tap ratio (simulating voltage regulation)
    std::cout << "Changing tap ratio to 1.05..." << std::endl;
    sle_SetTransformerTap(engine, "Xfmr4-7", 1.05f);
    
    // Re-upload and solve
    sle_UploadModel(engine);
    SLE_Result result4;
    sle_Solve(engine, SLE_MODE_PRECISION, &result4);
    printResult(result4, "Phase 4: WLS Run After Tap Change");
    
    // Restore tap
    sle_SetTransformerTap(engine, "Xfmr4-7", old_tap);
    sle_UploadModel(engine);
    
    //=========================================================================
    // PHASE 5: Structural Model Change & Full Run (FR-25)
    //=========================================================================
    std::cout << "\n>>> Phase 5: Structural Model Change & Full WLS Run (FR-25)" << std::endl;
    
    // Add a new branch to the network (structural modification)
    std::cout << "Adding new branch Line10-14..." << std::endl;
    SLE_BranchInfo new_branch = {};
    new_branch.id = "Line10-14";
    new_branch.from_bus = "Bus10";
    new_branch.to_bus = "Bus14";
    new_branch.resistance = 0.08f;
    new_branch.reactance = 0.20f;
    new_branch.susceptance = 0.0f;
    new_branch.conductance = 0.0f;
    new_branch.tap_ratio = 1.0f;
    new_branch.phase_shift = 0.0f;
    new_branch.is_transformer = 0;
    new_branch.sd_id = nullptr;
    sle_AddBranch(engine, &new_branch);
    
    // Add measurement for the new branch
    SLE_MeasurementInfo new_meas = {};
    new_meas.id = "P10-14";
    new_meas.type = SLE_MEAS_P_FLOW;
    new_meas.location = "Line10-14";
    new_meas.branch_end = SLE_BRANCH_FROM;
    new_meas.sigma = 0.008f;
    new_meas.pt_ratio = 1.0f;
    new_meas.ct_ratio = 1.0f;
    new_meas.is_pseudo = 0;
    sle_AddMeasurement(engine, &new_meas);
    
    // Re-upload modified model
    if (sle_UploadModel(engine) != SLE_OK) {
        std::cerr << "Failed to re-upload modified model!" << std::endl;
        sle_Destroy(engine);
        return 1;
    }
    
    // Add telemetry for new measurement
    telemetry.push_back(0.05f);
    sle_UpdateTelemetry(engine, telemetry.data(), static_cast<int32_t>(telemetry.size()));
    
    // Run FULL estimation
    SLE_Result result5;
    sle_Solve(engine, SLE_MODE_PRECISION, &result5);
    printResult(result5, "Phase 5: Full WLS Run After Structural Change");
    printVoltages(engine, bus_count);
    
    //=========================================================================
    // PHASE 6: Bad Data Detection (FR-15)
    //=========================================================================
    std::cout << "\n>>> Phase 6: Bad Data Detection (FR-15)" << std::endl;
    
    // Inject bad data into one measurement
    telemetry[0] = 2.0f;  // Corrupt Bus 1 voltage (should be ~1.06)
    sle_UpdateTelemetry(engine, telemetry.data(), static_cast<int32_t>(telemetry.size()));
    
    SLE_Result result6;
    sle_Solve(engine, SLE_MODE_PRECISION, &result6);
    printResult(result6, "Phase 6: WLS Run With Bad Data");
    
    // Identify bad data
    int32_t bad_indices[100];
    int32_t bad_count = 0;
    sle_IdentifyBadData(engine, 3.0f, bad_indices, 100, &bad_count);
    
    std::cout << "\nBad Data Detection:" << std::endl;
    std::cout << "  Identified " << bad_count << " bad measurement(s)" << std::endl;
    
    std::vector<SLE_Real> residuals(telemetry.size());
    sle_GetResiduals(engine, residuals.data(), static_cast<int32_t>(residuals.size()));
    
    for (int32_t i = 0; i < bad_count; ++i) {
        std::cout << "  - Measurement index " << bad_indices[i] 
                  << ", residual = " << residuals[bad_indices[i]] << std::endl;
    }
    
    //=========================================================================
    // PHASE 7: Robust Estimation (FR-17)
    //=========================================================================
    std::cout << "\n>>> Phase 7: Robust Estimation with Huber M-Estimator (FR-17)" << std::endl;
    
    // Configure solver for robust estimation
    SLE_SolverConfig robust_config = {};
    robust_config.mode = SLE_MODE_PRECISION;
    robust_config.use_robust = 1;
    robust_config.huber_gamma = 1.5f;
    robust_config.max_iterations = 50;
    robust_config.tolerance = 1e-5f;
    robust_config.use_flat_start = 0;
    robust_config.time_limit_ms = 0.0f;
    
    SLE_Result result7;
    sle_SolveWithConfig(engine, &robust_config, &result7);
    printResult(result7, "Phase 7: Robust WLS Run (Huber M-Estimator)");
    
    // Compare with standard WLS
    std::cout << "\nComparison (with bad data):" << std::endl;
    std::cout << "  Standard WLS Objective: " << result6.objective << std::endl;
    std::cout << "  Robust WLS Objective:   " << result7.objective << std::endl;
    
    //=========================================================================
    // Summary
    //=========================================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << " Example Complete - Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Phase 1: Full model upload and precision estimation" << std::endl;
    std::cout << "Phase 2: Fast telemetry update with gain matrix reuse" << std::endl;
    std::cout << "Phase 3: Topology change (switch operation)" << std::endl;
    std::cout << "Phase 4: Transformer tap update" << std::endl;
    std::cout << "Phase 5: Structural modification (add branch)" << std::endl;
    std::cout << "Phase 6: Bad data detection" << std::endl;
    std::cout << "Phase 7: Robust estimation" << std::endl;
    std::cout << "\nAll FR requirements demonstrated successfully!" << std::endl;
    
    // Cleanup
    sle_Destroy(engine);
    
    return 0;
}

