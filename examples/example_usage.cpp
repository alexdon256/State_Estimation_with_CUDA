/**
 * @file example_usage.cpp
 * @brief Comprehensive SLE Engine usage example (FR-26) - METER-CENTRIC
 * 
 * Demonstrates the full operational lifecycle of the SLE Engine using the
 * C-compatible DLL API with METER-BASED telemetry. All measurements come
 * through physical metering devices (voltmeters, multimeters, wattmeters).
 * 
 * METER-CENTRIC DESIGN:
 * - Voltmeters on buses provide voltage measurements
 * - Multimeters on branches provide P, Q, V, I measurements
 * - Wattmeters on branches provide P, Q measurements
 * - All telemetry updates go through sle_UpdateMeterReading()
 * - Estimated values retrieved via sle_GetMeterEstimate()
 * 
 * Demonstrates:
 * 1. Model Upload with Meters (FR-12)
 * 2. Meter Telemetry Update & Fast Run (FR-11, FR-16)
 * 3. SD Update & Fast Run (FR-05)
 * 4. Structural Update & Full Run (FR-25)
 * 5. Transformer Tap Update (FR-01 controllable taps)
 * 6. Bad Data Detection via Meters (FR-15)
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
 * @brief Load IEEE 14-Bus test case data with METER-BASED measurements
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
    
    //=========================================================================
    // METER-BASED MEASUREMENT SETUP
    // Meters are the PRIMARY source of telemetry - they own the measurements
    //=========================================================================
    
    // Bus IDs for reference
    const char* bus_ids[] = {
        "Bus1", "Bus2", "Bus3", "Bus4", "Bus5", "Bus6", "Bus7",
        "Bus8", "Bus9", "Bus10", "Bus11", "Bus12", "Bus13", "Bus14"
    };
    
    //-------------------------------------------------------------------------
    // 1. VOLTMETERS on all buses - for voltage measurements
    //    Each voltmeter creates a V_MAG measurement automatically
    //-------------------------------------------------------------------------
    for (int i = 0; i < 14; ++i) {
        char meter_id[32];
        snprintf(meter_id, sizeof(meter_id), "VM%d", i + 1);
        
        SLE_MeterInfo meter = {};
        meter.id = meter_id;
        meter.type = SLE_METER_VOLTMETER;
        meter.bus_id = bus_ids[i];
        meter.branch_id = nullptr;
        meter.branch_end = SLE_BRANCH_FROM;
        meter.pt_ratio = 1.0f;
        meter.ct_ratio = 1.0f;
        meter.sigma_v = 0.004f;
        meter.sigma_p = 0.01f;
        meter.sigma_i = 0.01f;
        
        if (sle_AddMeter(engine, &meter) != SLE_OK) {
            return false;
        }
    }
    
    //-------------------------------------------------------------------------
    // 2. MULTIMETERS on key branches - for P, Q, I, V measurements
    //    Each multimeter creates P_FLOW, Q_FLOW, I_MAG, V_MAG measurements
    //-------------------------------------------------------------------------
    const struct {
        const char* meter_id;
        const char* bus_id;     // Bus for voltage reference
        const char* branch_id;  // Branch for power/current
    } multimeters[] = {
        {"MM1-2",  "Bus1", "Line1-2"},
        {"MM1-5",  "Bus1", "Line1-5"},
        {"MM2-3",  "Bus2", "Line2-3"},
        {"MM2-4",  "Bus2", "Line2-4"}
    };
    
    for (const auto& mm : multimeters) {
        SLE_MeterInfo meter = {};
        meter.id = mm.meter_id;
        meter.type = SLE_METER_MULTIMETER;
        meter.bus_id = mm.bus_id;
        meter.branch_id = mm.branch_id;
        meter.branch_end = SLE_BRANCH_FROM;
        meter.pt_ratio = 1.0f;
        meter.ct_ratio = 1.0f;
        meter.sigma_v = 0.004f;
        meter.sigma_p = 0.008f;
        meter.sigma_i = 0.01f;
        
        if (sle_AddMeter(engine, &meter) != SLE_OK) {
            return false;
        }
    }
    
    //-------------------------------------------------------------------------
    // 3. WATTMETERS on additional branches - for P, Q measurements only
    //-------------------------------------------------------------------------
    const struct {
        const char* meter_id;
        const char* bus_id;
        const char* branch_id;
    } wattmeters[] = {
        {"WM2-5",  "Bus2", "Line2-5"},
        {"WM3-4",  "Bus3", "Line3-4"},
        {"WM6-11", "Bus6", "Line6-11"},
        {"WM6-12", "Bus6", "Line6-12"},
        {"WM6-13", "Bus6", "Line6-13"}
    };
    
    for (const auto& wm : wattmeters) {
        SLE_MeterInfo meter = {};
        meter.id = wm.meter_id;
        meter.type = SLE_METER_WATTMETER;
        meter.bus_id = wm.bus_id;
        meter.branch_id = wm.branch_id;
        meter.branch_end = SLE_BRANCH_FROM;
        meter.pt_ratio = 1.0f;
        meter.ct_ratio = 1.0f;
        meter.sigma_v = 0.004f;
        meter.sigma_p = 0.01f;
        meter.sigma_i = 0.01f;
        
        if (sle_AddMeter(engine, &meter) != SLE_OK) {
            return false;
        }
    }
    
    //-------------------------------------------------------------------------
    // 4. Pseudo measurements for zero injection buses (no physical meter)
    //    These are structural constraints, not actual meters
    //-------------------------------------------------------------------------
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
 * @brief Initialize all meter readings with true power flow values
 * 
 * This demonstrates meter-centric telemetry:
 * - Update voltmeters on buses
 * - Update multimeters on branches (P, Q, V, I)
 * - Update wattmeters on branches (P, Q)
 */
void initializeMeterReadings(SLE_Handle engine) {
    // True power flow solution voltages
    const SLE_Real v_true[] = {
        1.060f, 1.045f, 1.010f, 1.018f, 1.020f, 1.070f, 1.062f,
        1.090f, 1.056f, 1.051f, 1.057f, 1.055f, 1.050f, 1.036f
    };
    
    // True power flows [p.u.] on branches with multimeters
    // Line1-2, Line1-5, Line2-3, Line2-4
    const SLE_Real pf_true[] = {1.569f, 0.755f, 0.732f, 0.561f};
    const SLE_Real qf_true[] = {-0.204f, 0.039f, 0.035f, 0.028f};
    
    // True power flows on branches with wattmeters  
    // Line2-5, Line3-4, Line6-11, Line6-12, Line6-13
    const SLE_Real wm_p[] = {0.415f, -0.233f, 0.073f, 0.078f, 0.177f};
    const SLE_Real wm_q[] = {0.012f, 0.045f, 0.035f, 0.025f, 0.072f};
    
    //-------------------------------------------------------------------------
    // 1. Update VOLTMETERS (VM1-VM14) with bus voltage readings
    //-------------------------------------------------------------------------
    std::cout << "  Setting voltmeter readings..." << std::endl;
    for (int i = 0; i < 14; ++i) {
        char meter_id[32];
        snprintf(meter_id, sizeof(meter_id), "VM%d", i + 1);
        sle_UpdateMeterReading(engine, meter_id, "V", v_true[i]);
    }
    
    //-------------------------------------------------------------------------
    // 2. Update MULTIMETERS with P, Q readings
    //-------------------------------------------------------------------------
    std::cout << "  Setting multimeter readings..." << std::endl;
    const char* mm_ids[] = {"MM1-2", "MM1-5", "MM2-3", "MM2-4"};
    for (int i = 0; i < 4; ++i) {
        sle_UpdateMeterReading(engine, mm_ids[i], "kW", pf_true[i]);
        sle_UpdateMeterReading(engine, mm_ids[i], "kVAR", qf_true[i]);
    }
    
    //-------------------------------------------------------------------------
    // 3. Update WATTMETERS with P, Q readings
    //-------------------------------------------------------------------------
    std::cout << "  Setting wattmeter readings..." << std::endl;
    const char* wm_ids[] = {"WM2-5", "WM3-4", "WM6-11", "WM6-12", "WM6-13"};
    for (int i = 0; i < 5; ++i) {
        sle_UpdateMeterReading(engine, wm_ids[i], "kW", wm_p[i]);
        sle_UpdateMeterReading(engine, wm_ids[i], "kVAR", wm_q[i]);
    }
}

/**
 * @brief Add measurement noise to meter readings
 */
void addMeterNoise(SLE_Handle engine, unsigned int seed) {
    srand(seed);
    
    auto gaussNoise = [](float sigma) -> float {
        float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        return sigma * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI * u2);
    };
    
    // Add noise to voltmeters (sigma = 0.004)
    for (int i = 0; i < 14; ++i) {
        char meter_id[32];
        snprintf(meter_id, sizeof(meter_id), "VM%d", i + 1);
        
        SLE_Real current_val;
        sle_GetMeterReading(engine, meter_id, "V", &current_val);
        sle_UpdateMeterReading(engine, meter_id, "V", current_val + gaussNoise(0.004f));
    }
    
    // Add noise to multimeters (sigma_p = 0.008)
    const char* mm_ids[] = {"MM1-2", "MM1-5", "MM2-3", "MM2-4"};
    for (int i = 0; i < 4; ++i) {
        SLE_Real p_val, q_val;
        sle_GetMeterReading(engine, mm_ids[i], "kW", &p_val);
        sle_GetMeterReading(engine, mm_ids[i], "kVAR", &q_val);
        sle_UpdateMeterReading(engine, mm_ids[i], "kW", p_val + gaussNoise(0.008f));
        sle_UpdateMeterReading(engine, mm_ids[i], "kVAR", q_val + gaussNoise(0.008f));
    }
    
    // Add noise to wattmeters (sigma_p = 0.01)
    const char* wm_ids[] = {"WM2-5", "WM3-4", "WM6-11", "WM6-12", "WM6-13"};
    for (int i = 0; i < 5; ++i) {
        SLE_Real p_val, q_val;
        sle_GetMeterReading(engine, wm_ids[i], "kW", &p_val);
        sle_GetMeterReading(engine, wm_ids[i], "kVAR", &q_val);
        sle_UpdateMeterReading(engine, wm_ids[i], "kW", p_val + gaussNoise(0.01f));
        sle_UpdateMeterReading(engine, wm_ids[i], "kVAR", q_val + gaussNoise(0.01f));
    }
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
    
    // Load IEEE 14-Bus test case with METER-BASED measurements
    if (!loadIEEE14Bus(engine)) {
        std::cerr << "Failed to load IEEE 14-bus test case!" << std::endl;
        sle_Destroy(engine);
        return 1;
    }
    
    int32_t bus_count = sle_GetBusCount(engine);
    int32_t branch_count = sle_GetBranchCount(engine);
    int32_t meas_count = sle_GetMeasurementCount(engine);
    int32_t meter_count = sle_GetMeterCount(engine);
    
    std::cout << "Network Model (Meter-Centric):" << std::endl;
    std::cout << "  Buses:        " << bus_count << std::endl;
    std::cout << "  Branches:     " << branch_count << std::endl;
    std::cout << "  Meters:       " << meter_count << std::endl;
    std::cout << "  Measurements: " << meas_count << " (auto-created by meters)" << std::endl;
    
    // Upload model to GPU
    if (sle_UploadModel(engine) != SLE_OK) {
        std::cerr << "Failed to upload model to GPU!" << std::endl;
        sle_Destroy(engine);
        return 1;
    }
    
    std::cout << "GPU Memory Used: " << (sle_GetGPUMemoryUsage(engine) / 1024) << " KB" << std::endl;
    
    //-------------------------------------------------------------------------
    // METER-BASED TELEMETRY UPDATE (Phase 1)
    // All values come through meters - no raw measurement updates
    //-------------------------------------------------------------------------
    std::cout << "\nInitializing meter readings (true power flow values)..." << std::endl;
    initializeMeterReadings(engine);
    
    // Run FULL estimation (Precision Mode - FR-12)
    SLE_Result result1;
    sle_Solve(engine, SLE_MODE_PRECISION, &result1);
    printResult(result1, "Phase 1: Full WLS Run (Precision Mode)");
    printVoltages(engine, bus_count);
    
    // Show bus/branch estimates after solve
    std::cout << "\nEstimated State Values (after solve):" << std::endl;
    std::cout << "  Bus1 V: " << sle_GetBusVoltage(engine, "Bus1") << " p.u." << std::endl;
    std::cout << "  Bus1 angle: " << sle_GetBusAngle(engine, "Bus1") << " rad" << std::endl;
    std::cout << "  Bus2 V: " << sle_GetBusVoltage(engine, "Bus2") << " p.u." << std::endl;
    std::cout << "  Line1-2 P_from: " << sle_GetBranchP(engine, "Line1-2", SLE_BRANCH_FROM) << " p.u." << std::endl;
    std::cout << "  Line1-2 Q_from: " << sle_GetBranchQ(engine, "Line1-2", SLE_BRANCH_FROM) << " p.u." << std::endl;
    
    //=========================================================================
    // PHASE 2: Meter Telemetry Update & Fast WLS Run (FR-11, FR-16)
    //=========================================================================
    std::cout << "\n>>> Phase 2: Meter Update & Fast WLS Run (FR-11, FR-16)" << std::endl;
    
    std::cout << "\n=== METER-CENTRIC TELEMETRY ===" << std::endl;
    std::cout << "All updates go through physical meters:" << std::endl;
    std::cout << "  - VM1..VM14: Voltmeters on each bus" << std::endl;
    std::cout << "  - MM1-2, MM1-5, MM2-3, MM2-4: Multimeters on branches" << std::endl;
    std::cout << "  - WM2-5, WM3-4, etc.: Wattmeters on branches" << std::endl;
    
    // Update individual meter readings
    std::cout << "\nUpdating meter readings..." << std::endl;
    
    // Simulate voltage changes at key buses
    sle_UpdateMeterReading(engine, "VM1", "V", 1.058f);   // Bus 1 voltage
    sle_UpdateMeterReading(engine, "VM2", "V", 1.043f);   // Bus 2 voltage
    sle_UpdateMeterReading(engine, "VM3", "V", 1.008f);   // Bus 3 voltage
    
    // Simulate power flow changes on key branches
    sle_UpdateMeterReading(engine, "MM1-2", "kW", 1.55f);     // Line1-2 P
    sle_UpdateMeterReading(engine, "MM1-2", "kVAR", -0.21f);  // Line1-2 Q
    sle_UpdateMeterReading(engine, "MM1-5", "kW", 0.74f);     // Line1-5 P
    sle_UpdateMeterReading(engine, "MM1-5", "kVAR", 0.04f);   // Line1-5 Q
    
    // Batch meter updates using SLE_MeterReading
    std::cout << "Batch updating meter readings..." << std::endl;
    SLE_MeterReading batch_readings[] = {
        {"VM4", "V", 1.016f},
        {"VM5", "V", 1.018f},
        {"MM2-3", "kW", 0.73f},
        {"MM2-3", "kVAR", 0.03f},
        {"WM2-5", "kW", 0.41f},
        {"WM2-5", "kVAR", 0.01f}
    };
    sle_UpdateMeterReadings(engine, batch_readings, 6, 0);  // Don't sync yet
    
    // Add measurement noise to all meters
    std::cout << "Adding measurement noise to all meters..." << std::endl;
    addMeterNoise(engine, 123);
    
    // Run FAST estimation (Real-time Mode - FR-11)
    // Measurement values auto-sync to GPU during solve()
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
    
    // Show residuals using the convenient bus/branch residual getters
    std::cout << "\nResiduals via sle_GetBusResiduals / sle_GetBranchResiduals:" << std::endl;
    
    SLE_Real v_res, p_res, q_res, i_res;
    int32_t res_count;
    
    // Get residuals at Bus1
    if (sle_GetBusResiduals(engine, "Bus1", &v_res, &p_res, &q_res, &res_count) == SLE_OK) {
        std::cout << "  Bus1: V_resid=" << v_res;
        if (!std::isnan(p_res)) std::cout << ", P_resid=" << p_res;
        if (!std::isnan(q_res)) std::cout << ", Q_resid=" << q_res;
        std::cout << " (" << res_count << " measurements)" << std::endl;
    }
    
    // Get residuals at Bus2
    if (sle_GetBusResiduals(engine, "Bus2", &v_res, &p_res, &q_res, &res_count) == SLE_OK) {
        std::cout << "  Bus2: V_resid=" << v_res;
        if (!std::isnan(p_res)) std::cout << ", P_resid=" << p_res;
        if (!std::isnan(q_res)) std::cout << ", Q_resid=" << q_res;
        std::cout << " (" << res_count << " measurements)" << std::endl;
    }
    
    // Get residuals at Line1-2 (from end)
    if (sle_GetBranchResiduals(engine, "Line1-2", SLE_BRANCH_FROM, &p_res, &q_res, &i_res, &res_count) == SLE_OK) {
        std::cout << "  Line1-2 FROM:";
        if (!std::isnan(p_res)) std::cout << " P_resid=" << p_res;
        if (!std::isnan(q_res)) std::cout << ", Q_resid=" << q_res;
        if (!std::isnan(i_res)) std::cout << ", I_resid=" << i_res;
        std::cout << " (" << res_count << " measurements)" << std::endl;
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
    
    // Re-initialize meter readings after model change
    initializeMeterReadings(engine);
    
    // Update the new measurement value via low-level API (no meter for it)
    sle_UpdateMeasurement(engine, "P10-14", 0.05f);
    
    // Run FULL estimation
    SLE_Result result5;
    sle_Solve(engine, SLE_MODE_PRECISION, &result5);
    printResult(result5, "Phase 5: Full WLS Run After Structural Change");
    printVoltages(engine, bus_count);
    
    //=========================================================================
    // PHASE 6: Bad Data Detection (FR-15)
    //=========================================================================
    std::cout << "\n>>> Phase 6: Bad Data Detection (FR-15)" << std::endl;
    
    // Inject bad data via a meter (corrupt Bus 1 voltage)
    std::cout << "Injecting bad data: VM1 voltage = 2.0 p.u. (should be ~1.06)" << std::endl;
    sle_UpdateMeterReading(engine, "VM1", "V", 2.0f);
    
    SLE_Result result6;
    sle_Solve(engine, SLE_MODE_PRECISION, &result6);
    printResult(result6, "Phase 6: WLS Run With Bad Data");
    
    // Identify bad data
    int32_t bad_indices[100];
    int32_t bad_count_detected = 0;
    sle_IdentifyBadData(engine, 3.0f, bad_indices, 100, &bad_count_detected);
    
    std::cout << "\nBad Data Detection:" << std::endl;
    std::cout << "  Identified " << bad_count_detected << " bad measurement(s)" << std::endl;
    
    std::vector<SLE_Real> residuals(meas_count);
    sle_GetResiduals(engine, residuals.data(), meas_count);
    
    for (int32_t i = 0; i < bad_count_detected; ++i) {
        std::cout << "  - Measurement index " << bad_indices[i] 
                  << ", residual = " << residuals[bad_indices[i]] << std::endl;
    }
    
    // Restore good value
    sle_UpdateMeterReading(engine, "VM1", "V", 1.06f);
    
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
    std::cout << "Phase 1: Model upload with METER-BASED measurements" << std::endl;
    std::cout << "Phase 2: Meter telemetry update with gain matrix reuse" << std::endl;
    std::cout << "Phase 3: Topology change (switch operation)" << std::endl;
    std::cout << "Phase 4: Transformer tap update" << std::endl;
    std::cout << "Phase 5: Structural modification (add branch)" << std::endl;
    std::cout << "Phase 6: Bad data detection via meter" << std::endl;
    std::cout << "Phase 7: Robust estimation" << std::endl;
    std::cout << "\n=== METER-CENTRIC DESIGN ===" << std::endl;
    std::cout << "All telemetry flows through METERS (physical metering devices):" << std::endl;
    std::cout << "  - Voltmeters (VM1-VM14): Voltage on each bus" << std::endl;
    std::cout << "  - Multimeters (MM1-2, etc.): P, Q, V, I on branches" << std::endl;
    std::cout << "  - Wattmeters (WM2-5, etc.): P, Q on branches" << std::endl;
    std::cout << "  - sle_UpdateMeterReading() -> updates underlying measurement" << std::endl;
    std::cout << "  - sle_GetMeterEstimate() -> reads estimated value after solve" << std::endl;
    std::cout << "\nAll FR requirements demonstrated successfully!" << std::endl;
    
    // Cleanup
    sle_Destroy(engine);
    
    return 0;
}

