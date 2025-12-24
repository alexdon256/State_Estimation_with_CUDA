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
 * @file ieee14bus.h
 * @brief IEEE 14-Bus Test Case Data (FR-18)
 * 
 * Standard IEEE 14-bus test system for state estimation validation.
 * Data source: IEEE Power Systems Test Case Archive
 * 
 * System characteristics:
 * - 14 buses
 * - 20 branches (17 lines + 3 transformers)
 * - 5 generators (buses 1, 2, 3, 6, 8)
 * - Bus 1 is the slack bus
 * 
 * All values are in per-unit on 100 MVA base.
 */

#ifndef IEEE14BUS_H
#define IEEE14BUS_H

#include "../include/sle_types.cuh"
#include "../include/network_model.h"
#include "../include/sle_engine.h"

namespace sle {
namespace ieee14 {

//=============================================================================
// Bus Data
//=============================================================================

/**
 * @brief Load IEEE 14-bus test case into engine
 * 
 * Populates the SLE engine with buses, branches, and measurements
 * for the standard IEEE 14-bus test system.
 * 
 * @param engine SLE Engine to populate
 * @return true if loaded successfully
 */
inline bool loadIEEE14Bus(SLEEngine& engine) {
    
    // =========================================================================
    // BUS DATA
    // Bus: ID, Base kV, Type, V_setpoint, P_sched, Q_sched
    // =========================================================================
    
    // Bus 1 - Slack bus (swing)
    (void)engine.addBus(BusDescriptor("Bus1", 69.0f, BusType::SLACK, 1.060f, 0.0f, 0.0f));
    
    // Bus 2 - PV bus (generator)
    (void)engine.addBus(BusDescriptor("Bus2", 69.0f, BusType::PV, 1.045f, 0.40f, 0.0f));
    
    // Bus 3 - PV bus (synchronous condenser)
    (void)engine.addBus(BusDescriptor("Bus3", 69.0f, BusType::PV, 1.010f, 0.0f, 0.0f));
    
    // Bus 4 - PQ bus (load)
    (void)engine.addBus(BusDescriptor("Bus4", 69.0f, BusType::PQ, 1.0f, -0.478f, -0.039f));
    
    // Bus 5 - PQ bus (load)
    (void)engine.addBus(BusDescriptor("Bus5", 69.0f, BusType::PQ, 1.0f, -0.076f, -0.016f));
    
    // Bus 6 - PV bus (generator)
    (void)engine.addBus(BusDescriptor("Bus6", 13.8f, BusType::PV, 1.070f, 0.0f, 0.0f));
    
    // Bus 7 - PQ bus (no load)
    (void)engine.addBus(BusDescriptor("Bus7", 13.8f, BusType::PQ, 1.0f, 0.0f, 0.0f));
    
    // Bus 8 - PV bus (generator)
    (void)engine.addBus(BusDescriptor("Bus8", 18.0f, BusType::PV, 1.090f, 0.0f, 0.0f));
    
    // Bus 9 - PQ bus (load)
    (void)engine.addBus(BusDescriptor("Bus9", 13.8f, BusType::PQ, 1.0f, -0.295f, -0.166f));
    
    // Bus 10 - PQ bus (load)
    (void)engine.addBus(BusDescriptor("Bus10", 13.8f, BusType::PQ, 1.0f, -0.090f, -0.058f));
    
    // Bus 11 - PQ bus (load)
    (void)engine.addBus(BusDescriptor("Bus11", 13.8f, BusType::PQ, 1.0f, -0.035f, -0.018f));
    
    // Bus 12 - PQ bus (load)
    (void)engine.addBus(BusDescriptor("Bus12", 13.8f, BusType::PQ, 1.0f, -0.061f, -0.016f));
    
    // Bus 13 - PQ bus (load)
    (void)engine.addBus(BusDescriptor("Bus13", 13.8f, BusType::PQ, 1.0f, -0.135f, -0.058f));
    
    // Bus 14 - PQ bus (load)
    (void)engine.addBus(BusDescriptor("Bus14", 13.8f, BusType::PQ, 1.0f, -0.149f, -0.050f));
    
    // =========================================================================
    // BRANCH DATA
    // Branch: ID, From, To, R, X, B, G, Tap, Shift, IsTransformer
    // All values in p.u. on 100 MVA base
    // =========================================================================
    
    // Transmission Lines
    (void)engine.addBranch(BranchDescriptor("Line1-2", "Bus1", "Bus2", 
        0.01938f, 0.05917f, 0.0528f, 0.0f, 1.0f, 0.0f, false));
    
    (void)engine.addBranch(BranchDescriptor("Line1-5", "Bus1", "Bus5",
        0.05403f, 0.22304f, 0.0492f, 0.0f, 1.0f, 0.0f, false));
    
    (void)engine.addBranch(BranchDescriptor("Line2-3", "Bus2", "Bus3",
        0.04699f, 0.19797f, 0.0438f, 0.0f, 1.0f, 0.0f, false));
    
    (void)engine.addBranch(BranchDescriptor("Line2-4", "Bus2", "Bus4",
        0.05811f, 0.17632f, 0.0340f, 0.0f, 1.0f, 0.0f, false));
    
    (void)engine.addBranch(BranchDescriptor("Line2-5", "Bus2", "Bus5",
        0.05695f, 0.17388f, 0.0346f, 0.0f, 1.0f, 0.0f, false));
    
    (void)engine.addBranch(BranchDescriptor("Line3-4", "Bus3", "Bus4",
        0.06701f, 0.17103f, 0.0128f, 0.0f, 1.0f, 0.0f, false));
    
    (void)engine.addBranch(BranchDescriptor("Line4-5", "Bus4", "Bus5",
        0.01335f, 0.04211f, 0.0f, 0.0f, 1.0f, 0.0f, false));
    
    (void)engine.addBranch(BranchDescriptor("Line6-11", "Bus6", "Bus11",
        0.09498f, 0.19890f, 0.0f, 0.0f, 1.0f, 0.0f, false));
    
    (void)engine.addBranch(BranchDescriptor("Line6-12", "Bus6", "Bus12",
        0.12291f, 0.25581f, 0.0f, 0.0f, 1.0f, 0.0f, false));
    
    (void)engine.addBranch(BranchDescriptor("Line6-13", "Bus6", "Bus13",
        0.06615f, 0.13027f, 0.0f, 0.0f, 1.0f, 0.0f, false));
    
    (void)engine.addBranch(BranchDescriptor("Line7-8", "Bus7", "Bus8",
        0.0f, 0.17615f, 0.0f, 0.0f, 1.0f, 0.0f, false));
    
    (void)engine.addBranch(BranchDescriptor("Line7-9", "Bus7", "Bus9",
        0.11001f, 0.20640f, 0.0f, 0.0f, 1.0f, 0.0f, false));
    
    (void)engine.addBranch(BranchDescriptor("Line9-10", "Bus9", "Bus10",
        0.03181f, 0.08450f, 0.0f, 0.0f, 1.0f, 0.0f, false));
    
    (void)engine.addBranch(BranchDescriptor("Line9-14", "Bus9", "Bus14",
        0.12711f, 0.27038f, 0.0f, 0.0f, 1.0f, 0.0f, false));
    
    (void)engine.addBranch(BranchDescriptor("Line10-11", "Bus10", "Bus11",
        0.08205f, 0.19207f, 0.0f, 0.0f, 1.0f, 0.0f, false));
    
    (void)engine.addBranch(BranchDescriptor("Line12-13", "Bus12", "Bus13",
        0.22092f, 0.19988f, 0.0f, 0.0f, 1.0f, 0.0f, false));
    
    (void)engine.addBranch(BranchDescriptor("Line13-14", "Bus13", "Bus14",
        0.17093f, 0.34802f, 0.0f, 0.0f, 1.0f, 0.0f, false));
    
    // Transformers (3 transformers in the system)
    // Transformer 4-7: 69/13.8 kV, tap = 0.978
    (void)engine.addBranch(BranchDescriptor("Xfmr4-7", "Bus4", "Bus7",
        0.0f, 0.20912f, 0.0f, 0.0f, 0.978f, 0.0f, true));
    
    // Transformer 4-9: 69/13.8 kV, tap = 0.969
    (void)engine.addBranch(BranchDescriptor("Xfmr4-9", "Bus4", "Bus9",
        0.0f, 0.55618f, 0.0f, 0.0f, 0.969f, 0.0f, true));
    
    // Transformer 5-6: 69/13.8 kV, tap = 0.932
    (void)engine.addBranch(BranchDescriptor("Xfmr5-6", "Bus5", "Bus6",
        0.0f, 0.25202f, 0.0f, 0.0f, 0.932f, 0.0f, true));
    
    // =========================================================================
    // MEASUREMENT DATA
    // Typical measurement configuration for state estimation
    // =========================================================================
    
    // Voltage magnitude measurements at all buses
    // Standard deviation: 0.004 p.u. for voltage measurements
    for (int i = 1; i <= 14; ++i) {
        std::string bus_id = "Bus" + std::to_string(i);
        std::string meas_id = "V" + std::to_string(i);
        (void)engine.addMeasurement(MeasurementDescriptor(
            meas_id, MeasurementType::V_MAG, bus_id,
            BranchEnd::FROM, 0.004f, 1.0f, 1.0f, false));
    }
    
    // Power injection measurements at generator buses and major loads
    // Standard deviation: 0.01 p.u. for power measurements
    
    // P injections at generators
    (void)engine.addMeasurement(MeasurementDescriptor(
        "P1", MeasurementType::P_INJECTION, "Bus1",
        BranchEnd::FROM, 0.01f, 1.0f, 1.0f, false));
    (void)engine.addMeasurement(MeasurementDescriptor(
        "P2", MeasurementType::P_INJECTION, "Bus2",
        BranchEnd::FROM, 0.01f, 1.0f, 1.0f, false));
    (void)engine.addMeasurement(MeasurementDescriptor(
        "P3", MeasurementType::P_INJECTION, "Bus3",
        BranchEnd::FROM, 0.01f, 1.0f, 1.0f, false));
    (void)engine.addMeasurement(MeasurementDescriptor(
        "P6", MeasurementType::P_INJECTION, "Bus6",
        BranchEnd::FROM, 0.01f, 1.0f, 1.0f, false));
    (void)engine.addMeasurement(MeasurementDescriptor(
        "P8", MeasurementType::P_INJECTION, "Bus8",
        BranchEnd::FROM, 0.01f, 1.0f, 1.0f, false));
    
    // Q injections at generators
    (void)engine.addMeasurement(MeasurementDescriptor(
        "Q1", MeasurementType::Q_INJECTION, "Bus1",
        BranchEnd::FROM, 0.01f, 1.0f, 1.0f, false));
    (void)engine.addMeasurement(MeasurementDescriptor(
        "Q2", MeasurementType::Q_INJECTION, "Bus2",
        BranchEnd::FROM, 0.01f, 1.0f, 1.0f, false));
    (void)engine.addMeasurement(MeasurementDescriptor(
        "Q3", MeasurementType::Q_INJECTION, "Bus3",
        BranchEnd::FROM, 0.01f, 1.0f, 1.0f, false));
    
    // Power flow measurements on key branches
    (void)engine.addMeasurement(MeasurementDescriptor(
        "P1-2", MeasurementType::P_FLOW, "Line1-2",
        BranchEnd::FROM, 0.008f, 1.0f, 1.0f, false));
    (void)engine.addMeasurement(MeasurementDescriptor(
        "Q1-2", MeasurementType::Q_FLOW, "Line1-2",
        BranchEnd::FROM, 0.008f, 1.0f, 1.0f, false));
    
    (void)engine.addMeasurement(MeasurementDescriptor(
        "P1-5", MeasurementType::P_FLOW, "Line1-5",
        BranchEnd::FROM, 0.008f, 1.0f, 1.0f, false));
    (void)engine.addMeasurement(MeasurementDescriptor(
        "Q1-5", MeasurementType::Q_FLOW, "Line1-5",
        BranchEnd::FROM, 0.008f, 1.0f, 1.0f, false));
    
    (void)engine.addMeasurement(MeasurementDescriptor(
        "P2-3", MeasurementType::P_FLOW, "Line2-3",
        BranchEnd::FROM, 0.008f, 1.0f, 1.0f, false));
    
    (void)engine.addMeasurement(MeasurementDescriptor(
        "P4-7", MeasurementType::P_FLOW, "Xfmr4-7",
        BranchEnd::FROM, 0.008f, 1.0f, 1.0f, false));
    (void)engine.addMeasurement(MeasurementDescriptor(
        "Q4-7", MeasurementType::Q_FLOW, "Xfmr4-7",
        BranchEnd::FROM, 0.008f, 1.0f, 1.0f, false));
    
    // Zero injection pseudo-measurements for buses with no load or generation
    // (Bus 7 has no load - use zero injection constraint)
    (void)engine.addMeasurement(MeasurementDescriptor(
        "P7_pseudo", MeasurementType::P_PSEUDO, "Bus7",
        BranchEnd::FROM, 0.001f, 1.0f, 1.0f, true));
    (void)engine.addMeasurement(MeasurementDescriptor(
        "Q7_pseudo", MeasurementType::Q_PSEUDO, "Bus7",
        BranchEnd::FROM, 0.001f, 1.0f, 1.0f, true));
    
    return true;
}

/**
 * @brief Get "true" measurement values for IEEE 14-bus (from power flow solution)
 * 
 * These are the values that would be measured for the IEEE 14-bus
 * system at the power flow solution. Use these as initial telemetry.
 * 
 * @param voltages Output voltage measurements [14]
 * @param p_inj Output P injection measurements [5] - buses 1,2,3,6,8
 * @param q_inj Output Q injection measurements [3] - buses 1,2,3
 * @param p_flow Output P flow measurements [4]
 * @param q_flow Output Q flow measurements [3]
 */
inline void getIEEE14TrueMeasurements(
    Real* voltages,
    Real* p_inj,
    Real* q_inj,
    Real* p_flow,
    Real* q_flow)
{
    // Voltage magnitudes at buses 1-14 (p.u.)
    // From IEEE 14-bus power flow solution
    voltages[0] = 1.060f;   // Bus 1
    voltages[1] = 1.045f;   // Bus 2
    voltages[2] = 1.010f;   // Bus 3
    voltages[3] = 1.018f;   // Bus 4
    voltages[4] = 1.020f;   // Bus 5
    voltages[5] = 1.070f;   // Bus 6
    voltages[6] = 1.062f;   // Bus 7
    voltages[7] = 1.090f;   // Bus 8
    voltages[8] = 1.056f;   // Bus 9
    voltages[9] = 1.051f;   // Bus 10
    voltages[10] = 1.057f;  // Bus 11
    voltages[11] = 1.055f;  // Bus 12
    voltages[12] = 1.050f;  // Bus 13
    voltages[13] = 1.036f;  // Bus 14
    
    // Active power injections (p.u.)
    p_inj[0] = 2.324f;   // Bus 1 (slack generation)
    p_inj[1] = 0.400f;   // Bus 2
    p_inj[2] = 0.000f;   // Bus 3 (synchronous condenser)
    p_inj[3] = 0.000f;   // Bus 6
    p_inj[4] = 0.000f;   // Bus 8
    
    // Reactive power injections (p.u.)
    q_inj[0] = -0.169f;  // Bus 1
    q_inj[1] = 0.424f;   // Bus 2
    q_inj[2] = 0.234f;   // Bus 3
    
    // Active power flows (p.u.)
    p_flow[0] = 1.569f;  // P1-2
    p_flow[1] = 0.755f;  // P1-5
    p_flow[2] = 0.732f;  // P2-3
    p_flow[3] = 0.281f;  // P4-7
    
    // Reactive power flows (p.u.)
    q_flow[0] = -0.204f; // Q1-2
    q_flow[1] = 0.039f;  // Q1-5
    q_flow[2] = -0.157f; // Q4-7
}

/**
 * @brief Add noise to measurements for testing
 * 
 * @param values Measurement values to corrupt
 * @param sigmas Standard deviations
 * @param n Number of measurements
 * @param seed Random seed
 */
inline void addMeasurementNoise(Real* values, const Real* sigmas, int n, unsigned int seed = 42) {
    // Simple pseudo-random noise using linear congruential generator
    unsigned int state = seed;
    
    for (int i = 0; i < n; ++i) {
        // Generate uniform random in [0,1)
        state = state * 1103515245 + 12345;
        Real u1 = static_cast<Real>((state >> 16) & 0x7FFF) / 32768.0f;
        
        state = state * 1103515245 + 12345;
        Real u2 = static_cast<Real>((state >> 16) & 0x7FFF) / 32768.0f;
        
        // Box-Muller transform for normal distribution
        Real z = sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(2.0f * 3.14159265f * u2);
        
        // Add noise
        values[i] += z * sigmas[i];
    }
}

} // namespace ieee14
} // namespace sle

#endif // IEEE14BUS_H

