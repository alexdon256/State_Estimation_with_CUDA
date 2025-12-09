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
 * @file sle_api.cu
 * @brief C-compatible DLL API implementation (NFR-20)
 * 
 * Implements the C API functions defined in sle_api.h.
 * Provides the external interface for the SLE Engine DLL.
 * 
 * Note: This file is compiled as CUDA (.cu) because it needs access to
 * internal engine classes, but the exported API is pure C.
 */

#include "../include/sle_api.h"
#include "../include/sle_engine.h"
#include "../include/sle_config.h"
#include "../include/jsf_compliance.h"
#include "../data/ieee14bus.h"
#include <cuda_runtime.h>
#include <string>
#include <cstring>
#include <algorithm>

using namespace sle;

//=============================================================================
// Internal Helper Functions
//=============================================================================

/**
 * @brief Convert internal EngineConfig to C API SLE_Config
 */
static void configToC(const EngineConfig& cfg, SLE_Config* out) {
    out->device_id = cfg.device_id;
    out->max_buses = cfg.max_buses;
    out->max_branches = cfg.max_branches;
    out->max_measurements = cfg.max_measurements;
    out->max_switching_devices = cfg.max_switching_devices;
    out->enable_robust = cfg.enable_robust_estimation ? 1 : 0;
    out->huber_gamma = cfg.huber_gamma;
    out->convergence_tolerance = cfg.convergence_tolerance;
    out->max_realtime_iterations = cfg.max_realtime_iterations;
    out->max_precision_iterations = cfg.max_precision_iterations;
    out->realtime_time_limit_ms = cfg.realtime_time_limit_ms;
    out->use_flat_start_default = cfg.use_flat_start_default ? 1 : 0;
    out->enable_profiling = cfg.enable_profiling ? 1 : 0;
    out->use_cuda_graphs = cfg.use_cuda_graphs ? 1 : 0;
    out->block_size_standard = 256;
    out->block_size_reduction = 512;
    out->use_double_precision = 0;
    out->voltage_min_pu = cfg.voltage_min_pu;
    out->voltage_max_pu = cfg.voltage_max_pu;
    out->bad_data_threshold = cfg.bad_data_threshold;
    out->auto_bad_data_rejection = cfg.auto_bad_data_rejection ? 1 : 0;
}

/**
 * @brief Convert C API SLE_Config to internal EngineConfig
 */
static EngineConfig configFromC(const SLE_Config* cfg) {
    EngineConfig out;
    out.device_id = cfg->device_id;
    out.max_buses = cfg->max_buses;
    out.max_branches = cfg->max_branches;
    out.max_measurements = cfg->max_measurements;
    out.max_switching_devices = cfg->max_switching_devices;
    out.enable_robust_estimation = cfg->enable_robust != 0;
    out.huber_gamma = cfg->huber_gamma;
    out.convergence_tolerance = cfg->convergence_tolerance;
    out.max_realtime_iterations = cfg->max_realtime_iterations;
    out.max_precision_iterations = cfg->max_precision_iterations;
    out.realtime_time_limit_ms = cfg->realtime_time_limit_ms;
    out.use_flat_start_default = cfg->use_flat_start_default != 0;
    out.enable_profiling = cfg->enable_profiling != 0;
    out.use_cuda_graphs = cfg->use_cuda_graphs != 0;
    out.voltage_min_pu = cfg->voltage_min_pu;
    out.voltage_max_pu = cfg->voltage_max_pu;
    out.bad_data_threshold = cfg->bad_data_threshold;
    out.auto_bad_data_rejection = cfg->auto_bad_data_rejection != 0;
    return out;
}

/**
 * @brief Convert internal result to C API result
 */
static void resultToC(const EstimationResult& res, SLE_Result* out) {
    out->status = static_cast<SLE_ConvergenceStatus>(res.status);
    out->iterations = res.iterations;
    out->max_mismatch = res.max_mismatch;
    out->objective = res.objective_value;
    out->computation_time_ms = res.computation_time_ms;
    out->largest_residual = res.largest_residual;
    out->largest_residual_idx = res.largest_residual_idx;
    out->bad_data_count = res.bad_data_count;
}

// Thread-local error message storage
static thread_local std::string last_error_message;
static thread_local std::string device_info_cache;
static thread_local std::string config_path_cache;

//=============================================================================
// Engine Lifecycle Functions
//=============================================================================

SLE_API SLE_StatusCode SLE_CALL sle_Create(SLE_Handle* handle) {
    if (!handle) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        *handle = reinterpret_cast<SLE_Handle>(new SLEEngine());
        return SLE_OK;
    } catch (...) {
        last_error_message = "Failed to create engine";
        return SLE_ERROR_OUT_OF_MEMORY;
    }
}

SLE_API SLE_StatusCode SLE_CALL sle_CreateWithConfig(SLE_Handle* handle, 
                                                      const SLE_Config* config) {
    if (!handle || !config) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        EngineConfig cfg = configFromC(config);
        *handle = reinterpret_cast<SLE_Handle>(new SLEEngine(cfg));
        return SLE_OK;
    } catch (...) {
        last_error_message = "Failed to create engine with config";
        return SLE_ERROR_OUT_OF_MEMORY;
    }
}

SLE_API SLE_StatusCode SLE_CALL sle_CreateFromConfigFile(SLE_Handle* handle,
                                                          const char* config_path) {
    if (!handle || !config_path) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        *handle = reinterpret_cast<SLE_Handle>(new SLEEngine(std::string(config_path)));
        return SLE_OK;
    } catch (...) {
        last_error_message = "Failed to create engine from config file";
        return SLE_ERROR_OUT_OF_MEMORY;
    }
}

SLE_API SLE_StatusCode SLE_CALL sle_Initialize(SLE_Handle handle) {
    if (!handle) {
        return SLE_ERROR_INVALID_HANDLE;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    if (engine->initialize()) {
        return SLE_OK;
    }
    
    last_error_message = "CUDA initialization failed";
    return SLE_ERROR_CUDA_FAILURE;
}

SLE_API void SLE_CALL sle_Destroy(SLE_Handle handle) {
    if (handle) {
        delete reinterpret_cast<SLEEngine*>(handle);
    }
}

//=============================================================================
// Configuration Functions
//=============================================================================

SLE_API SLE_StatusCode SLE_CALL sle_LoadConfigFile(const char* config_path, 
                                                    SLE_Config* config) {
    if (!config_path || !config) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    EngineConfiguration cfg;
    if (!ConfigLoader::loadFromFile(config_path, cfg)) {
        last_error_message = "Failed to load config file";
        return SLE_ERROR_CONFIG_FILE;
    }
    
    EngineConfig internal_cfg(cfg);
    configToC(internal_cfg, config);
    
    return SLE_OK;
}

SLE_API SLE_StatusCode SLE_CALL sle_SaveConfigFile(const char* config_path, 
                                                    const SLE_Config* config) {
    if (!config_path || !config) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    EngineConfig cfg = configFromC(config);
    if (!cfg.saveToFile(config_path)) {
        last_error_message = "Failed to save config file";
        return SLE_ERROR_CONFIG_FILE;
    }
    
    return SLE_OK;
}

SLE_API SLE_StatusCode SLE_CALL sle_ReloadConfig(SLE_Handle handle) {
    if (!handle) {
        return SLE_ERROR_INVALID_HANDLE;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    if (engine->reloadConfig()) {
        return SLE_OK;
    }
    
    last_error_message = "Failed to reload config";
    return SLE_ERROR_CONFIG_FILE;
}

SLE_API SLE_StatusCode SLE_CALL sle_ReloadConfigFromFile(SLE_Handle handle,
                                                          const char* config_path) {
    if (!handle || !config_path) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    if (engine->reloadConfig(config_path)) {
        return SLE_OK;
    }
    
    last_error_message = "Failed to reload config from file";
    return SLE_ERROR_CONFIG_FILE;
}

SLE_API const char* SLE_CALL sle_GetDefaultConfigPath(void) {
    return "sle_engine.config";
}

SLE_API const char* SLE_CALL sle_GetConfigPath(SLE_Handle handle) {
    if (!handle) return "";
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    config_path_cache = engine->getConfigPath();
    return config_path_cache.c_str();
}

SLE_API SLE_StatusCode SLE_CALL sle_EnsureConfigFile(const char* config_path) {
    std::string path = config_path ? config_path : "";
    if (SLEEngine::ensureConfigFile(path)) {
        return SLE_OK;
    }
    return SLE_ERROR_CONFIG_FILE;
}

SLE_API SLE_StatusCode SLE_CALL sle_GetConfig(SLE_Handle handle, SLE_Config* config) {
    if (!handle || !config) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    configToC(engine->getConfig(), config);
    
    return SLE_OK;
}

SLE_API SLE_StatusCode SLE_CALL sle_SetConfig(SLE_Handle handle, const SLE_Config* config) {
    if (!handle || !config) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    EngineConfig cfg = configFromC(config);
    
    if (engine->updateConfig(cfg)) {
        return SLE_OK;
    }
    
    last_error_message = "Failed to update config";
    return SLE_ERROR_INVALID_ARGUMENT;
}

//=============================================================================
// Error Handling
//=============================================================================

SLE_API const char* SLE_CALL sle_GetLastError(SLE_Handle handle) {
    (void)handle;  // Error is thread-local, not per-handle
    return last_error_message.c_str();
}

SLE_API const char* SLE_CALL sle_GetDeviceInfo(SLE_Handle handle) {
    if (!handle) return "Not initialized";
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    device_info_cache = engine->getDeviceInfo();
    return device_info_cache.c_str();
}

//=============================================================================
// Model Building Functions
//=============================================================================

SLE_API SLE_StatusCode SLE_CALL sle_AddBus(SLE_Handle handle, const SLE_BusInfo* info) {
    if (!handle || !info || !info->id) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    
    BusDescriptor desc(
        info->id,
        info->base_kv,
        static_cast<BusType>(info->type),
        info->v_setpoint,
        info->p_scheduled,
        info->q_scheduled
    );
    
    int32_t idx = engine->addBus(desc);
    if (idx == INVALID_INDEX) {
        last_error_message = "Failed to add bus - duplicate ID or invalid";
        return SLE_ERROR_DUPLICATE_ID;
    }
    
    return SLE_OK;
}

SLE_API SLE_StatusCode SLE_CALL sle_AddBuses(SLE_Handle handle, 
                                              const SLE_BusInfo* buses, 
                                              int32_t count) {
    if (!handle || !buses || count <= 0) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    for (int32_t i = 0; i < count; ++i) {
        SLE_StatusCode status = sle_AddBus(handle, &buses[i]);
        if (status != SLE_OK) return status;
    }
    return SLE_OK;
}

SLE_API SLE_StatusCode SLE_CALL sle_AddBranch(SLE_Handle handle, const SLE_BranchInfo* info) {
    if (!handle || !info || !info->id || !info->from_bus || !info->to_bus) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    
    BranchDescriptor desc(
        info->id,
        info->from_bus,
        info->to_bus,
        info->resistance,
        info->reactance,
        info->susceptance,
        info->conductance,
        info->tap_ratio,
        info->phase_shift,
        info->is_transformer != 0,
        info->sd_id ? info->sd_id : ""
    );
    
    int32_t idx = engine->addBranch(desc);
    if (idx == INVALID_INDEX) {
        last_error_message = "Failed to add branch - bus not found or duplicate ID";
        return SLE_ERROR_ELEMENT_NOT_FOUND;
    }
    
    return SLE_OK;
}

SLE_API SLE_StatusCode SLE_CALL sle_AddBranches(SLE_Handle handle,
                                                 const SLE_BranchInfo* branches,
                                                 int32_t count) {
    if (!handle || !branches || count <= 0) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    for (int32_t i = 0; i < count; ++i) {
        SLE_StatusCode status = sle_AddBranch(handle, &branches[i]);
        if (status != SLE_OK) return status;
    }
    return SLE_OK;
}

SLE_API SLE_StatusCode SLE_CALL sle_AddMeasurement(SLE_Handle handle, 
                                                    const SLE_MeasurementInfo* info) {
    if (!handle || !info || !info->id || !info->location) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    
    MeasurementDescriptor desc(
        info->id,
        static_cast<MeasurementType>(info->type),
        info->location,
        static_cast<BranchEnd>(info->branch_end),
        info->sigma,
        info->pt_ratio,
        info->ct_ratio,
        info->is_pseudo != 0
    );
    
    int32_t idx = engine->addMeasurement(desc);
    if (idx == INVALID_INDEX) {
        last_error_message = "Failed to add measurement - location not found";
        return SLE_ERROR_ELEMENT_NOT_FOUND;
    }
    
    return SLE_OK;
}

SLE_API SLE_StatusCode SLE_CALL sle_AddMeasurements(SLE_Handle handle,
                                                     const SLE_MeasurementInfo* meas,
                                                     int32_t count) {
    if (!handle || !meas || count <= 0) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    for (int32_t i = 0; i < count; ++i) {
        SLE_StatusCode status = sle_AddMeasurement(handle, &meas[i]);
        if (status != SLE_OK) return status;
    }
    return SLE_OK;
}

SLE_API SLE_StatusCode SLE_CALL sle_AddSwitch(SLE_Handle handle, const SLE_SwitchInfo* info) {
    if (!handle || !info || !info->id || !info->branch_id) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    
    SwitchingDeviceDescriptor desc(
        info->id,
        info->branch_id,
        static_cast<SwitchStatus>(info->status)
    );
    
    int32_t idx = engine->addSwitchingDevice(desc);
    if (idx == INVALID_INDEX) {
        last_error_message = "Failed to add switch - branch not found";
        return SLE_ERROR_ELEMENT_NOT_FOUND;
    }
    
    return SLE_OK;
}

SLE_API SLE_StatusCode SLE_CALL sle_RemoveBus(SLE_Handle handle, const char* bus_id) {
    if (!handle || !bus_id) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    if (engine->removeBus(bus_id)) {
        return SLE_OK;
    }
    
    return SLE_ERROR_ELEMENT_NOT_FOUND;
}

SLE_API SLE_StatusCode SLE_CALL sle_RemoveBranch(SLE_Handle handle, const char* branch_id) {
    if (!handle || !branch_id) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    if (engine->removeBranch(branch_id)) {
        return SLE_OK;
    }
    
    return SLE_ERROR_ELEMENT_NOT_FOUND;
}

SLE_API SLE_StatusCode SLE_CALL sle_UpdateBranch(SLE_Handle handle, const SLE_BranchInfo* info) {
    if (!handle || !info || !info->id) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    
    BranchDescriptor desc(
        info->id,
        info->from_bus ? info->from_bus : "",
        info->to_bus ? info->to_bus : "",
        info->resistance,
        info->reactance,
        info->susceptance,
        info->conductance,
        info->tap_ratio,
        info->phase_shift,
        info->is_transformer != 0,
        info->sd_id ? info->sd_id : ""
    );
    
    if (engine->updateBranch(info->id, desc)) {
        return SLE_OK;
    }
    
    last_error_message = "Failed to update branch - branch not found";
    return SLE_ERROR_ELEMENT_NOT_FOUND;
}

SLE_API SLE_StatusCode SLE_CALL sle_SetTransformerTap(SLE_Handle handle,
                                                       const char* branch_id,
                                                       SLE_Real tap_ratio) {
    if (!handle || !branch_id) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    if (engine->setTransformerTap(branch_id, tap_ratio)) {
        return SLE_OK;
    }
    
    last_error_message = "Failed to set tap - branch not found or not a transformer";
    return SLE_ERROR_ELEMENT_NOT_FOUND;
}

SLE_API SLE_StatusCode SLE_CALL sle_SetTransformerPhaseShift(SLE_Handle handle,
                                                              const char* branch_id,
                                                              SLE_Real phase_shift) {
    if (!handle || !branch_id) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    if (engine->setTransformerPhaseShift(branch_id, phase_shift)) {
        return SLE_OK;
    }
    
    last_error_message = "Failed to set phase shift - branch not found";
    return SLE_ERROR_ELEMENT_NOT_FOUND;
}

SLE_API SLE_StatusCode SLE_CALL sle_SetBranchImpedance(SLE_Handle handle,
                                                        const char* branch_id,
                                                        SLE_Real r,
                                                        SLE_Real x,
                                                        SLE_Real b,
                                                        SLE_Real g) {
    if (!handle || !branch_id) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    if (engine->setBranchImpedance(branch_id, r, x, b, g)) {
        return SLE_OK;
    }
    
    last_error_message = "Failed to set impedance - branch not found";
    return SLE_ERROR_ELEMENT_NOT_FOUND;
}

SLE_API SLE_Real SLE_CALL sle_GetTransformerTap(SLE_Handle handle, const char* branch_id) {
    if (!handle || !branch_id) return 1.0f;
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    const NetworkModel& model = engine->getModel();
    int32_t idx = model.getBranchIndex(branch_id);
    if (idx != INVALID_INDEX) {
        const auto* branch = model.getBranch(idx);
        if (branch) {
            return branch->descriptor.tap_ratio;
        }
    }
    return 1.0f;
}

SLE_API SLE_Real SLE_CALL sle_GetTransformerPhaseShift(SLE_Handle handle, const char* branch_id) {
    if (!handle || !branch_id) return 0.0f;
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    const NetworkModel& model = engine->getModel();
    int32_t idx = model.getBranchIndex(branch_id);
    if (idx != INVALID_INDEX) {
        const auto* branch = model.getBranch(idx);
        if (branch) {
            return branch->descriptor.phase_shift;
        }
    }
    return 0.0f;
}

//=============================================================================
// Meter Device Management
//=============================================================================

SLE_API SLE_StatusCode SLE_CALL sle_AddMeter(SLE_Handle handle, const SLE_MeterInfo* info) {
    if (!handle || !info || !info->id || !info->bus_id) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    
    MeterDescriptor desc;
    desc.id = info->id;
    desc.type = static_cast<MeterType>(info->type);
    desc.bus_id = info->bus_id;
    desc.branch_id = info->branch_id ? info->branch_id : "";
    desc.branch_end = static_cast<BranchEnd>(info->branch_end);
    desc.pt_ratio = info->pt_ratio;
    desc.ct_ratio = info->ct_ratio;
    desc.sigma_v = info->sigma_v;
    desc.sigma_p = info->sigma_p;
    desc.sigma_i = info->sigma_i;
    
    if (engine->addMeter(desc) != INVALID_INDEX) {
        return SLE_OK;
    }
    
    last_error_message = "Failed to add meter: check bus/branch IDs";
    return SLE_ERROR_ELEMENT_NOT_FOUND;
}

SLE_API SLE_StatusCode SLE_CALL sle_UpdateMeterReading(SLE_Handle handle,
                                                        const char* meter_id,
                                                        const char* channel,
                                                        SLE_Real value) {
    if (!handle || !meter_id || !channel) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    if (engine->updateMeterReading(meter_id, channel, value)) {
        return SLE_OK;
    }
    
    last_error_message = "Meter or channel not found: " + std::string(meter_id) + "." + channel;
    return SLE_ERROR_ELEMENT_NOT_FOUND;
}

SLE_API SLE_StatusCode SLE_CALL sle_UpdateMeterReadings(SLE_Handle handle,
                                                         const SLE_MeterReading* readings,
                                                         int32_t count,
                                                         int32_t sync_to_device) {
    if (!handle || !readings || count <= 0) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    
    // Update all readings
    for (int32_t i = 0; i < count; ++i) {
        if (!readings[i].meter_id || !readings[i].channel) {
            last_error_message = "Null meter_id or channel at index " + std::to_string(i);
            return SLE_ERROR_INVALID_ARGUMENT;
        }
        if (!engine->updateMeterReading(readings[i].meter_id, 
                                        readings[i].channel, 
                                        readings[i].value)) {
            last_error_message = "Meter or channel not found: " + 
                std::string(readings[i].meter_id) + "." + readings[i].channel;
            return SLE_ERROR_ELEMENT_NOT_FOUND;
        }
    }
    
    // Optionally sync to GPU
    if (sync_to_device) {
        if (!engine->syncMeasurementsToDevice()) {
            last_error_message = "Failed to sync to device";
            return SLE_ERROR_OUT_OF_MEMORY;
        }
    }
    
    return SLE_OK;
}

SLE_API SLE_StatusCode SLE_CALL sle_GetMeterReading(SLE_Handle handle,
                                                     const char* meter_id,
                                                     const char* channel,
                                                     SLE_Real* value) {
    if (!handle || !meter_id || !channel || !value) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    Real v;
    if (engine->getMeterReading(meter_id, channel, v)) {
        *value = v;
        return SLE_OK;
    }
    
    return SLE_ERROR_ELEMENT_NOT_FOUND;
}

SLE_API SLE_StatusCode SLE_CALL sle_GetMeterEstimate(SLE_Handle handle,
                                                      const char* meter_id,
                                                      const char* channel,
                                                      SLE_Real* value) {
    if (!handle || !meter_id || !channel || !value) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    Real v;
    if (engine->getMeterEstimate(meter_id, channel, v)) {
        *value = v;
        return SLE_OK;
    }
    
    return SLE_ERROR_ELEMENT_NOT_FOUND;
}

SLE_API SLE_StatusCode SLE_CALL sle_GetMeterResidual(SLE_Handle handle,
                                                      const char* meter_id,
                                                      const char* channel,
                                                      SLE_Real* residual) {
    if (!handle || !meter_id || !channel || !residual) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    Real v;
    if (engine->getMeterResidual(meter_id, channel, v)) {
        *residual = v;
        return SLE_OK;
    }
    
    return SLE_ERROR_ELEMENT_NOT_FOUND;
}

//=============================================================================
// GPU Data Management
//=============================================================================

SLE_API SLE_StatusCode SLE_CALL sle_UploadModel(SLE_Handle handle) {
    if (!handle) {
        return SLE_ERROR_INVALID_HANDLE;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    if (engine->uploadModel()) {
        return SLE_OK;
    }
    
    last_error_message = "Failed to upload model to GPU";
    return SLE_ERROR_CUDA_FAILURE;
}

SLE_API SLE_StatusCode SLE_CALL sle_UpdateTelemetry(SLE_Handle handle,
                                                     const SLE_Real* values,
                                                     int32_t count) {
    if (!handle || !values || count <= 0) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    if (engine->updateTelemetry(values, count)) {
        return SLE_OK;
    }
    
    last_error_message = "Failed to update telemetry";
    return SLE_ERROR_INVALID_ARGUMENT;
}

SLE_API SLE_StatusCode SLE_CALL sle_UpdateMeasurement(SLE_Handle handle,
                                                       const char* meas_id,
                                                       SLE_Real value) {
    if (!handle || !meas_id) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    if (engine->updateMeasurement(meas_id, value)) {
        return SLE_OK;
    }
    
    return SLE_ERROR_ELEMENT_NOT_FOUND;
}

SLE_API SLE_StatusCode SLE_CALL sle_UpdateMeasurementBatch(SLE_Handle handle,
                                                            const SLE_MeasurementUpdate* updates,
                                                            int32_t count,
                                                            int32_t sync_to_device) {
    if (!handle || !updates || count <= 0) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    
    // Update all measurements in host memory
    for (int32_t i = 0; i < count; ++i) {
        if (!updates[i].meas_id) {
            last_error_message = "Null measurement ID at index " + std::to_string(i);
            return SLE_ERROR_INVALID_ARGUMENT;
        }
        if (!engine->updateMeasurement(updates[i].meas_id, updates[i].value)) {
            last_error_message = "Measurement not found: " + std::string(updates[i].meas_id);
            return SLE_ERROR_ELEMENT_NOT_FOUND;
        }
    }
    
    // Optionally sync to GPU
    if (sync_to_device) {
        if (!engine->syncMeasurementsToDevice()) {
            last_error_message = "Failed to sync measurements to device";
            return SLE_ERROR_OUT_OF_MEMORY;
        }
    }
    
    return SLE_OK;
}

SLE_API SLE_StatusCode SLE_CALL sle_SyncMeasurementsToDevice(SLE_Handle handle) {
    if (!handle) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    if (engine->syncMeasurementsToDevice()) {
        return SLE_OK;
    }
    
    last_error_message = "Failed to sync measurements to device";
    return SLE_ERROR_OUT_OF_MEMORY;
}

SLE_API SLE_StatusCode SLE_CALL sle_GetMeasurementDetails(SLE_Handle handle,
                                                           const char* meas_id,
                                                           SLE_MeasurementDetails* details) {
    if (!handle || !meas_id || !details) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    const NetworkModel& model = engine->getModel();
    
    int32_t idx = model.getMeasurementIndex(meas_id);
    if (idx == INVALID_INDEX) {
        return SLE_ERROR_ELEMENT_NOT_FOUND;
    }
    
    const MeasurementElement* elem = model.getMeasurement(idx);
    if (!elem) {
        return SLE_ERROR_ELEMENT_NOT_FOUND;
    }
    
    // Thread-local storage for string IDs (valid until next call)
    static thread_local std::string s_meas_id;
    static thread_local std::string s_location_id;
    
    s_meas_id = elem->descriptor.id;
    s_location_id = elem->descriptor.location_id;
    
    details->id = s_meas_id.c_str();
    details->type = static_cast<SLE_MeasurementType>(elem->descriptor.type);
    details->location_id = s_location_id.c_str();
    details->branch_end = static_cast<SLE_BranchEnd>(elem->descriptor.branch_end);
    details->sigma = elem->descriptor.sigma;
    details->pt_ratio = elem->descriptor.pt_ratio;
    details->ct_ratio = elem->descriptor.ct_ratio;
    details->is_pseudo = elem->descriptor.is_pseudo ? 1 : 0;
    details->internal_index = elem->index;
    details->current_value = elem->value;
    details->estimated_value = elem->estimated;
    details->residual = elem->residual;
    details->is_active = elem->is_active ? 1 : 0;
    
    return SLE_OK;
}

SLE_API const char* SLE_CALL sle_GetMeasurementIdByIndex(SLE_Handle handle, int32_t index) {
    if (!handle || index < 0) {
        return nullptr;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    const NetworkModel& model = engine->getModel();
    
    const MeasurementElement* elem = model.getMeasurement(index);
    if (!elem) {
        return nullptr;
    }
    
    // Thread-local storage for the ID string (valid until next call from same thread)
    static thread_local std::string s_meas_id;
    s_meas_id = elem->descriptor.id;
    
    return s_meas_id.c_str();
}

//=============================================================================
// Switch Control
//=============================================================================

SLE_API SLE_StatusCode SLE_CALL sle_SetSwitchStatus(SLE_Handle handle,
                                                     const char* sd_id,
                                                     SLE_SwitchStatus status) {
    if (!handle || !sd_id) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    if (engine->setSwitchStatus(sd_id, static_cast<SwitchStatus>(status))) {
        return SLE_OK;
    }
    
    return SLE_ERROR_ELEMENT_NOT_FOUND;
}

SLE_API SLE_SwitchStatus SLE_CALL sle_GetSwitchStatus(SLE_Handle handle, const char* sd_id) {
    if (!handle || !sd_id) {
        return SLE_SWITCH_OPEN;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    return static_cast<SLE_SwitchStatus>(engine->getSwitchStatus(sd_id));
}

SLE_API SLE_StatusCode SLE_CALL sle_ApplySwitchChanges(SLE_Handle handle) {
    if (!handle) {
        return SLE_ERROR_INVALID_HANDLE;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    if (engine->applyPendingSwitchChanges()) {
        return SLE_OK;
    }
    
    return SLE_ERROR_CUDA_FAILURE;
}

//=============================================================================
// State Estimation
//=============================================================================

SLE_API SLE_StatusCode SLE_CALL sle_Solve(SLE_Handle handle,
                                           SLE_EstimationMode mode,
                                           SLE_Result* result) {
    if (!handle) {
        return SLE_ERROR_INVALID_HANDLE;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    EstimationResult res = engine->solve(static_cast<EstimationMode>(mode));
    
    if (result) {
        resultToC(res, result);
    }
    
    switch (res.status) {
        case ConvergenceStatus::CONVERGED:
            return SLE_OK;
        case ConvergenceStatus::MAX_ITERATIONS:
            return SLE_ERROR_MAX_ITERATIONS;
        case ConvergenceStatus::DIVERGED:
            return SLE_ERROR_DIVERGED;
        case ConvergenceStatus::SINGULAR_MATRIX:
            return SLE_ERROR_SINGULAR_MATRIX;
        case ConvergenceStatus::NOT_OBSERVABLE:
            return SLE_ERROR_NOT_OBSERVABLE;
        default:
            return SLE_ERROR_UNKNOWN;
    }
}

SLE_API SLE_StatusCode SLE_CALL sle_SolveWithConfig(SLE_Handle handle,
                                                     const SLE_SolverConfig* config,
                                                     SLE_Result* result) {
    if (!handle || !config) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    
    SolverConfig solver_config;
    solver_config.mode = static_cast<EstimationMode>(config->mode);
    solver_config.max_iterations = config->max_iterations;
    solver_config.convergence_tolerance = config->tolerance;
    solver_config.huber_gamma = config->huber_gamma;
    solver_config.use_robust_estimation = config->use_robust != 0;
    solver_config.use_flat_start = config->use_flat_start != 0;
    solver_config.time_limit_ms = config->time_limit_ms;
    
    EstimationResult res = engine->solve(solver_config);
    
    if (result) {
        resultToC(res, result);
    }
    
    return (res.status == ConvergenceStatus::CONVERGED) ? SLE_OK : SLE_ERROR_MAX_ITERATIONS;
}

SLE_API SLE_StatusCode SLE_CALL sle_CheckObservability(SLE_Handle handle,
                                                        int32_t* is_observable,
                                                        int32_t* island_count) {
    if (!handle) {
        return SLE_ERROR_INVALID_HANDLE;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    bool observable = engine->checkObservability();
    
    if (is_observable) *is_observable = observable ? 1 : 0;
    if (island_count) *island_count = engine->getIslandCount();
    
    return SLE_OK;
}

SLE_API SLE_StatusCode SLE_CALL sle_ApplyFlatStart(SLE_Handle handle) {
    if (!handle) {
        return SLE_ERROR_INVALID_HANDLE;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    engine->applyFlatStart();
    
    return SLE_OK;
}

//=============================================================================
// Result Access
//=============================================================================

SLE_API SLE_StatusCode SLE_CALL sle_GetVoltageMagnitudes(SLE_Handle handle,
                                                          SLE_Real* values,
                                                          int32_t count) {
    if (!handle || !values || count <= 0) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    auto voltages = engine->getVoltageMagnitudes();
    
    int32_t n = std::min(count, static_cast<int32_t>(voltages.size()));
    std::memcpy(values, voltages.data(), n * sizeof(SLE_Real));
    
    return SLE_OK;
}

SLE_API SLE_StatusCode SLE_CALL sle_GetVoltageAngles(SLE_Handle handle,
                                                      SLE_Real* values,
                                                      int32_t count) {
    if (!handle || !values || count <= 0) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    auto angles = engine->getVoltageAngles();
    
    int32_t n = std::min(count, static_cast<int32_t>(angles.size()));
    std::memcpy(values, angles.data(), n * sizeof(SLE_Real));
    
    return SLE_OK;
}

SLE_API SLE_Real SLE_CALL sle_GetBusVoltage(SLE_Handle handle, const char* bus_id) {
    if (!handle || !bus_id) return 0.0f;
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    return engine->getVoltageMagnitude(bus_id);
}

SLE_API SLE_Real SLE_CALL sle_GetBusAngle(SLE_Handle handle, const char* bus_id) {
    if (!handle || !bus_id) return 0.0f;
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    return engine->getVoltageAngle(bus_id);
}

SLE_API SLE_StatusCode SLE_CALL sle_GetPowerInjections(SLE_Handle handle,
                                                        SLE_Real* p_values,
                                                        SLE_Real* q_values,
                                                        int32_t count) {
    if (!handle || count <= 0) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    
    if (p_values) {
        auto p = engine->getPowerInjections();
        int32_t n = std::min(count, static_cast<int32_t>(p.size()));
        std::memcpy(p_values, p.data(), n * sizeof(SLE_Real));
    }
    
    if (q_values) {
        auto q = engine->getReactivePowerInjections();
        int32_t n = std::min(count, static_cast<int32_t>(q.size()));
        std::memcpy(q_values, q.data(), n * sizeof(SLE_Real));
    }
    
    return SLE_OK;
}

SLE_API SLE_StatusCode SLE_CALL sle_GetBranchFlows(SLE_Handle handle,
                                                    SLE_Real* p_from,
                                                    SLE_Real* q_from,
                                                    SLE_Real* p_to,
                                                    SLE_Real* q_to,
                                                    int32_t count) {
    if (!handle || count <= 0) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    const NetworkModel& model = engine->getModel();
    int32_t n = std::min(count, model.getBranchCount());
    
    for (int32_t i = 0; i < n; ++i) {
        const auto* branch = model.getBranch(i);
        if (branch) {
            if (p_from) p_from[i] = branch->p_flow_from;
            if (q_from) q_from[i] = branch->q_flow_from;
            if (p_to) p_to[i] = branch->p_flow_to;
            if (q_to) q_to[i] = branch->q_flow_to;
        }
    }
    
    return SLE_OK;
}

SLE_API SLE_Real SLE_CALL sle_GetBranchP(SLE_Handle handle, 
                                          const char* branch_id,
                                          SLE_BranchEnd end) {
    if (!handle || !branch_id) return 0.0f;
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    return engine->getPowerFlow(branch_id, static_cast<BranchEnd>(end));
}

SLE_API SLE_Real SLE_CALL sle_GetBranchQ(SLE_Handle handle,
                                          const char* branch_id,
                                          SLE_BranchEnd end) {
    if (!handle || !branch_id) return 0.0f;
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    return engine->getReactivePowerFlow(branch_id, static_cast<BranchEnd>(end));
}

SLE_API SLE_StatusCode SLE_CALL sle_GetBusResiduals(SLE_Handle handle,
                                                     const char* bus_id,
                                                     SLE_Real* v_residual,
                                                     SLE_Real* p_residual,
                                                     SLE_Real* q_residual,
                                                     int32_t* count) {
    if (!handle || !bus_id) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    const NetworkModel& model = engine->getModel();
    
    // Verify bus exists
    if (model.getBusIndex(bus_id) == INVALID_INDEX) {
        return SLE_ERROR_ELEMENT_NOT_FOUND;
    }
    
    int32_t found_count = 0;
    
    // Initialize outputs to NaN (not found)
    if (v_residual) *v_residual = std::numeric_limits<SLE_Real>::quiet_NaN();
    if (p_residual) *p_residual = std::numeric_limits<SLE_Real>::quiet_NaN();
    if (q_residual) *q_residual = std::numeric_limits<SLE_Real>::quiet_NaN();
    
    // Search through all measurements for ones at this bus
    int32_t n_meas = static_cast<int32_t>(model.getMeasurementCount());
    for (int32_t i = 0; i < n_meas; ++i) {
        const MeasurementElement* elem = model.getMeasurement(i);
        if (!elem || elem->descriptor.location_id != bus_id) continue;
        
        switch (elem->descriptor.type) {
            case MeasurementType::V_MAG:
                if (v_residual) *v_residual = elem->residual;
                found_count++;
                break;
            case MeasurementType::P_INJECTION:
            case MeasurementType::P_PSEUDO:
                if (p_residual) *p_residual = elem->residual;
                found_count++;
                break;
            case MeasurementType::Q_INJECTION:
            case MeasurementType::Q_PSEUDO:
                if (q_residual) *q_residual = elem->residual;
                found_count++;
                break;
            default:
                break;
        }
    }
    
    if (count) *count = found_count;
    return SLE_OK;
}

SLE_API SLE_StatusCode SLE_CALL sle_GetBranchResiduals(SLE_Handle handle,
                                                        const char* branch_id,
                                                        SLE_BranchEnd end,
                                                        SLE_Real* p_residual,
                                                        SLE_Real* q_residual,
                                                        SLE_Real* i_residual,
                                                        int32_t* count) {
    if (!handle || !branch_id) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    const NetworkModel& model = engine->getModel();
    
    // Verify branch exists
    if (model.getBranchIndex(branch_id) == INVALID_INDEX) {
        return SLE_ERROR_ELEMENT_NOT_FOUND;
    }
    
    int32_t found_count = 0;
    BranchEnd target_end = static_cast<BranchEnd>(end);
    
    // Initialize outputs to NaN (not found)
    if (p_residual) *p_residual = std::numeric_limits<SLE_Real>::quiet_NaN();
    if (q_residual) *q_residual = std::numeric_limits<SLE_Real>::quiet_NaN();
    if (i_residual) *i_residual = std::numeric_limits<SLE_Real>::quiet_NaN();
    
    // Search through all measurements for ones at this branch
    int32_t n_meas = static_cast<int32_t>(model.getMeasurementCount());
    for (int32_t i = 0; i < n_meas; ++i) {
        const MeasurementElement* elem = model.getMeasurement(i);
        if (!elem || elem->descriptor.location_id != branch_id) continue;
        if (elem->descriptor.branch_end != target_end) continue;
        
        switch (elem->descriptor.type) {
            case MeasurementType::P_FLOW:
                if (p_residual) *p_residual = elem->residual;
                found_count++;
                break;
            case MeasurementType::Q_FLOW:
                if (q_residual) *q_residual = elem->residual;
                found_count++;
                break;
            case MeasurementType::I_MAG:
                if (i_residual) *i_residual = elem->residual;
                found_count++;
                break;
            default:
                break;
        }
    }
    
    if (count) *count = found_count;
    return SLE_OK;
}

SLE_API SLE_StatusCode SLE_CALL sle_GetResiduals(SLE_Handle handle,
                                                  SLE_Real* values,
                                                  int32_t count) {
    if (!handle || !values || count <= 0) {
        return SLE_ERROR_INVALID_ARGUMENT;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    auto residuals = engine->getResiduals();
    
    int32_t n = std::min(count, static_cast<int32_t>(residuals.size()));
    std::memcpy(values, residuals.data(), n * sizeof(SLE_Real));
    
    return SLE_OK;
}

SLE_API SLE_Real SLE_CALL sle_GetObjective(SLE_Handle handle) {
    if (!handle) return 0.0f;
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    return engine->getObjectiveValue();
}

SLE_API SLE_StatusCode SLE_CALL sle_IdentifyBadData(SLE_Handle handle,
                                                     SLE_Real threshold,
                                                     int32_t* indices,
                                                     int32_t max_count,
                                                     int32_t* actual_count) {
    if (!handle) {
        return SLE_ERROR_INVALID_HANDLE;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    auto bad_data = engine->identifyBadData(threshold);
    
    int32_t n = std::min(max_count, static_cast<int32_t>(bad_data.size()));
    if (indices && n > 0) {
        std::memcpy(indices, bad_data.data(), n * sizeof(int32_t));
    }
    if (actual_count) {
        *actual_count = static_cast<int32_t>(bad_data.size());
    }
    
    return SLE_OK;
}

//=============================================================================
// Model Information
//=============================================================================

SLE_API int32_t SLE_CALL sle_GetBusCount(SLE_Handle handle) {
    if (!handle) return 0;
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    return engine->getModel().getBusCount();
}

SLE_API int32_t SLE_CALL sle_GetBranchCount(SLE_Handle handle) {
    if (!handle) return 0;
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    return engine->getModel().getBranchCount();
}

SLE_API int32_t SLE_CALL sle_GetMeasurementCount(SLE_Handle handle) {
    if (!handle) return 0;
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    return engine->getModel().getMeasurementCount();
}

SLE_API int32_t SLE_CALL sle_GetMeterCount(SLE_Handle handle) {
    if (!handle) return 0;
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    return engine->getModel().getMeterCount();
}

SLE_API size_t SLE_CALL sle_GetGPUMemoryUsage(SLE_Handle handle) {
    if (!handle) return 0;
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    return engine->getGPUMemoryUsage();
}

SLE_API SLE_Real SLE_CALL sle_GetLastSolveTime(SLE_Handle handle) {
    if (!handle) return 0.0f;
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    return engine->getLastSolveTime();
}

//=============================================================================
// IEEE 14-Bus Test Case
//=============================================================================

SLE_API SLE_StatusCode SLE_CALL sle_LoadIEEE14Bus(SLE_Handle handle) {
    if (!handle) {
        return SLE_ERROR_INVALID_HANDLE;
    }
    
    SLEEngine* engine = reinterpret_cast<SLEEngine*>(handle);
    if (ieee14::loadIEEE14Bus(*engine)) {
        return SLE_OK;
    }
    
    last_error_message = "Failed to load IEEE 14-bus test case";
    return SLE_ERROR_UNKNOWN;
}

//=============================================================================
// Version Information
//=============================================================================

SLE_API const char* SLE_CALL sle_GetVersion(void) {
    return SLE_VERSION_STRING;
}

SLE_API int32_t SLE_CALL sle_GetCUDAVersion(void) {
    int version = 0;
    cudaRuntimeGetVersion(&version);
    return version;
}

SLE_API int32_t SLE_CALL sle_IsGPUAvailable(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0) ? 1 : 0;
}
