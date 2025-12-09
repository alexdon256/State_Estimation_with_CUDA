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
 * @file sle_config.h
 * @brief Configuration file loader for SLE Engine
 * 
 * Provides functionality to load engine configuration from .config files.
 * Configuration files use a simple key=value format with support for:
 * - Comments (lines starting with # or ;)
 * - Sections (optional, in [section] format)
 * - Integer, floating-point, boolean, and string values
 * 
 * Example config file:
 * @code
 * # SLE Engine Configuration
 * [device]
 * device_id = 0
 * 
 * [capacity]
 * max_buses = 200000
 * max_branches = 400000
 * max_measurements = 2000000
 * 
 * [solver]
 * enable_robust_estimation = false
 * huber_gamma = 1.5
 * convergence_tolerance = 1e-4
 * max_realtime_iterations = 5
 * max_precision_iterations = 100
 * realtime_time_limit_ms = 20.0
 * 
 * [performance]
 * enable_profiling = false
 * use_cuda_graphs = true
 * @endcode
 */

#ifndef SLE_CONFIG_H
#define SLE_CONFIG_H

#include "sle_types.cuh"
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace sle {

//=============================================================================
// Configuration Keys (String Constants)
//=============================================================================

namespace config_keys {
    // Device section
    constexpr const char* DEVICE_ID = "device_id";
    
    // Capacity section
    constexpr const char* MAX_BUSES = "max_buses";
    constexpr const char* MAX_BRANCHES = "max_branches";
    constexpr const char* MAX_MEASUREMENTS = "max_measurements";
    constexpr const char* MAX_SWITCHING_DEVICES = "max_switching_devices";
    
    // Solver section
    constexpr const char* ENABLE_ROBUST_ESTIMATION = "enable_robust_estimation";
    constexpr const char* HUBER_GAMMA = "huber_gamma";
    constexpr const char* CONVERGENCE_TOLERANCE = "convergence_tolerance";
    constexpr const char* MAX_REALTIME_ITERATIONS = "max_realtime_iterations";
    constexpr const char* MAX_PRECISION_ITERATIONS = "max_precision_iterations";
    constexpr const char* REALTIME_TIME_LIMIT_MS = "realtime_time_limit_ms";
    constexpr const char* USE_FLAT_START_DEFAULT = "use_flat_start_default";
    
    // Performance section
    constexpr const char* ENABLE_PROFILING = "enable_profiling";
    constexpr const char* USE_CUDA_GRAPHS = "use_cuda_graphs";
    constexpr const char* BLOCK_SIZE_STANDARD = "block_size_standard";
    constexpr const char* BLOCK_SIZE_REDUCTION = "block_size_reduction";
    
    // Numerical section
    constexpr const char* USE_DOUBLE_PRECISION = "use_double_precision";
    constexpr const char* VOLTAGE_MIN_PU = "voltage_min_pu";
    constexpr const char* VOLTAGE_MAX_PU = "voltage_max_pu";
    
    // Bad data detection section
    constexpr const char* BAD_DATA_THRESHOLD = "bad_data_threshold";
    constexpr const char* AUTO_BAD_DATA_REJECTION = "auto_bad_data_rejection";
}

//=============================================================================
// Configuration Structure
//=============================================================================

/**
 * @struct EngineConfiguration
 * @brief Complete engine configuration loaded from file
 * 
 * All configurable parameters for the SLE Engine.
 * Can be loaded from a .config file or set programmatically.
 */
struct EngineConfiguration {
    //-------------------------------------------------------------------------
    // Device Settings
    //-------------------------------------------------------------------------
    int32_t device_id = 0;                          ///< CUDA device ID
    
    //-------------------------------------------------------------------------
    // Capacity Settings (NFR-01)
    //-------------------------------------------------------------------------
    size_t max_buses = 200000;                      ///< Maximum number of buses
    size_t max_branches = 400000;                   ///< Maximum number of branches
    size_t max_measurements = 2000000;              ///< Maximum number of measurements
    size_t max_switching_devices = 100000;          ///< Maximum switching devices
    
    //-------------------------------------------------------------------------
    // Solver Settings (FR-11, FR-12, FR-17)
    //-------------------------------------------------------------------------
    bool enable_robust_estimation = false;          ///< Enable Huber M-estimator
    Real huber_gamma = 1.5f;                        ///< Huber threshold parameter
    Real convergence_tolerance = 1e-4f;             ///< WLS convergence tolerance
    int32_t max_realtime_iterations = 5;            ///< Max iterations for real-time
    int32_t max_precision_iterations = 100;         ///< Max iterations for precision
    Real realtime_time_limit_ms = 20.0f;            ///< Time limit for real-time [ms]
    bool use_flat_start_default = false;            ///< Default to flat start
    
    //-------------------------------------------------------------------------
    // Performance Settings
    //-------------------------------------------------------------------------
    bool enable_profiling = false;                  ///< Enable CUDA event timing
    bool use_cuda_graphs = true;                    ///< Use CUDA graphs for iteration
    int32_t block_size_standard = 256;              ///< Standard kernel block size
    int32_t block_size_reduction = 512;             ///< Reduction kernel block size
    
    //-------------------------------------------------------------------------
    // Numerical Settings
    //-------------------------------------------------------------------------
    bool use_double_precision = false;              ///< Use fp64 instead of fp32
    Real voltage_min_pu = 0.5f;                     ///< Minimum voltage limit [p.u.]
    Real voltage_max_pu = 1.5f;                     ///< Maximum voltage limit [p.u.]
    
    //-------------------------------------------------------------------------
    // Bad Data Detection Settings (FR-15)
    //-------------------------------------------------------------------------
    Real bad_data_threshold = 3.0f;                 ///< Normalized residual threshold
    bool auto_bad_data_rejection = false;           ///< Automatically reject bad data
    
    /**
     * @brief Validate configuration values
     * @return true if all values are within valid ranges
     */
    [[nodiscard]] bool validate() const {
        if (device_id < 0) return false;
        if (max_buses == 0) return false;
        if (max_branches == 0) return false;
        if (max_measurements == 0) return false;
        if (huber_gamma <= 0) return false;
        if (convergence_tolerance <= 0) return false;
        if (max_realtime_iterations < 1) return false;
        if (max_precision_iterations < 1) return false;
        if (realtime_time_limit_ms <= 0) return false;
        if (voltage_min_pu <= 0 || voltage_min_pu >= voltage_max_pu) return false;
        if (bad_data_threshold <= 0) return false;
        if (block_size_standard < 32 || block_size_standard > 1024) return false;
        if (block_size_reduction < 32 || block_size_reduction > 1024) return false;
        return true;
    }
};

//=============================================================================
// Configuration File Loader
//=============================================================================

/**
 * @class ConfigLoader
 * @brief Loads and parses SLE configuration files
 * 
 * Supports loading from:
 * - File path
 * - String content
 * - Default values
 * 
 * File format:
 * - Lines starting with # or ; are comments
 * - Empty lines are ignored
 * - [section] headers are optional (for organization)
 * - key = value pairs define settings
 * - Values are trimmed of whitespace
 * - Boolean values: true/false, yes/no, 1/0
 */
class ConfigLoader {
public:
    /**
     * @brief Load configuration from file
     * 
     * @param filepath Path to .config file
     * @param config Output configuration structure
     * @return true if loaded successfully
     */
    [[nodiscard]] static bool loadFromFile(const std::string& filepath,
                                           EngineConfiguration& config) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            return false;
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        return loadFromString(buffer.str(), config);
    }
    
    /**
     * @brief Load configuration from string content
     * 
     * @param content Configuration file content
     * @param config Output configuration structure
     * @return true if parsed successfully
     */
    [[nodiscard]] static bool loadFromString(const std::string& content,
                                              EngineConfiguration& config) {
        std::unordered_map<std::string, std::string> values;
        
        std::istringstream stream(content);
        std::string line;
        std::string current_section;
        
        while (std::getline(stream, line)) {
            // Trim whitespace
            line = trim(line);
            
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#' || line[0] == ';') {
                continue;
            }
            
            // Check for section header
            if (line[0] == '[' && line.back() == ']') {
                current_section = line.substr(1, line.length() - 2);
                continue;
            }
            
            // Parse key=value pair
            size_t eq_pos = line.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = trim(line.substr(0, eq_pos));
                std::string value = trim(line.substr(eq_pos + 1));
                
                // Store with section prefix if applicable
                std::string full_key = current_section.empty() ? 
                    key : (current_section + "." + key);
                values[key] = value;  // Also store without section for flexibility
                values[full_key] = value;
            }
        }
        
        // Apply values to configuration
        applyValues(values, config);
        
        return config.validate();
    }
    
    /**
     * @brief Save configuration to file
     * 
     * @param filepath Output file path
     * @param config Configuration to save
     * @return true if saved successfully
     */
    [[nodiscard]] static bool saveToFile(const std::string& filepath,
                                          const EngineConfiguration& config) {
        std::ofstream file(filepath);
        if (!file.is_open()) {
            return false;
        }
        
        file << generateConfigString(config);
        return true;
    }
    
    /**
     * @brief Generate configuration file content string
     * 
     * @param config Configuration to serialize
     * @return Configuration file content
     */
    [[nodiscard]] static std::string generateConfigString(
            const EngineConfiguration& config) {
        std::ostringstream ss;
        
        ss << "# SLE Engine Configuration File\n";
        ss << "# Generated automatically - edit values as needed\n";
        ss << "#\n";
        ss << "# Lines starting with # or ; are comments\n";
        ss << "# Format: key = value\n\n";
        
        ss << "[device]\n";
        ss << "# CUDA device ID (0 = first GPU)\n";
        ss << config_keys::DEVICE_ID << " = " << config.device_id << "\n\n";
        
        ss << "[capacity]\n";
        ss << "# Maximum network size (pre-allocation)\n";
        ss << "# NFR-01: Support up to 200,000 buses and 2,000,000 measurements\n";
        ss << config_keys::MAX_BUSES << " = " << config.max_buses << "\n";
        ss << config_keys::MAX_BRANCHES << " = " << config.max_branches << "\n";
        ss << config_keys::MAX_MEASUREMENTS << " = " << config.max_measurements << "\n";
        ss << config_keys::MAX_SWITCHING_DEVICES << " = " 
           << config.max_switching_devices << "\n\n";
        
        ss << "[solver]\n";
        ss << "# WLS solver settings\n";
        ss << "# FR-17: Robust estimation using Huber M-estimator\n";
        ss << config_keys::ENABLE_ROBUST_ESTIMATION << " = " 
           << (config.enable_robust_estimation ? "true" : "false") << "\n";
        ss << config_keys::HUBER_GAMMA << " = " << config.huber_gamma << "\n";
        ss << "\n# Convergence settings\n";
        ss << config_keys::CONVERGENCE_TOLERANCE << " = " 
           << config.convergence_tolerance << "\n";
        ss << "\n# FR-11: Real-time mode settings\n";
        ss << config_keys::MAX_REALTIME_ITERATIONS << " = " 
           << config.max_realtime_iterations << "\n";
        ss << config_keys::REALTIME_TIME_LIMIT_MS << " = " 
           << config.realtime_time_limit_ms << "\n";
        ss << "\n# FR-12: Precision mode settings\n";
        ss << config_keys::MAX_PRECISION_ITERATIONS << " = " 
           << config.max_precision_iterations << "\n";
        ss << "\n# Initial state\n";
        ss << config_keys::USE_FLAT_START_DEFAULT << " = " 
           << (config.use_flat_start_default ? "true" : "false") << "\n\n";
        
        ss << "[performance]\n";
        ss << "# Performance tuning options\n";
        ss << config_keys::ENABLE_PROFILING << " = " 
           << (config.enable_profiling ? "true" : "false") << "\n";
        ss << config_keys::USE_CUDA_GRAPHS << " = " 
           << (config.use_cuda_graphs ? "true" : "false") << "\n";
        ss << "# CUDA kernel block sizes (must be power of 2, 32-1024)\n";
        ss << config_keys::BLOCK_SIZE_STANDARD << " = " 
           << config.block_size_standard << "\n";
        ss << config_keys::BLOCK_SIZE_REDUCTION << " = " 
           << config.block_size_reduction << "\n\n";
        
        ss << "[numerical]\n";
        ss << "# Numerical precision and limits\n";
        ss << "# NFR-04: fp32 for real-time, fp64 for precision\n";
        ss << config_keys::USE_DOUBLE_PRECISION << " = " 
           << (config.use_double_precision ? "true" : "false") << "\n";
        ss << "# Voltage limits [p.u.]\n";
        ss << config_keys::VOLTAGE_MIN_PU << " = " << config.voltage_min_pu << "\n";
        ss << config_keys::VOLTAGE_MAX_PU << " = " << config.voltage_max_pu << "\n\n";
        
        ss << "[bad_data]\n";
        ss << "# FR-15: Bad data detection settings\n";
        ss << "# Threshold for normalized residual (typically 3.0 for 99.7% confidence)\n";
        ss << config_keys::BAD_DATA_THRESHOLD << " = " 
           << config.bad_data_threshold << "\n";
        ss << config_keys::AUTO_BAD_DATA_REJECTION << " = " 
           << (config.auto_bad_data_rejection ? "true" : "false") << "\n";
        
        return ss.str();
    }
    
    /**
     * @brief Create default configuration file
     * 
     * @param filepath Output file path
     * @return true if created successfully
     */
    [[nodiscard]] static bool createDefaultConfigFile(const std::string& filepath) {
        EngineConfiguration default_config;
        return saveToFile(filepath, default_config);
    }
    
    /**
     * @brief Get default configuration file path
     * 
     * Returns path relative to executable or working directory.
     */
    [[nodiscard]] static std::string getDefaultConfigPath() {
        return "sle_engine.config";
    }

private:
    /**
     * @brief Trim whitespace from string
     */
    static std::string trim(const std::string& str) {
        const char* whitespace = " \t\r\n";
        size_t start = str.find_first_not_of(whitespace);
        if (start == std::string::npos) return "";
        size_t end = str.find_last_not_of(whitespace);
        return str.substr(start, end - start + 1);
    }
    
    /**
     * @brief Parse boolean value from string
     */
    static bool parseBool(const std::string& value, bool default_val = false) {
        std::string lower = value;
        std::transform(lower.begin(), lower.end(), lower.begin(), 
                      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        
        if (lower == "true" || lower == "yes" || lower == "1" || lower == "on") {
            return true;
        }
        if (lower == "false" || lower == "no" || lower == "0" || lower == "off") {
            return false;
        }
        return default_val;
    }
    
    /**
     * @brief Apply parsed values to configuration structure
     */
    static void applyValues(const std::unordered_map<std::string, std::string>& values,
                           EngineConfiguration& config) {
        // Helper lambdas for safe parsing
        auto getInt = [&values](const char* key, int32_t def) -> int32_t {
            auto it = values.find(key);
            if (it != values.end()) {
                try { return std::stoi(it->second); }
                catch (...) { return def; }
            }
            return def;
        };
        
        auto getSize = [&values](const char* key, size_t def) -> size_t {
            auto it = values.find(key);
            if (it != values.end()) {
                try { return std::stoull(it->second); }
                catch (...) { return def; }
            }
            return def;
        };
        
        auto getReal = [&values](const char* key, Real def) -> Real {
            auto it = values.find(key);
            if (it != values.end()) {
                try { return static_cast<Real>(std::stod(it->second)); }
                catch (...) { return def; }
            }
            return def;
        };
        
        auto getBool = [&values](const char* key, bool def) -> bool {
            auto it = values.find(key);
            if (it != values.end()) {
                return parseBool(it->second, def);
            }
            return def;
        };
        
        // Apply all values (using explicit namespace qualification to avoid 
        // ambiguity with constants of the same name in sle_types.cuh)
        
        config.device_id = getInt(config_keys::DEVICE_ID, config.device_id);
        
        config.max_buses = getSize(config_keys::MAX_BUSES, config.max_buses);
        config.max_branches = getSize(config_keys::MAX_BRANCHES, config.max_branches);
        config.max_measurements = getSize(config_keys::MAX_MEASUREMENTS, config.max_measurements);
        config.max_switching_devices = getSize(config_keys::MAX_SWITCHING_DEVICES, 
                                               config.max_switching_devices);
        
        config.enable_robust_estimation = getBool(config_keys::ENABLE_ROBUST_ESTIMATION, 
                                                   config.enable_robust_estimation);
        config.huber_gamma = getReal(config_keys::HUBER_GAMMA, config.huber_gamma);
        config.convergence_tolerance = getReal(config_keys::CONVERGENCE_TOLERANCE, 
                                               config.convergence_tolerance);
        config.max_realtime_iterations = getInt(config_keys::MAX_REALTIME_ITERATIONS, 
                                                config.max_realtime_iterations);
        config.max_precision_iterations = getInt(config_keys::MAX_PRECISION_ITERATIONS, 
                                                 config.max_precision_iterations);
        config.realtime_time_limit_ms = getReal(config_keys::REALTIME_TIME_LIMIT_MS, 
                                                config.realtime_time_limit_ms);
        config.use_flat_start_default = getBool(config_keys::USE_FLAT_START_DEFAULT, 
                                                config.use_flat_start_default);
        
        config.enable_profiling = getBool(config_keys::ENABLE_PROFILING, config.enable_profiling);
        config.use_cuda_graphs = getBool(config_keys::USE_CUDA_GRAPHS, config.use_cuda_graphs);
        config.block_size_standard = getInt(config_keys::BLOCK_SIZE_STANDARD, 
                                            config.block_size_standard);
        config.block_size_reduction = getInt(config_keys::BLOCK_SIZE_REDUCTION, 
                                             config.block_size_reduction);
        
        config.use_double_precision = getBool(config_keys::USE_DOUBLE_PRECISION, 
                                              config.use_double_precision);
        config.voltage_min_pu = getReal(config_keys::VOLTAGE_MIN_PU, config.voltage_min_pu);
        config.voltage_max_pu = getReal(config_keys::VOLTAGE_MAX_PU, config.voltage_max_pu);
        
        config.bad_data_threshold = getReal(config_keys::BAD_DATA_THRESHOLD, 
                                            config.bad_data_threshold);
        config.auto_bad_data_rejection = getBool(config_keys::AUTO_BAD_DATA_REJECTION, 
                                                  config.auto_bad_data_rejection);
    }
};

} // namespace sle

#endif // SLE_CONFIG_H

