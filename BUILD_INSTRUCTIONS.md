# Build Instructions

Complete guide for building the SLE Engine from source.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Visual Studio Configuration](#visual-studio-configuration)
- [Building the DLL](#building-the-dll)
- [Building Examples](#building-examples)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

1. **Windows 10/11** (64-bit)
2. **Visual Studio 2019 or later**
   - Visual Studio 2022 recommended
   - Desktop development with C++ workload
   - CUDA toolkit integration (usually auto-detected)

3. **NVIDIA CUDA Toolkit 11.0 or later**
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Version 12.6 recommended
   - Ensure CUDA is added to PATH

4. **NVIDIA GPU**
   - Compute Capability 7.0+ (Volta or newer)
   - Recommended: RTX 20/30/40 series, Tesla V100, A100

5. **NuGet Package Manager**
   - Included with Visual Studio
   - Or download separately: https://www.nuget.org/downloads

### Required Libraries

The following are automatically downloaded via NuGet:

- **Boost 1.87.0** - Host-side data structures (stable_vector)

## Quick Start

1. **Clone or extract the project**:
   ```powershell
   cd C:\Users\oleks\Documents\Projects\SLE\SLE
   ```

2. **Restore NuGet packages**:
   ```powershell
   nuget restore SLE.sln
   ```

3. **Open in Visual Studio**:
   - Double-click `SLE.sln`
   - Or: `File > Open > Project/Solution`

4. **Build**:
   - Select `Release x64` configuration
   - Press `F7` or `Build > Build Solution`

5. **Output**:
   - DLL: `bin\Release\SLE.dll`
   - Import library: `bin\Release\SLE.lib`
   - Config: `bin\Release\sle_engine.config` (auto-copied)

## Detailed Setup

### Step 1: Install CUDA Toolkit

1. Download CUDA Toolkit from NVIDIA
2. Run installer and select:
   - ✅ CUDA Toolkit
   - ✅ Visual Studio Integration
   - ✅ Samples (optional)
3. Verify installation:
   ```powershell
   nvcc --version
   ```

### Step 2: Restore NuGet Packages

The project uses NuGet for Boost library:

```powershell
# From project directory
nuget restore SLE.sln

# Or from Visual Studio:
# Right-click solution > Restore NuGet Packages
```

This downloads Boost 1.87.0 to `packages\boost.1.87.0\`.

### Step 3: Verify Include Paths

The project is configured with these include paths:
- `$(ProjectDir)include` - SLE Engine headers
- `$(ProjectDir)packages\boost.1.87.0\lib\native\include` - Boost headers

Verify these paths exist:
```powershell
dir include\sle_api.h
dir packages\boost.1.87.0\lib\native\include\boost
```

## Visual Studio Configuration

### Project Settings

The project is pre-configured with optimal settings:

**Debug Configuration:**
- Optimization: Disabled
- Runtime: Multi-threaded Debug DLL
- CUDA: Debug symbols enabled
- Warnings: Level 3

**Release Configuration:**
- Optimization: Maximum Speed (/O2)
- Runtime: Multi-threaded DLL
- CUDA: Fast math, O3 optimization
- Whole Program Optimization: Enabled
- Buffer Security Check: Disabled (for performance)

### Post-Build Events

The project automatically copies `sle_engine.config` to the output directory after each build.

### CUDA Settings

**Compute Capabilities:**
- `compute_70,sm_70` - Volta
- `compute_75,sm_75` - Turing
- `compute_80,sm_80` - Ampere
- `compute_86,sm_86` - Ampere (laptop)

**CUDA Compiler Options:**
- `-Xcompiler "/openmp"` - OpenMP support
- `--extended-lambda` - Required for device lambdas
- `-std=c++17` - C++17 features
- Fast math enabled in Release
- Line info enabled in Debug

### Language Standards

- **C++**: C++20 (`stdcpp20`)
- **C**: C17 (`stdc17`)

## Building the DLL

### Standard Build

1. **Select Configuration**:
   - `Release x64` (recommended for production)
   - `Debug x64` (for debugging)

2. **Build Solution**:
   - `Build > Build Solution` (F7)
   - Or: Right-click project > Build

3. **Check Output**:
   ```
   bin\Release\
   ├── SLE.dll              # Main DLL
   ├── SLE.lib              # Import library
   ├── SLE.pdb              # Debug symbols (Debug only)
   └── sle_engine.config    # Configuration file (auto-copied)
   ```

### Clean Build

To rebuild from scratch:

```powershell
# Delete intermediate files
Remove-Item -Recurse -Force obj\
Remove-Item -Recurse -Force bin\

# Rebuild
msbuild SLE.sln /t:Rebuild /p:Configuration=Release /p:Platform=x64
```

### Build Output

**Success indicators:**
```
========== Build: 1 succeeded, 0 failed, 0 up-to-date, 0 skipped ==========
```

**Common output files:**
- `SLE.dll` - Dynamic link library (required at runtime)
- `SLE.lib` - Import library (required for linking)
- `SLE.pdb` - Program database (Debug builds)
- `sle_engine.config` - Engine configuration file

## Building Examples

The example (`examples/example_usage.cpp`) uses the C API and can be built separately.

### Option 1: Separate Project

Create a new Visual Studio C++ project:

1. **New Project**:
   - Visual C++ > Windows Desktop > Console Application
   - Name: `example_usage`

2. **Add Source**:
   - Add `examples/example_usage.cpp` to project

3. **Configure**:
   - **Include Directories**:
     - `$(SolutionDir)SLE\include`
   - **Library Directories**:
     - `$(SolutionDir)SLE\bin\$(Configuration)`
   - **Additional Dependencies**:
     - `SLE.lib`

4. **Build**:
   - Ensure SLE.dll is built first
   - Copy `SLE.dll` and `sle_engine.config` to example output directory

### Option 2: Command Line

```powershell
# Compile example
cl /EHsc /I"include" /I"packages\boost.1.87.0\lib\native\include" `
   examples\example_usage.cpp /link SLE.lib /LIBPATH:"bin\Release"

# Copy DLL and config to same directory
Copy-Item bin\Release\SLE.dll examples\
Copy-Item bin\Release\sle_engine.config examples\
```

### Option 3: CMake (Future)

A CMakeLists.txt can be added for cross-platform builds.

## Troubleshooting

### Error: CUDA Toolkit Not Found

**Symptoms:**
```
Cannot find CUDA installation; provide its path via '--cuda-path'
```

**Solution:**
1. Verify CUDA is installed: `nvcc --version`
2. Check Visual Studio CUDA integration:
   - `Tools > Options > Projects and Solutions > VC++ Directories`
   - Ensure CUDA paths are present
3. Restart Visual Studio

### Error: Boost Not Found

**Symptoms:**
```
fatal error C1083: Cannot open include file: 'boost/stable_vector.hpp'
```

**Solution:**
```powershell
# Restore NuGet packages
nuget restore SLE.sln

# Verify Boost location
dir packages\boost.1.87.0\lib\native\include\boost
```

### CUDA 12.x Compatibility Notes

This project is compatible with CUDA 12.x. The following NVCC flags are automatically configured:
- `--extended-lambda` - Required for device lambdas
- `-std=c++17` - Required for C++17 features like structured bindings

The project uses the high-level cuSOLVER sparse Cholesky API (`cusolverSpScsrlsvchol`) which is compatible with CUDA 12.x. The deprecated low-level API (`csrcholInfo_t`) has been replaced.

### Error: LNK2019 Unresolved External

**Symptoms:**
```
error LNK2019: unresolved external symbol "..." referenced in function "..."
```

**Solution:**
1. Check CUDA libraries are linked:
   - `cudart_static.lib`
   - `cusparse.lib`
   - `cusolver.lib`
2. Verify library paths in project settings
3. Ensure Release/Debug configurations match

### Error: DLL Not Found at Runtime

**Symptoms:**
```
The program can't start because SLE.dll is missing
```

**Solution:**
1. Copy `SLE.dll` to:
   - Same directory as executable
   - System PATH directory
2. The config file is auto-copied during build, but can also be manually copied:
   ```powershell
   Copy-Item bin\Release\sle_engine.config <your_app_directory>
   ```

### CUDA Compute Capability Mismatch

**Symptoms:**
```
CUDA error: no kernel image is available for execution
```

**Solution:**
1. Check GPU compute capability:
   ```powershell
   nvidia-smi --query-gpu=compute_cap --format=csv
   ```
2. Ensure project targets your GPU's compute capability
3. Add missing compute capability in project settings

### Out of Memory Errors

**Symptoms:**
```
CUDA error: out of memory
```

**Solution:**
1. Reduce `max_buses`, `max_branches`, `max_measurements` in `sle_engine.config`
2. Use smaller test cases
3. Close other GPU applications
4. Check GPU memory: `nvidia-smi`

## Build Verification

After building, verify the DLL:

```powershell
# Check DLL exports
dumpbin /EXPORTS bin\Release\SLE.dll

# Check dependencies
dumpbin /DEPENDENTS bin\Release\SLE.dll

# Should show:
# - cudart64_XX.dll
# - cusparse64_XX.dll
# - cusolver64_XX.dll

# Verify config was copied
dir bin\Release\sle_engine.config
```

## Performance Optimization

For maximum performance:

1. **Use Release Configuration**:
   - `/O2` optimization
   - Whole Program Optimization
   - Fast math enabled

2. **CUDA Settings**:
   - Fast math (`-use_fast_math`)
   - O3 optimization
   - Appropriate compute capability

3. **Runtime**:
   - Multi-threaded DLL (not static)
   - Link-time code generation

4. **Configuration** (`sle_engine.config`):
   - Set `use_cuda_graphs = true` for production
   - Adjust `max_buses`, `max_branches` to match your network size
   - Use `use_double_precision = false` for real-time mode

## Next Steps

After successful build:

1. **Test the DLL**:
   - Run `examples/example_usage.cpp`
   - Verify IEEE 14-Bus test case

2. **Integration**:
   - See [API Reference](API_REFERENCE.md)
   - See [Examples](EXAMPLES.md)

3. **Configuration**:
   - Edit `sle_engine.config` in output directory
   - See comments in config file for all options
