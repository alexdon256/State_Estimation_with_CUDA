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
 * @file jsf_compliance.h
 * @brief JSF C++ Coding Standard compliance utilities (NFR-24)
 * 
 * This header provides:
 * - Safe assumption macros (NFR-08)
 * - Cache-line alignment utilities (NFR-11)
 * - Branchless computation helpers (NFR-12)
 * - Thread affinity support (NFR-13)
 * - Inlining control attributes (NFR-16)
 * - SIMD/OpenMP pragma helpers (NFR-21, NFR-22)
 * 
 * @note All utilities follow JSF C++ compliance with deterministic behavior.
 */

#ifndef JSF_COMPLIANCE_H
#define JSF_COMPLIANCE_H

#include <cstdint>
#include <cstddef>
#include <cassert>

#ifdef _WIN32
#include <windows.h>
#endif

//=============================================================================
// SECTION 1: Compiler Detection and Feature Macros
//=============================================================================

#if defined(_MSC_VER)
    #define SLE_COMPILER_MSVC 1
    #define SLE_COMPILER_VERSION _MSC_VER
#elif defined(__GNUC__)
    #define SLE_COMPILER_GCC 1
    #define SLE_COMPILER_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100)
#elif defined(__clang__)
    #define SLE_COMPILER_CLANG 1
    #define SLE_COMPILER_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100)
#endif

// C++23 [[assume]] support check
#if __cplusplus >= 202302L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202302L)
    #define SLE_HAS_CPP23_ASSUME 1
#else
    #define SLE_HAS_CPP23_ASSUME 0
#endif

//=============================================================================
// SECTION 2: NFR-16 - Function Inlining Control
//=============================================================================

/**
 * @brief Force function to be inlined (NFR-16)
 * 
 * Use for small, hot-path functions to eliminate call overhead.
 */
#if defined(SLE_COMPILER_MSVC)
    #define SLE_FORCE_INLINE __forceinline
    #define SLE_NO_INLINE __declspec(noinline)
#elif defined(SLE_COMPILER_GCC) || defined(SLE_COMPILER_CLANG)
    #define SLE_FORCE_INLINE __attribute__((always_inline)) inline
    #define SLE_NO_INLINE __attribute__((noinline))
#else
    #define SLE_FORCE_INLINE inline
    #define SLE_NO_INLINE
#endif

/**
 * @brief Mark function as hot (frequently called)
 * 
 * Hints to compiler for aggressive optimization.
 */
#if defined(SLE_COMPILER_GCC) || defined(SLE_COMPILER_CLANG)
    #define SLE_HOT __attribute__((hot))
    #define SLE_COLD __attribute__((cold))
#else
    #define SLE_HOT
    #define SLE_COLD
#endif

//=============================================================================
// SECTION 3: NFR-08 - Safe Assumption Macros
//=============================================================================

/**
 * @brief Safe assumption macro with runtime validation in debug builds
 * 
 * In Release builds, provides compiler hints for optimization.
 * In Debug builds, validates the assumption at runtime.
 * 
 * @note JSF C++ Rule 129: Assertions shall be used to verify invariants.
 * 
 * Usage:
 *   SLE_ASSUME(n > 0);
 *   SLE_ASSUME(ptr != nullptr);
 *   SLE_ASSUME((block_size & (block_size - 1)) == 0);  // power of 2
 */
#ifdef __CUDA_ARCH__
    // CUDA device code: use assert for debug, no-op for release
    #ifdef NDEBUG
        #define SLE_ASSUME(expr) ((void)0)
    #else
        #define SLE_ASSUME(expr) assert(expr)
    #endif
#elif defined(NDEBUG)
    // Release: Use compiler-specific assume intrinsics
    #if SLE_HAS_CPP23_ASSUME
        #define SLE_ASSUME(expr) [[assume(expr)]]
    #elif defined(SLE_COMPILER_MSVC)
        #define SLE_ASSUME(expr) __assume(expr)
    #elif defined(SLE_COMPILER_GCC) && SLE_COMPILER_VERSION >= 130000
        #define SLE_ASSUME(expr) __attribute__((assume(expr)))
    #elif defined(__has_builtin) && __has_builtin(__builtin_assume)
        #define SLE_ASSUME(expr) __builtin_assume(expr)
    #elif defined(SLE_COMPILER_CLANG)
        #define SLE_ASSUME(expr) __builtin_assume(expr)
    #else
        // Fallback: unreachable hint if assumption is false
        #define SLE_ASSUME(expr) do { if (!(expr)) __builtin_unreachable(); } while(0)
    #endif
#else
    // Debug: Validate assumption at runtime
    #define SLE_ASSUME(expr) assert(expr)
#endif

/**
 * @brief CUDA kernel precondition check (host-side only)
 * 
 * Use before kernel launch to validate parameters.
 * In Release, provides compiler hints. In Debug, validates at runtime.
 */
#define SLE_KERNEL_PRECONDITION(expr) SLE_ASSUME(expr)

/**
 * @brief Safe assumption for loop bounds (common case)
 * 
 * Ensures n > 0 which helps loop vectorization.
 */
#define SLE_ASSUME_POSITIVE(n) SLE_ASSUME((n) > 0)

/**
 * @brief Safe assumption for power-of-2 values (alignment/block sizes)
 */
#define SLE_ASSUME_POWER_OF_2(n) SLE_ASSUME(((n) & ((n) - 1)) == 0 && (n) > 0)

/**
 * @brief Safe assumption for pointer validity
 */
#define SLE_ASSUME_VALID_PTR(ptr) SLE_ASSUME((ptr) != nullptr)

/**
 * @brief Safe assumption for aligned pointers
 */
#define SLE_ASSUME_ALIGNED(ptr, alignment) \
    SLE_ASSUME(reinterpret_cast<std::uintptr_t>(ptr) % (alignment) == 0)

//=============================================================================
// SECTION 4: NFR-11 - Cache-Line Alignment
//=============================================================================

/// Standard cache line size (64 bytes on most modern CPUs)
constexpr std::size_t SLE_CACHE_LINE_SIZE = 64;

/**
 * @brief Align a type to cache line boundary (NFR-11)
 * 
 * Prevents false sharing in multi-threaded scenarios.
 */
#define SLE_CACHE_ALIGNED alignas(SLE_CACHE_LINE_SIZE)

/**
 * @brief Pad structure to prevent false sharing
 * 
 * Add to end of frequently-modified fields in shared structures.
 */
#define SLE_PADDING(name) char name[SLE_CACHE_LINE_SIZE]

/**
 * @brief Check if pointer is cache-aligned
 */
SLE_FORCE_INLINE bool is_cache_aligned(const void* ptr) noexcept {
    return (reinterpret_cast<std::uintptr_t>(ptr) % SLE_CACHE_LINE_SIZE) == 0;
}

/**
 * @brief Round up size to next cache line boundary
 */
SLE_FORCE_INLINE constexpr std::size_t align_to_cache_line(std::size_t size) noexcept {
    return (size + SLE_CACHE_LINE_SIZE - 1) & ~(SLE_CACHE_LINE_SIZE - 1);
}

//=============================================================================
// SECTION 5: NFR-12 - Branchless Computation Helpers
//=============================================================================

namespace sle {
namespace branchless {

/**
 * @brief Branchless minimum (NFR-12)
 * 
 * Avoids branch misprediction in tight loops.
 */
template<typename T>
SLE_FORCE_INLINE constexpr T min(T a, T b) noexcept {
    return b ^ ((a ^ b) & -(a < b));
}

/**
 * @brief Branchless maximum (NFR-12)
 */
template<typename T>
SLE_FORCE_INLINE constexpr T max(T a, T b) noexcept {
    return a ^ ((a ^ b) & -(a < b));
}

/**
 * @brief Branchless absolute value for signed integers
 */
template<typename T>
SLE_FORCE_INLINE constexpr T abs(T x) noexcept {
    static_assert(std::is_signed<T>::value, "Use with signed types only");
    T mask = x >> (sizeof(T) * 8 - 1);
    return (x + mask) ^ mask;
}

/**
 * @brief Branchless clamp to range [lo, hi]
 */
template<typename T>
SLE_FORCE_INLINE constexpr T clamp(T x, T lo, T hi) noexcept {
    return min(max(x, lo), hi);
}

/**
 * @brief Branchless conditional select: (cond ? a : b)
 * 
 * @param cond Boolean condition
 * @param a Value if true
 * @param b Value if false
 */
template<typename T>
SLE_FORCE_INLINE constexpr T select(bool cond, T a, T b) noexcept {
    return b ^ ((a ^ b) & -static_cast<T>(cond));
}

/**
 * @brief Branchless sign function: returns -1, 0, or 1
 */
template<typename T>
SLE_FORCE_INLINE constexpr T sign(T x) noexcept {
    return static_cast<T>((x > 0) - (x < 0));
}

/**
 * @brief Branchless positive part: max(x, 0)
 */
template<typename T>
SLE_FORCE_INLINE constexpr T positive_part(T x) noexcept {
    return x & ~(x >> (sizeof(T) * 8 - 1));
}

} // namespace branchless
} // namespace sle

//=============================================================================
// SECTION 6: NFR-13 - Thread Affinity Support
//=============================================================================

namespace sle {
namespace threading {

/**
 * @brief Set thread affinity to a specific CPU core (NFR-13)
 * 
 * Reduces cache misses and context switching overhead.
 * 
 * @param core_id CPU core index (0-based)
 * @return true if successful, false otherwise
 * 
 * @note JSF C++ compliant: No exceptions, explicit error handling.
 */
[[nodiscard]] SLE_FORCE_INLINE bool set_thread_affinity(int core_id) noexcept {
#ifdef _WIN32
    DWORD_PTR mask = 1ULL << core_id;
    DWORD_PTR result = SetThreadAffinityMask(GetCurrentThread(), mask);
    return result != 0;
#elif defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0;
#else
    (void)core_id;
    return false;  // Platform not supported
#endif
}

/**
 * @brief Get current thread's CPU core (NFR-13)
 * 
 * @return Current core ID, or -1 if unavailable
 */
[[nodiscard]] SLE_FORCE_INLINE int get_current_core() noexcept {
#ifdef _WIN32
    return static_cast<int>(GetCurrentProcessorNumber());
#elif defined(__linux__)
    return sched_getcpu();
#else
    return -1;
#endif
}

/**
 * @brief Get number of available CPU cores
 */
[[nodiscard]] SLE_FORCE_INLINE int get_num_cores() noexcept {
#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return static_cast<int>(sysinfo.dwNumberOfProcessors);
#elif defined(__linux__)
    return static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
#else
    return 1;
#endif
}

/**
 * @brief Set thread priority to high for real-time processing
 * 
 * @return true if successful
 */
[[nodiscard]] SLE_FORCE_INLINE bool set_high_priority() noexcept {
#ifdef _WIN32
    return SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST) != 0;
#elif defined(__linux__)
    struct sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    return pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) == 0;
#else
    return false;
#endif
}

} // namespace threading
} // namespace sle

//=============================================================================
// SECTION 7: NFR-21/22 - OpenMP SIMD Pragma Helpers
//=============================================================================

/**
 * @brief Parallel for loop with SIMD vectorization (NFR-21, NFR-22)
 * 
 * Usage:
 *   SLE_PARALLEL_FOR
 *   for (int i = 0; i < n; ++i) { ... }
 */
#ifdef _OPENMP
    #define SLE_PARALLEL_FOR _Pragma("omp parallel for")
    #define SLE_PARALLEL_FOR_SIMD _Pragma("omp parallel for simd")
    #define SLE_SIMD _Pragma("omp simd")
    #define SLE_SIMD_REDUCTION(op, var) _Pragma("omp simd reduction(" #op ":" #var ")")
    #define SLE_PARALLEL_REDUCTION(op, var) _Pragma("omp parallel for reduction(" #op ":" #var ")")
#else
    #define SLE_PARALLEL_FOR
    #define SLE_PARALLEL_FOR_SIMD
    #define SLE_SIMD
    #define SLE_SIMD_REDUCTION(op, var)
    #define SLE_PARALLEL_REDUCTION(op, var)
#endif

/**
 * @brief Schedule hint for parallel loops
 * 
 * dynamic: variable work per iteration
 * static: equal work per iteration  
 * guided: decreasing chunk sizes
 */
#ifdef _OPENMP
    #define SLE_PARALLEL_FOR_DYNAMIC _Pragma("omp parallel for schedule(dynamic)")
    #define SLE_PARALLEL_FOR_STATIC _Pragma("omp parallel for schedule(static)")
    #define SLE_PARALLEL_FOR_GUIDED _Pragma("omp parallel for schedule(guided)")
#else
    #define SLE_PARALLEL_FOR_DYNAMIC
    #define SLE_PARALLEL_FOR_STATIC
    #define SLE_PARALLEL_FOR_GUIDED
#endif

//=============================================================================
// SECTION 8: NFR-23 - Loop Unrolling Hints
//=============================================================================

/**
 * @brief Hint compiler to unroll loop (NFR-23)
 * 
 * Usage:
 *   SLE_UNROLL(4)
 *   for (int i = 0; i < 16; ++i) { ... }
 */
#if defined(SLE_COMPILER_MSVC)
    // MSVC uses #pragma loop(hint_parallel(n))
    #define SLE_UNROLL(n) __pragma(loop(hint_parallel(n)))
    #define SLE_UNROLL_FULL
#elif defined(SLE_COMPILER_GCC) || defined(SLE_COMPILER_CLANG)
    #define SLE_UNROLL(n) _Pragma("GCC unroll " #n)
    #define SLE_UNROLL_FULL _Pragma("GCC unroll 16")
#else
    #define SLE_UNROLL(n)
    #define SLE_UNROLL_FULL
#endif

//=============================================================================
// SECTION 9: JSF C++ Compliance Helpers
//=============================================================================

/**
 * @brief Mark function as not throwing (JSF Rule 149)
 * 
 * JSF C++ prohibits exceptions in most code. This macro documents
 * and enforces the no-throw guarantee.
 */
#define SLE_NOEXCEPT noexcept

/**
 * @brief Explicit type conversion helper (JSF Rule 180)
 * 
 * JSF requires explicit type conversions to be visible.
 */
template<typename To, typename From>
SLE_FORCE_INLINE constexpr To explicit_cast(From value) SLE_NOEXCEPT {
    return static_cast<To>(value);
}

/**
 * @brief Bounds-checked array access for debug builds
 * 
 * In release builds, compiles to direct access.
 */
template<typename T, std::size_t N>
SLE_FORCE_INLINE T& safe_at(T (&arr)[N], std::size_t index) SLE_NOEXCEPT {
    SLE_ASSUME(index < N);
    return arr[index];
}

/**
 * @brief Suppress unused variable warnings (JSF Rule 1)
 */
#define SLE_UNUSED(x) (void)(x)

/**
 * @brief Mark variable as intentionally uninitialized for performance
 * 
 * Use sparingly and only when immediately overwritten.
 */
#if defined(SLE_COMPILER_MSVC)
    #define SLE_UNINITIALIZED __pragma(warning(suppress: 26495))
#else
    #define SLE_UNINITIALIZED
#endif

//=============================================================================
// SECTION 10: Restrict Pointer Qualifier
//=============================================================================

/**
 * @brief Restrict pointer qualifier for alias optimization
 * 
 * Tells compiler that pointers don't alias, enabling vectorization.
 */
#if defined(SLE_COMPILER_MSVC)
    #define SLE_RESTRICT __restrict
#elif defined(SLE_COMPILER_GCC) || defined(SLE_COMPILER_CLANG)
    #define SLE_RESTRICT __restrict__
#else
    #define SLE_RESTRICT
#endif

#endif // JSF_COMPLIANCE_H

