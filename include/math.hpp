// math.hpp
// Reusable, header-only linear-algebra / math helpers for HIP GEMM work.
//
// The host-only helpers (gemm_ref, fill_*, error metrics) are deliberately
// kept free of any HIP dependency so they can also be used from plain C++
// test code. Only the tiny indexing helpers are marked host/device.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>


#if defined(__HIPCC__)
  #include <hip/hip_runtime.h>
  #define MATH_HOST_DEVICE __host__ __device__
#elif defined(__CUDACC__)
  #define MATH_HOST_DEVICE __host__ __device__
#else
  #define MATH_HOST_DEVICE
#endif

namespace math {



// Ceiling integer division, useful for grid dimensions.
MATH_HOST_DEVICE inline std::size_t ceil_div(std::size_t n, std::size_t d) {
    return (n + d - 1) / d;
}

// Row-major linear index for a matrix element with leading dimension ld.
MATH_HOST_DEVICE inline std::size_t index(std::size_t row,
                                          std::size_t col,
                                          std::size_t ld) {
    return row * ld + col;
}

//Matrix Helpers
//
template <typename T>
inline void fill_uniform(T* data, std::size_t n, std::uint32_t seed = 0xC0FFEEu) {
    std::uint32_t s = seed;
    const T scale = T(1) / T(std::numeric_limits<std::uint32_t>::max());
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        data[i] = (T(2) * T(s) * scale) - T(1);
    }
}

// Fill [data, data + n) with a constant value.
template <typename T>
inline void fill_constant(T* data, std::size_t n, T value = T(0)) {
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = value;
    }
}

//CPU validation against kernel
template <typename T>
inline void gemm_ref(std::size_t M,
                     std::size_t N,
                     std::size_t K,
                     T alpha,
                     const T* A, std::size_t lda,
                     const T* B, std::size_t ldb,
                     T beta,
                     T* C, std::size_t ldc) {
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            T acc = T(0);
            for (std::size_t k = 0; k < K; ++k) {
                acc += A[index(i, k, lda)] * B[index(k, j, ldb)];
            }
            C[index(i, j, ldc)] = alpha * acc + beta * C[index(i, j, ldc)];
        }
    }
}

// Convenience overload for tightly packed row-major matrices
// (A is MxK, B is KxN, C is MxN).
template <typename T>
inline void gemm_ref(std::size_t M,
                     std::size_t N,
                     std::size_t K,
                     const T* A,
                     const T* B,
                     T* C,
                     T alpha = T(1),
                     T beta = T(0)) {
    gemm_ref<T>(M, N, K, alpha, A, K, B, N, beta, C, N);
}


// Maximum absolute difference between two arrays.
template <typename T>
inline T max_abs_error(const T* a, const T* b, std::size_t n) {
    T max_err = T(0);
    for (std::size_t i = 0; i < n; ++i) {
        const T err = std::fabs(a[i] - b[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

// Relative L2 error: ||a - b||_2 / (||b||_2 + tiny).
template <typename T>
inline T rel_l2_error(const T* a, const T* b, std::size_t n) {
    T num = T(0);
    T den = T(0);
    for (std::size_t i = 0; i < n; ++i) {
        const T d = a[i] - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    return std::sqrt(num) / (std::sqrt(den) + T(1e-30));
}

}  // namespace math

#undef MATH_HOST_DEVICE
