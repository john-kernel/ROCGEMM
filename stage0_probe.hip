// stage0_probe.hip
// Basic GEMM: C = alpha * (A @ B) + beta * C on the GPU, validated against a
// CPU reference GEMM provided by include/math.hpp.
//
// Build:
//   hipcc --offload-arch=gfx1101 -std=c++17 -O2 stage0_probe.hip -lamdhip64 -o stage0_probe
//   (some hipcc installs do not auto-link the HIP runtime; -lamdhip64 is
//    harmless to keep even when they do)
// Run:
//   ./stage0_probe [M] [N] [K]
//
// The kernel is intentionally naive: one thread computes one output element.
// It is a correctness baseline, not a performance kernel.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "include/math.hpp"

#define HIP_CHECK(expr)                                                        \
    do {                                                                       \
        hipError_t e_ = (expr);                                                \
        if (e_ != hipSuccess) {                                                \
            std::fprintf(stderr, "HIP error: %s\n  at %s:%d\n  in %s\n",       \
                         hipGetErrorString(e_), __FILE__, __LINE__, #expr);    \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Device kernel
//
// A is MxK, B is KxN, C is MxN, all row-major and tightly packed. Each thread
// computes C[row, col] by walking the full inner dimension K. This is the
// simplest possible GEMM; its performance is bounded by global-memory traffic
// because A and B are re-read from DRAM for every output element.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void gemm_naive_kernel(const T* __restrict__ A,
                                  const T* __restrict__ B,
                                  T* __restrict__ C,
                                  int M,
                                  int N,
                                  int K,
                                  T alpha,
                                  T beta) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    }

    T acc = T(0);
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

// ---------------------------------------------------------------------------
// Host-side runner: allocates, launches, times, and validates the kernel.
// ---------------------------------------------------------------------------
template <typename T>
void run_gemm(int M, int N, int K, T alpha, T beta) {
    const std::size_t size_a = static_cast<std::size_t>(M) * K;
    const std::size_t size_b = static_cast<std::size_t>(K) * N;
    const std::size_t size_c = static_cast<std::size_t>(M) * N;

    std::vector<T> h_A(size_a);
    std::vector<T> h_B(size_b);
    std::vector<T> h_C(size_c);
    std::vector<T> h_ref(size_c);

    math::fill_uniform(h_A.data(), size_a, 0xABCDu);
    math::fill_uniform(h_B.data(), size_b, 0x1234u);
    math::fill_constant(h_C.data(), size_c, T(0));
    math::fill_constant(h_ref.data(), size_c, T(0));

    // CPU oracle.
    math::gemm_ref<T>(M, N, K, h_A.data(), h_B.data(), h_ref.data(), alpha, beta);

    T *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, size_a * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_B, size_b * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_C, size_c * sizeof(T)));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_a * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_b * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_C, 0, size_c * sizeof(T)));

    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned>(math::ceil_div(N, block.x)),
                    static_cast<unsigned>(math::ceil_div(M, block.y)));

    // Warm-up: the first launch includes one-time module/context costs.
    gemm_naive_kernel<T><<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    HIP_CHECK(hipEventCreate(&t0));
    HIP_CHECK(hipEventCreate(&t1));

    const int iters = 20;
    HIP_CHECK(hipEventRecord(t0));
    for (int i = 0; i < iters; ++i) {
        gemm_naive_kernel<T><<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    HIP_CHECK(hipEventRecord(t1));
    HIP_CHECK(hipEventSynchronize(t1));

    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));

    HIP_CHECK(hipMemcpy(h_C.data(), d_C, size_c * sizeof(T), hipMemcpyDeviceToHost));

    const double sec = static_cast<double>(ms) / 1e3 / iters;
    const double flops = 2.0 * static_cast<double>(M) * N * K;
    const double tflops = flops / sec / 1e12;

    const T max_err = math::max_abs_error(h_C.data(), h_ref.data(), size_c);
    const T rel_err = math::rel_l2_error(h_C.data(), h_ref.data(), size_c);
    const T tol = static_cast<T>(1e-3);
    const bool pass = max_err < tol;

    std::printf("=== GEMM (naive) ===\n");
    std::printf("Shape       : %d x %d x %d (M x N x K)\n", M, N, K);
    std::printf("Grid/block  : (%u, %u) / (%u, %u)\n",
                grid.x, grid.y, block.x, block.y);
    std::printf("Time        : %.3f ms/iter\n", sec * 1e3);
    std::printf("Throughput  : %.2f TFLOPS\n", tflops);
    std::printf("Max abs err : %g (tol %g)\n", static_cast<double>(max_err),
                static_cast<double>(tol));
    std::printf("Rel L2 err  : %g\n", static_cast<double>(rel_err));
    std::printf("Result      : %s\n\n", pass ? "PASS" : "FAIL");

    HIP_CHECK(hipEventDestroy(t0));
    HIP_CHECK(hipEventDestroy(t1));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
}

static void print_device_info() {
    int dev = 0;
    HIP_CHECK(hipGetDevice(&dev));

    hipDeviceProp_t p;
    HIP_CHECK(hipGetDeviceProperties(&p, dev));

    std::printf("=== Device %d ===\n", dev);
    std::printf("Name        : %s\n", p.name);
    std::printf("Arch        : %s\n", p.gcnArchName);
    std::printf("Compute CUs : %d\n", p.multiProcessorCount);
    std::printf("Wavefront   : %d\n", p.warpSize);
    std::printf("L2 cache    : %d KB\n", p.l2CacheSize / 1024);
    std::printf("\n");
}

int main(int argc, char** argv) {
    int M = 512;
    int N = 512;
    int K = 512;

    if (argc > 1) M = std::atoi(argv[1]);
    if (argc > 2) N = std::atoi(argv[2]);
    if (argc > 3) K = std::atoi(argv[3]);

    print_device_info();
    run_gemm<float>(M, N, K, 1.0f, 0.0f);
    return 0;
}
