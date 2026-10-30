# ROCGEMM

A basic ROCm GEMM kernel/library. Early WIP. 


Current features:

| Piece | File | State |
|---|---|---|
| Naive GEMM kernel + benchmark/validation harness | `stage0_probe.hip` | Working |
| Header-only host helpers (reference GEMM, fills, error metrics) | `include/math.hpp` | Working |
| Runtime / library entry point | `src/runtime.hip` | Stub |



## Requirements

- ROCm with `hipcc` 
- An AMD GPU supported by your ROCm install.
- C++ 17 

## Building

Find your GPU's target name:

```bash
rocminfo | grep -m1 -o 'gfx[0-9a-f]*'
```

Build stage 0 for target:

```bash
hipcc --offload-arch=<gfx target> -std=c++17 -O2 stage0_probe.hip -lamdhip64 -o stage0_probe
```

`-lamdhip64` is needed on some `hipcc` installs that don't link the HIP runtime automatically. It does no harm on installs that do.

## Running

```bash
./stage0_probe [M] [N] [K]     # defaults to 512 512 512
```

The probe prints device info, runs one warm-up launch and 20 timed launches, then compares the GPU result against `math::gemm_ref` on the CPU:

```
=== GEMM (naive) ===
Shape       : 512 x 512 x 512 (M x N x K)
Grid/block  : (32, 32) / (16, 16)
Time        : ... ms/iter
Throughput  : ... TFLOPS
Max abs err : ...
Rel L2 err  : ...
Result      : PASS
```

The CPU reference is a plain triple loop, so validation at large sizes (above about 2048³) takes a while.

## Layout

```
.
├── stage0_probe.hip     # naive kernel, timing, and validation
├── include/
│   └── math.hpp         # host-side reference GEMM, fills, error metrics
└── src/
    └── runtime.hip      # future library entry point (stub)
```

`math.hpp` doesn't depend on HIP outside the small `ceil_div` / `index` helpers, plain C++ tests can include it.

## To Do

- [x] naive kernel, CPU oracle, timing harness
- [ ] shared-memory (LDS) tiling
- [ ] register blocking (several outputs per thread)
- [ ] vectorized global loads, double-buffered LDS
- [ ] RDNA3 WMMA for FP16/BF16 inputs
- [ ] Hardware detection and optimization
- [ ] Turn `src/` into a callable library API

