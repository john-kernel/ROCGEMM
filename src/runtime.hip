#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

__global__ void gpu_matmul(float *a, float *b, float *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;


}
