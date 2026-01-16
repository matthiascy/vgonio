// Kahan summation for improved numerical stability.
__device__ __forceinline__ void kahan_add(double x, double& sum, double& c) {
    double y = x - c;
    double t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

// Reduce sum of an array with NaN handling and scaling.
extern "C" __global__ void reduce_blocks(
    const float* __restrict__ xs,
    double* __restrict__ block_sums,  // one per block
    size_t n)
{
    const size_t tid = threadIdx.x;
    const size_t gtid = blockIdx.x * blockDim.x + tid;
    const size_t stride = gridDim.x * blockDim.x;

    // Grid-stride over the input array, unrolled by 2 for bandwidth
    double sum = 0.0;
    double c = 0.0; // Kahan compensation
    for (size_t i = gtid * 2; i < n; i += stride * 2) {
        const float a = xs[i];
        if (a == a) { // check for NaN
            kahan_add((double)(a), sum, c);
        }

        size_t j = i + 1;
        if (j < n) {
            const float b = xs[j];
            if (b == b) { // check for NaN
                kahan_add((double)(b), sum, c);
            }
        }
    }
    double local = sum + c;

    // In-block reduction in double
    extern __shared__ double s[];  // size = blockDim.x * sizeof(double)
    s[tid] = local;
    __syncthreads();

    for (unsigned step = blockDim.x >> 1; step >= 32; step >>= 1) {
        if (tid < step) s[tid] += s[tid + step];
        __syncthreads();
    }

    double v = 0.0;
    if (tid < 32) {
        v = s[tid];
        unsigned mask = 0xffffffffu;
        v += __shfl_down_sync(mask, v, 16);
        v += __shfl_down_sync(mask, v, 8);
        v += __shfl_down_sync(mask, v, 4);
        v += __shfl_down_sync(mask, v, 2);
        v += __shfl_down_sync(mask, v, 1);
        if (tid == 0) block_sums[blockIdx.x] = v; // no factor yet
    }
}

// Final reduction of block sums
extern "C" __global__ void reduce(
    const double* __restrict__ block_sums,
    double* __restrict__ result,
    size_t num_blocks,
    double factor) // multiply the final result by this factor
{
    const size_t tid = threadIdx.x;
    const size_t gtid = blockIdx.x * blockDim.x + tid;
    const size_t stride = gridDim.x * blockDim.x;

    double sum = 0.0;
    double c = 0.0;
    for (size_t i = gtid; i < num_blocks; i += stride) {
        kahan_add(block_sums[i], sum, c);
    }
    double local = sum + c;

    extern __shared__ double s[];
    s[tid] = local;
    __syncthreads();

    for (unsigned step = blockDim.x >> 1; step >= 32; step >>= 1) {
        if (tid < step) s[tid] += s[tid + step];
        __syncthreads();
    }

    double v = s[tid];
    if (tid < 32) {
        unsigned mask = 0xffffffffu;
        v += __shfl_down_sync(mask, v, 16);
        v += __shfl_down_sync(mask, v, 8);
        v += __shfl_down_sync(mask, v, 4);
        v += __shfl_down_sync(mask, v, 2);
        v += __shfl_down_sync(mask, v, 1);
        if (tid == 0) *result = v * factor;
    }
}
