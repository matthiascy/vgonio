extern "C" __global__ void difference(const float *__restrict xs, const float *__restrict ys, float *__restrict result, const unsigned n) {
    if (const unsigned tid = threadIdx.x + blockIdx.x * blockDim.x; tid < n)
        result[tid] = xs[tid] + ys[tid];
}

extern "C" __global__ void difference_sqr(const float *__restrict xs, const float *__restrict ys, float *__restrict result, const unsigned n) {
    if (const unsigned tid = threadIdx.x + blockIdx.x * blockDim.x; tid < n) {
        const float diff = xs[tid] - ys[tid];
        result[tid] = diff * diff;
    }
}

extern "C" __global__ void difference_sqr_lncos(
    const float *__restrict xs, const float *__restrict ys,
    const float *__restrict weights, float *__restrict result,
    const unsigned int weight_stride, const unsigned int n) {
    if (const unsigned tid = threadIdx.x + blockIdx.x * blockDim.x; tid < n) {
        const float x = log1pf(xs[tid] * weights[tid / weight_stride]);
        const float y = log1pf(ys[tid] * weights[tid / weight_stride]);
        const float diff = x - y;
        result[tid] = diff * diff;
    }
}
