__device__ __forceinline__ size_t global_tid() {
    return (size_t)blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ float sqr(const float x) {
    return __fmaf_rn(x, x, 0.0f); // x * x with FMA
}

// Element-wise difference of two arrays: result[i] = xs[i] - ys[i]
extern "C" __global__ void difference(const float *__restrict__ xs,
                                      const float *__restrict__ ys,
                                      float *__restrict__ result, size_t n) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t tid = global_tid(); tid < n; tid += stride) {
        result[tid] = xs[tid] - ys[tid];
    }
}

// Element-wise squared difference of two arrays: result[i] = (xs[i] - ys[i])^2
extern "C" __global__ void difference_sqr(const float *__restrict__ xs,
                                          const float *__restrict__ ys,
                                          float *__restrict__ result,
                                          size_t n) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t tid = global_tid(); tid < n; tid += stride) {
        const float diff = xs[tid] - ys[tid];
        result[tid] = sqr(diff);
    }
}

// Element-wise squared difference of two arrays if the arrays are 16-byte
// aligned
extern "C" __global__ void difference_sqr_vec4(const float4 *__restrict__ xs,
                                               const float4 *__restrict__ ys,
                                               float4 *__restrict__ result,
                                               size_t n_vec4) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t tid = global_tid(); tid < n_vec4; tid += stride) {
        const float4 a = xs[tid];
        const float4 b = ys[tid];
        float4 r;
        float d;

        d = a.x - b.x;
        r.x = __fmaf_rn(d, d, 0.0f);
        d = a.y - b.y;
        r.y = __fmaf_rn(d, d, 0.0f);
        d = a.z - b.z;
        r.z = __fmaf_rn(d, d, 0.0f);
        d = a.w - b.w;
        r.w = __fmaf_rn(d, d, 0.0f);

        result[tid] = r;
    }
}

// Squared difference after log1p(weight * x)
//    If weight_stride is power of two, shift >= 0 and pass mask=-1.
//    Else set shift=-1 and mask=weight_stride (for division); OR pre-expand
//    weights.
extern "C" __global__ void
difference_sqr_lncos(const float *__restrict__ xs, const float *__restrict__ ys,
                     const float *__restrict__ weights,
                     float *__restrict result, const unsigned int weight_stride,
                     size_t n) {
    // Detect power-of-two stride for fast indexing
    int shift = -1;
    if (weight_stride > 0 && (weight_stride & (weight_stride - 1)) == 0) {
        // weight_stride is power of two: compute shift = log2(weight_stride)
        shift = __ffs(weight_stride) - 1; // log2(weight_stride)
    }

    for (size_t tid = global_tid(); tid < n; tid += gridDim.x * blockDim.x) {
        int widx =
            (shift >= 0) ? (int)(tid >> shift) : (int)(tid / weight_stride);

        const float w = weights[widx];

        // log1p(w * value)
        const float x = log1pf(xs[tid] * w);
        const float y = log1pf(ys[tid] * w);
        const float diff = x - y;
        result[tid] = sqr(diff);
    }
}
