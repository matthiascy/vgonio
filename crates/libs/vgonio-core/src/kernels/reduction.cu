extern "C" __global__ void reduce(const float *__restrict xs, float *__restrict out, const float factor) {
    extern __shared__ float partial[];

    // Load elements AND do first add of reduction
    // Vector now 2x as long as the number of threads, so scale i
    const unsigned i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Store first partial result instead of just the elements
    const float x = isnan(xs[i])? 0.0f : xs[i];
    const float y = isnan(xs[i + blockDim.x])? 0.0f : xs[i + blockDim.x];
    partial[threadIdx.x] = x * factor  + y * factor;
    __syncthreads();

    // Start at 1/2 block stride and divide by two each iteration
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial[threadIdx.x] += partial[threadIdx.x + s];
        }
        __syncthreads();
    }

    // The Result is indexed by this block
    if (threadIdx.x == 0) {
        out[blockIdx.x] = partial[0];
    }
}
