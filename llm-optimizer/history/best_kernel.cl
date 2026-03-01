__kernel void FIR(__global float* output, __global float* coeff,
                  __global float* input, __global float* history,
                  uint num_tap) {
    // Work-item identifiers
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint lsize = get_local_size(0);

    // Load FIR coefficients into LDS (once per work‑group)
    __local float coeff_l[256];                     // assumes num_tap ≤ 256
    for (uint i = lid; i < num_tap; i += lsize) {
        coeff_l[i] = coeff[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0.0f;

    // Fast path: all required samples are inside the input buffer
    if (gid >= num_tap - 1) {
        // Unrolled loop for better ILP
        #pragma unroll 8
        for (uint i = 0; i < num_tap; ++i) {
            sum += coeff_l[i] * input[gid - i];
        }
    } else {
        // Edge case: need to fetch missing samples from history buffer
        #pragma unroll 8
        for (uint i = 0; i < num_tap; ++i) {
            int idx = (int)gid - (int)i;
            if (idx >= 0) {
                sum += coeff_l[i] * input[(uint)idx];
            } else {
                // idx is negative → fetch from history
                // history is organized so that history[0] corresponds to the oldest sample
                sum += coeff_l[i] * history[num_tap + idx];
            }
        }
    }

    output[gid] = sum;
}