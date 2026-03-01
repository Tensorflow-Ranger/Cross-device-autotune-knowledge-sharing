
__kernel void FIR(__global float* output, __global float* coeff,
                  __global float* input, __global float* history, 
                  uint num_tap) {
  uint tid = get_global_id(0);
  uint num_data = get_global_size(0);

  float sum = 0;
  uint i = 0;
  for (i = 0; i < num_tap; i++) {
    if (tid >= i) {
        sum = sum + coeff[i] * input[tid - i];
    } else {
        sum = sum + coeff[i] * history[num_tap - (i - tid)];
    }
  }
  output[tid] = sum;

  /*barrier(CLK_GLOBAL_MEM_FENCE);*/

  /*[> fill the history buffer <]*/
  /*if (tid >= numData - numTap + 1)*/
    /*temp_input[tid - (numData - numTap + 1)] = temp_input[xid];*/

  /*barrier(CLK_GLOBAL_MEM_FENCE);*/
}
