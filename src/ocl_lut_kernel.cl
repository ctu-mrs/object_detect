__kernel void ocl_lut_kernel(
    __global const uchar* src_ptr, int src_step, int src_offset, int src_rows, int src_cols,
    __global const uchar* lut_ptr, int lut_step, int lut_offset, int lut_rows, int lut_cols,
    __global uchar* out_ptr, int out_step, int out_offset, int out_rows, int out_cols,
    )
{

  __local CT smem[LOCAL_SIZE];
  __local float localmem_max[WGS2_ALIGNED];
  __local uint localmem_maxloc[WGS2_ALIGNED];

  const int x = get_global_id(0);
  const int y = get_group_id(1);
}
