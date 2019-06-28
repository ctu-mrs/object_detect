__kernel void ocl_lut_kernel(
    const int lut_dim,
    __global const uchar* src_ptr, const int src_step, const int src_offset,
    __global const uchar* lut_ptr, const int lut_step, const int lut_offset,
    __global uchar* out_ptr, const int out_step, const int out_offset
    )
{
  const int it = get_global_id(0);
  const uchar r = src_ptr[3*it + 0];
  const uchar g = src_ptr[3*it + 1];
  const uchar b = src_ptr[3*it + 2];
  const uchar l = lut_ptr[(int)r + (int)g*lut_dim + (int)b*lut_dim*lut_dim];
  out_ptr[it] = l;
}

