__kernel void ocl_bitwise_and_kernel(
    const uchar value,
    __global const uchar* src_ptr,
    __global uchar* out_ptr
    )
{
  const int it = get_global_id(0);
  const uchar src = src_ptr[it];
  out_ptr[it] = src & value;
}

__kernel void ocl_lut_kernel(
    const int lut_dim,
    __global const uchar* src_ptr,
    __global const uchar* lut_ptr,
    __global uchar* out_ptr
    )
{
  const int it = get_global_id(0);
  const uchar r = src_ptr[3*it];
  const uchar g = src_ptr[mad24(3, it, 1)];
  const uchar b = src_ptr[mad24(3, it, 2)];
  /* const uchar l = lut_ptr[(int)r + (int)g*lut_dim + (int)b*lut_dim*lut_dim]; */
  const uchar l = lut_ptr[mad24(lut_dim, b, mad24(lut_dim, g, r))];
  out_ptr[it] = l;
}

__kernel void ocl_seg_kernel(
    __global const uchar* lbs_ptr, const int lbs_len,
    __global const uchar* src_ptr, const int src_len,
    __global const uchar* lut_ptr, const int lut_dim,
    __global uchar* lbs_img_ptr,
    __global uchar* out_ptr
    )
{
  const int it = get_global_id(0);
  const uchar r = src_ptr[3*it];
  const uchar g = src_ptr[mad24(3, it, 1)];
  const uchar b = src_ptr[mad24(3, it, 2)];
  /* const uchar l = lut_ptr[(int)r + (int)g*lut_dim + (int)b*lut_dim*lut_dim]; */
  const uchar l = lut_ptr[mad24(lut_dim, b, mad24(lut_dim, g, r))];
  out_ptr[it] = l;
}


/* __kernel void ocl_segment_kernel() {} */
