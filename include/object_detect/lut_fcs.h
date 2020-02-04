#ifndef LUT_FCS_H
#define LUT_FCS_H

#include "object_detect/lut.h"
#include "object_detect/BallConfig.h"

namespace object_detect
{

  lut_elem_t lookup_lut(cv::InputArray lut, size_t r, size_t g, size_t b);
  lut_elem_t lookup_lut(const lutss_t lutss, size_t x, size_t y, size_t z, size_t xdim = lut_dim, size_t ydim = lut_dim);
  std::optional<lut_t> generate_lut(const BallConfig& cfg);

}

#endif // LUT_FCS_H
