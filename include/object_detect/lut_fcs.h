#ifndef LUT_FCS_H
#define LUT_FCS_H

#include "object_detect/lut.h"
#include "object_detect/BallConfig.h"

namespace object_detect
{

  lut_elem_t lookup_lut(cv::InputArray lut, size_t r, size_t g, size_t b);
  std::optional<lut_t> generate_lut(const BallConfig& cfg);

}

#endif // LUT_FCS_H
