#ifndef SEGCONF_H
#define SEGCONF_H

#include <iostream>
#include <vector>
#include <memory>
#include <stdint.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "object_detect/color_mapping.h"
#include <object_detect/DetectionParamsConfig.h>

namespace object_detect
{
  using drcfg_t = object_detect::DetectionParamsConfig;

  constexpr size_t lut_dim = 256;
  constexpr size_t lut_size = lut_dim*lut_dim*lut_dim;
  using lut_elem_t = uint8_t;
  using lut_t = std::vector<lut_elem_t>;

  enum bin_method_t
  {
    unknown_method = -1,
    hsv = 0,
    lab = 1,
  };

  lut_t generate_lut(const drcfg_t& drcfg);
  lut_elem_t lookup_lut(cv::InputArray lut, size_t r, size_t g, size_t b);

}

#endif // SEGCONF_H
