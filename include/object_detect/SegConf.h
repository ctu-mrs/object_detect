#ifndef SEGCONF_H
#define SEGCONF_H

#include <iostream>
#include <vector>
#include <memory>
#include <stdint.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "object_detect/color_mapping.h"

namespace object_detect
{
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

  struct SegConf
  {
    bool active;

    color_id_t color_id;   // id of the color, segmented using this config
    std::string color_name;
    double physical_size;

    bin_method_t method;  // id of the segmentation method used

    // HSV thresholding parameters
    double hue_center;
    double hue_range;
    double sat_center;
    double sat_range;
    double val_center;
    double val_range;

    // L*a*b* thresholding parameters
    double l_center;
    double l_range;
    double a_center;
    double a_range;
    double b_center;
    double b_range;
  };

  using SegConfPtr = std::shared_ptr<SegConf>;

  lut_t generate_lut(const std::vector<SegConfPtr>& seg_confs);
  lut_elem_t lookup_lut(const lut_t& lut, size_t r, size_t g, size_t b);

}

#endif // SEGCONF_H
