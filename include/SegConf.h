#ifndef SEGCONF_H
#define SEGCONF_H

#include <vector>
#include <stdint.h>
#include <opencv2/imgproc/imgproc.hpp>

constexpr size_t lut_dim = 256;
constexpr size_t lut_size = lut_dim*lut_dim*lut_dim;
using lut_elem_t = uint8_t;
using lut_t = std::vector<lut_elem_t>;

struct SegConf
{
  bool active;

  int color;   // id of the color, segmented using this config
  std::string color_name;
  int method;  // id of the segmentation method used

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

void generate_lut(lut_t& ret, const std::vector<SegConf>& seg_confs);
lut_elem_t lookup_lut(const lut_t& lut, size_t r, size_t g, size_t b);

#endif // SEGCONF_H
