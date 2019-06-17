#ifndef SEGCONF_H
#define SEGCONF_H

#include <vector>
#include <stdint.h>
#include <opencv2/imgproc/imgproc.hpp>

constexpr auto lut_size = 256*256*256;
using lut_t = std::vector<uint8_t>;

struct SegConf
{
  bool active;

  int color;   // id of the color, segmented using this config
  int method;  // id of the segmentation method used
  lut_t lut;

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

lut_t generate_lut(const SegConf& seg_conf);

#endif // SEGCONF_H
