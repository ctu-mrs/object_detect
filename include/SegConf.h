#ifndef SEGCONF_H
#define SEGCONF_H

struct SegConf
{
  int color;
  int method;

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

#endif // SEGCONF_H
