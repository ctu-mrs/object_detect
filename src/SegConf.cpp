#include "SegConf.h"

void fill_lut_hsv(lut_t& ret, const SegConf& seg_conf)
{
  double hue_lower = seg_conf.hue_center - seg_conf.hue_range / 2.0;
  double hue_higher = seg_conf.hue_center + seg_conf.hue_range / 2.0;
  bool overflow;
  /* calculate the correct bounds for the pixel HSV values //{ */
  {
    overflow = false;
    if (hue_lower < 0)
    {
      hue_lower += 180;
      overflow = true;
    }
    if (hue_higher > 179)
    {
      hue_higher -= 180;
      overflow = true;
    }
  }
  //}

  double sat_lower = seg_conf.sat_center - seg_conf.sat_range / 2.0;
  double sat_higher = seg_conf.sat_center + seg_conf.sat_range / 2.0;

  double val_lower = seg_conf.val_center - seg_conf.val_range / 2.0;
  double val_higher = seg_conf.val_center + seg_conf.val_range / 2.0;

  cv::Mat color_rgb;
  cv::Mat color_hsv;
  for (int r = 0; r < 256; r++)
  {
    for (int g = 0; g < 256; g++)
    {
      for (int b = 0; b < 256; b++)
      {
        color_rgb = cv::Mat(cv::Size(1, 1), CV_8UC3, cv::Scalar(r, g, b));
        cv::cvtColor(color_rgb, color_hsv, cv::COLOR_RGB2HSV);
        const cv::Vec<uint8_t, 3> hsv = color_hsv.at<cv::Vec<uint8_t, 3>>(0, 0);
        const auto cur_h = hsv[0];
        const auto cur_s = hsv[0];
        const auto cur_v = hsv[0];
        const bool h_ok = (!overflow && cur_h > hue_lower && cur_h < hue_higher) || (overflow && (cur_h > hue_lower || cur_h < hue_higher));
        const bool s_ok = cur_s > sat_lower && cur_s < sat_higher;
        const bool v_ok = cur_v > val_lower && cur_v < val_higher;
        if (h_ok && s_ok && v_ok)
          ret.at(r + 256*g + 256*256*b) = seg_conf.color;
        else
          ret.at(r + 256*g + 256*256*b) = 0;
      }
    }
  }
}

void fill_lut_lab(lut_t& ret, const SegConf& seg_conf)
{

}

void fill_lut(lut_t& ret, const SegConf& seg_conf)
{
  ret.resize(lut_size);
  switch (seg_conf.method)
  {
    case 0:
      fill_lut_hsv(ret, seg_conf);
      break;
    case 1:
      fill_lut_lab(ret, seg_conf);
      break;
  }
}
