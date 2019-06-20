#include "SegConf.h"

using namespace object_detect;

/* add_lut_hsv() //{ */

void add_lut_hsv(lut_t& ret, const SegConf& seg_conf)
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

  const double sat_lower = seg_conf.sat_center - seg_conf.sat_range / 2.0;
  const double sat_higher = seg_conf.sat_center + seg_conf.sat_range / 2.0;

  const double val_lower = seg_conf.val_center - seg_conf.val_range / 2.0;
  const double val_higher = seg_conf.val_center + seg_conf.val_range / 2.0;

  cv::Mat color_rgb;
  cv::Mat color_hsv;
  for (size_t r = 0; r < lut_dim; r++)
  {
    for (size_t g = 0; g < lut_dim; g++)
    {
      for (size_t b = 0; b < lut_dim; b++)
      {
        color_rgb = cv::Mat(cv::Size(1, 1), CV_8UC3, cv::Scalar(r, g, b));
        cv::cvtColor(color_rgb, color_hsv, cv::COLOR_RGB2HSV);
        const cv::Vec<uint8_t, 3> hsv = color_hsv.at<cv::Vec<uint8_t, 3>>(0, 0);
        const auto cur_h = hsv[0];
        const auto cur_s = hsv[1];
        const auto cur_v = hsv[2];
        const bool h_ok = (!overflow && cur_h > hue_lower && cur_h < hue_higher) || (overflow && (cur_h > hue_lower || cur_h < hue_higher));
        const bool s_ok = cur_s > sat_lower && cur_s < sat_higher;
        const bool v_ok = cur_v > val_lower && cur_v < val_higher;
        if (h_ok && s_ok && v_ok)
          ret.at(r + lut_dim*g + lut_dim*lut_dim*b) |= seg_conf.color;
      }
    }
  }
}

//}

/* add_lut_lab //{ */

void add_lut_lab(lut_t& ret, const SegConf& seg_conf)
{
  const double l_lower = seg_conf.l_center - seg_conf.l_range / 2.0;
  const double l_higher = seg_conf.l_center + seg_conf.l_range / 2.0;

  const double a_lower = seg_conf.a_center - seg_conf.a_range / 2.0;
  const double a_higher = seg_conf.a_center + seg_conf.a_range / 2.0;

  const double b_lower = seg_conf.b_center - seg_conf.b_range / 2.0;
  const double b_higher = seg_conf.b_center + seg_conf.b_range / 2.0;

  cv::Mat color_rgb;
  cv::Mat color_lab;
  for (size_t r = 0; r < lut_dim; r++)
  {
    for (size_t g = 0; g < lut_dim; g++)
    {
      for (size_t b = 0; b < lut_dim; b++)
      {
        color_rgb = cv::Mat(cv::Size(1, 1), CV_8UC3, cv::Scalar(r, g, b));
        cv::cvtColor(color_rgb, color_lab, cv::COLOR_RGB2Lab);
        const cv::Vec<uint8_t, 3> lab = color_lab.at<cv::Vec<uint8_t, 3>>(0, 0);
        const auto cur_l = lab[0];
        const auto cur_a = lab[1];
        const auto cur_b = lab[2];
        const bool l_ok = cur_l > l_lower && cur_l < l_higher;
        const bool a_ok = cur_a > a_lower && cur_a < a_higher;
        const bool b_ok = cur_b > b_lower && cur_b < b_higher;
        if (l_ok && a_ok && b_ok)
          ret.at(r + lut_dim*g + lut_dim*lut_dim*b) |= seg_conf.color;
      }
    }
  }
}

//}

/* combine_luts() //{ */

void combine_luts(lut_t& ret, const lut_t& lut1, const lut_t& lut2)
{
  assert(lut1.size() == lut_dim*lut_dim*lut_dim);
  assert(lut2.size() == lut_dim*lut_dim*lut_dim);
  if (ret.size() != lut_dim*lut_dim*lut_dim)
    ret.resize(lut_dim*lut_dim*lut_dim);
  for (size_t it = 0; it < lut_dim*lut_dim*lut_dim; it++)
  {
    const uint8_t val1 = lut1[it];
    const uint8_t val2 = lut2[it];
    ret[it] = val1 | val2;
  }
}

//}

void generate_lut(lut_t& ret, const std::vector<SegConf>& seg_confs)
{
  ret.resize(lut_size);
  for (size_t it = 0; it < lut_size; it++)
    ret[it] = 0;
  for (const auto& seg_conf : seg_confs)
  {
    switch (seg_conf.method)
    {
      case bin_method_t::hsv:
        add_lut_hsv(ret, seg_conf);
        break;
      case bin_method_t::lab:
        add_lut_lab(ret, seg_conf);
        break;
      case bin_method_t::unknown_method:
        std::cerr << "[generate_lut]: Unknown binarization method selected - cannot generate lookup table!";
        break;
    }
  }
}

lut_elem_t lookup_lut(const lut_t& lut, size_t r, size_t g, size_t b)
{
  assert(lut.size() == lut_size);
  return lut[r + lut_dim*g + lut_dim*lut_dim*b];
}
