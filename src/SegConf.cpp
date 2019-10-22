#include "object_detect/SegConf.h"

namespace object_detect
{

  /* add_lut_hsv() //{ */

  /* class parallelHSVLUT //{ */

  class parallelHSVLUT : public cv::ParallelLoopBody
  {
    public:
        parallelHSVLUT(
            cv::Mat rgb2hsv_lut,
            const double hue_lower, const double hue_higher, const bool overflow,
            const double sat_lower, const double sat_higher,
            const double val_lower, const double val_higher,
            lut_elem_t color_label, lut_t& lut)
          : rgb2hsv_lut(rgb2hsv_lut),
            hue_lower(hue_lower), hue_higher(hue_higher), overflow(overflow),
            sat_lower(sat_lower), sat_higher(sat_higher),
            val_lower(val_lower), val_higher(val_higher),
            color_label(color_label), lut(lut)
        {
        }

        void operator()(const cv::Range &range) const
        {
          for (auto r = range.start; r < range.end; r++)
          {
            for (size_t g = 0; g < lut_dim; g++)
            {
              for (size_t b = 0; b < lut_dim; b++)
              {
                const cv::Vec<uint8_t, 3> color_hsv = rgb2hsv_lut.at<cv::Vec<uint8_t, 3>>(r + lut_dim*g + lut_dim*lut_dim*b);
                const auto cur_h = color_hsv[0];
                const auto cur_s = color_hsv[1];
                const auto cur_v = color_hsv[2];
                const bool h_ok = (!overflow && cur_h > hue_lower && cur_h < hue_higher) || (overflow && (cur_h > hue_lower || cur_h < hue_higher));
                const bool s_ok = cur_s > sat_lower && cur_s < sat_higher;
                const bool v_ok = cur_v > val_lower && cur_v < val_higher;
                if (h_ok && s_ok && v_ok)
                  lut.at(r + lut_dim*g + lut_dim*lut_dim*b) |= color_label;
              }
            }
          }
        }

    private:
        const cv::Mat rgb2hsv_lut;
        const double hue_lower;
        const double hue_higher;
        const bool overflow;
        const double sat_lower;
        const double sat_higher;
        const double val_lower;
        const double val_higher;
        const lut_elem_t color_label;
        lut_t& lut;
  };

  //}

  void add_lut_hsv(lut_t& ret, SegConfPtr seg_conf)
  {
    double hue_lower = seg_conf->hue_center - seg_conf->hue_range / 2.0;
    double hue_higher = seg_conf->hue_center + seg_conf->hue_range / 2.0;
    bool overflow;
    static cv::Mat rgb2hsv_lut;
    if (rgb2hsv_lut.empty())
    {
      rgb2hsv_lut = cv::Mat(1, lut_size, CV_8UC3);
      for (size_t r = 0; r < lut_dim; r++)
      {
        for (size_t g = 0; g < lut_dim; g++)
        {
          for (size_t b = 0; b < lut_dim; b++)
          {
            rgb2hsv_lut.at<cv::Vec<uint8_t, 3>>(r + lut_dim*g + lut_dim*lut_dim*b) = cv::Vec<uint8_t, 3>(r, g, b);
          }
        }
      }
      cv::cvtColor(rgb2hsv_lut, rgb2hsv_lut, cv::COLOR_RGB2HSV);
    }
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

    const double sat_lower = seg_conf->sat_center - seg_conf->sat_range / 2.0;
    const double sat_higher = seg_conf->sat_center + seg_conf->sat_range / 2.0;

    const double val_lower = seg_conf->val_center - seg_conf->val_range / 2.0;
    const double val_higher = seg_conf->val_center + seg_conf->val_range / 2.0;

    parallel_for_(cv::Range(0, lut_dim), parallelHSVLUT(
          rgb2hsv_lut,
          hue_lower, hue_higher, overflow,
          sat_lower, sat_higher,
          val_lower, val_higher,
          seg_conf->color_id, ret));
}

  //}

  /* add_lut_lab //{ */

  /* class parallelLabLUT //{ */

  class parallelLabLUT : public cv::ParallelLoopBody
  {
    public:
        parallelLabLUT(
            cv::Mat rgb2lab_lut,
            const double l_lower, const double l_higher,
            const double a_lower, const double a_higher,
            const double b_lower, const double b_higher,
            lut_elem_t color_label, lut_t& lut)
          : rgb2lab_lut(rgb2lab_lut),
            l_lower(l_lower), l_higher(l_higher),
            a_lower(a_lower), a_higher(a_higher),
            b_lower(b_lower), b_higher(b_higher),
            color_label(color_label), lut(lut)
        {
        }

        void operator()(const cv::Range &range) const
        {
          cv::Vec<uint8_t, 3> color_rgb;
          for (auto r = range.start; r < range.end; r++)
          {
            for (size_t g = 0; g < lut_dim; g++)
            {
              for (size_t b = 0; b < lut_dim; b++)
              {
                const cv::Vec<uint8_t, 3> color_lab = rgb2lab_lut.at<cv::Vec<uint8_t, 3>>(r + lut_dim*g + lut_dim*lut_dim*b);
                const auto cur_l = color_lab[0];
                const auto cur_a = color_lab[1];
                const auto cur_b = color_lab[2];
                const bool l_ok = cur_l > l_lower && cur_l < l_higher;
                const bool a_ok = cur_a > a_lower && cur_a < a_higher;
                const bool b_ok = cur_b > b_lower && cur_b < b_higher;
                if (l_ok && a_ok && b_ok)
                  lut.at(r + lut_dim*g + lut_dim*lut_dim*b) |= color_label;
              }
            }
          }
        }

    private:
        const cv::Mat rgb2lab_lut;
        const double l_lower;
        const double l_higher;
        const double a_lower;
        const double a_higher;
        const double b_lower;
        const double b_higher;
        const lut_elem_t color_label;
        lut_t& lut;
  };

  //}

  void add_lut_lab(lut_t& ret, const SegConfPtr& seg_conf)
  {
    static cv::Mat rgb2lab_lut;
    if (rgb2lab_lut.empty())
    {
      rgb2lab_lut = cv::Mat(1, lut_size, CV_8UC3);
      for (size_t r = 0; r < lut_dim; r++)
      {
        for (size_t g = 0; g < lut_dim; g++)
        {
          for (size_t b = 0; b < lut_dim; b++)
          {
            rgb2lab_lut.at<cv::Vec<uint8_t, 3>>(r + lut_dim*g + lut_dim*lut_dim*b) = cv::Vec<uint8_t, 3>(r, g, b);
          }
        }
      }
      cv::cvtColor(rgb2lab_lut, rgb2lab_lut, cv::COLOR_RGB2Lab);
    }

    const double l_lower = seg_conf->l_center - seg_conf->l_range / 2.0;
    const double l_higher = seg_conf->l_center + seg_conf->l_range / 2.0;

    const double a_lower = seg_conf->a_center - seg_conf->a_range / 2.0;
    const double a_higher = seg_conf->a_center + seg_conf->a_range / 2.0;

    const double b_lower = seg_conf->b_center - seg_conf->b_range / 2.0;
    const double b_higher = seg_conf->b_center + seg_conf->b_range / 2.0;

    parallel_for_(cv::Range(0, lut_dim), parallelLabLUT(
          rgb2lab_lut,
          l_lower, l_higher,
          a_lower, a_higher,
          b_lower, b_higher,
          seg_conf->color_id, ret
        ));
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

  /* generate_lut() //{ */

  lut_t generate_lut(const std::vector<SegConfPtr>& seg_confs)
  {
    lut_t ret;
    ret.resize(lut_size);
    for (size_t it = 0; it < lut_size; it++)
      ret[it] = 0;
    for (const auto& seg_conf : seg_confs)
    {
      switch (seg_conf->method)
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
    return ret;
  }

  //}

  /* lookup_lut() //{ */

  lut_elem_t lookup_lut(cv::InputArray lut, size_t r, size_t g, size_t b)
  {
    /* assert(lut.size() == lut_size); */
    const auto lut_mat = lut.getMat();
    return lut_mat.at<lut_elem_t>(r + lut_dim*g + lut_dim*lut_dim*b);
  }

  //}

}
