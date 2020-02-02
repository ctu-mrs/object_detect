#ifndef LUT_H
#define LUT_H

#include <iostream>
#include <vector>
#include <memory>
#include <stdint.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <map>

namespace object_detect
{
  constexpr size_t lut_dim = 256;
  constexpr size_t lut_size = lut_dim*lut_dim*lut_dim;
  using lut_elem_t = uint8_t;
  using lut_t = std::vector<lut_elem_t>;

  struct lutss_t
  {
    lut_t lut;
    int subsampling_x;
    int subsampling_y;
    int subsampling_z;
  };

  enum bin_method_t
  {
    unknown_method = -1,
    hsv = 0,
    lab = 1,
    hs_lut = 2,
    ab_lut = 3,
  };

  /* helper functions for bin_method_t //{ */
  
  static std::map<std::string, bin_method_t> binname2id =
    {
      {"hsv", bin_method_t::hsv},
      {"lab", bin_method_t::lab},
    };
  
  bin_method_t binarization_method_id(std::string name);
  
  std::string binarization_method_name(bin_method_t id);
  
  
  //}

}

#endif // LUT_H
