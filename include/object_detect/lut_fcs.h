#include "object_detect/lut.h"
#include "object_detect/BallConfig.h"

namespace object_detect
{

  lut_elem_t lookup_lut(cv::InputArray lut, size_t r, size_t g, size_t b);
  lut_elem_t lookup_lut(const lutss_t lutss, size_t x, size_t y, size_t z);
  std::optional<lut_t> generate_lut(const BallConfig& cfg);

}
