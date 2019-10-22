#ifndef COLOR_MAPPING_H
#define COLOR_MAPPING_H

#include <map>
#include <opencv2/imgproc/imgproc.hpp>

namespace object_detect
{
  enum color_id_t
  {
    unknown_color = -1,
    red     = ( 0x01 << 0 ),
    green   = ( 0x01 << 1 ),
    blue    = ( 0x01 << 2 ),
    yellow  = ( 0x01 << 3 ),
    orange  = ( 0x01 << 4 ),
    white   = ( 0x01 << 5 ),
  };

  // THESE MUST CORRESPOND TO THE VALUES, SPECIFIED IN THE DYNAMIC RECONFIGURE SCRIPT (DetectionParams.cfg)!
  static std::map<std::string, std::pair<color_id_t, cv::Scalar>> colors =
    {
      {"red",    {color_id_t::red, cv::Scalar(0, 0, 128)}},
      {"green",  {color_id_t::green, cv::Scalar(0, 128, 0)}},
      {"blue",   {color_id_t::blue, cv::Scalar(128, 0, 0)}},
      {"yellow", {color_id_t::yellow, cv::Scalar(0, 128, 128)}},
      {"orange", {color_id_t::orange, cv::Scalar(0, 88, 168)}},
      {"white",  {color_id_t::white, cv::Scalar(128, 0, 128)}},
      {"unknown",{color_id_t::unknown_color, cv::Scalar(88, 88, 88)}},
    };

  /* binarization_method_id() and color_id() helper functions //{ */
  color_id_t color_id(std::string name);

  cv::Scalar color_highlight(color_id_t id);

  std::string color_name(color_id_t id);
  //}

}  // namespace object_detect

#endif // COLOR_MAPPING_H
