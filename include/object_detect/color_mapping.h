#ifndef COLOR_MAPPING_H
#define COLOR_MAPPING_H

#include <map>
#include <algorithm>

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
  static std::map<std::string, color_id_t> colors =
    {
      {"red",     color_id_t::red},
      {"green",   color_id_t::green},
      {"blue",    color_id_t::blue},
      {"yellow",  color_id_t::yellow},
      {"orange",  color_id_t::orange},
      {"white",   color_id_t::white},
      {"unknown", color_id_t::unknown_color},
    };

  /* binarization_method_id() and color_id() helper functions //{ */
  color_id_t color_id(std::string name);

  std::string color_name(color_id_t id);
  //}

}  // namespace object_detect

#endif // COLOR_MAPPING_H
