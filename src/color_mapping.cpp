#include "object_detect/color_mapping.h"

namespace object_detect
{

  /* color_id_t helper functions //{ */
  color_id_t color_id(std::string name)
  {
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
    if (colors.find(name) == std::end(colors))
      return color_id_t::unknown_color;
    else
      return colors.at(name);
  }

  std::string color_name(color_id_t id)
  {
    for (const auto& keyval : colors)
    {
      if (keyval.second == id)
        return keyval.first;
    }
    return "unknown";
  }
  //}

}  // namespace object_detect
