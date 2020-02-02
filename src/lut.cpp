#include "object_detect/lut.h"

namespace object_detect
{

  bin_method_t binarization_method_id(std::string name)
  {
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
    if (binname2id.find(name) == std::end(binname2id))
      return bin_method_t::unknown_method;
    else
      return binname2id.at(name);
  }

  std::string binarization_method_name(bin_method_t id)
  {
    for (const auto& keyval : binname2id)
    {
      if (keyval.second == id)
        return keyval.first;
    }
    return "unknown";
  }

}
