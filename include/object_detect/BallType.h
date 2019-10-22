#ifndef BALLTYPE_H
#define BALLTYPE_H

#include "object_detect/SegConf.h"
#include <memory>

namespace object_detect
{
  struct BallType
  {
    SegConfPtr seg_conf = nullptr;
    double physical_radius = std::numeric_limits<double>::quiet_NaN();

    bool unknown()
    {
      return seg_conf == nullptr || seg_conf->color_id == color_id_t::unknown_color;
    }
  };

  using BallTypePtr = std::shared_ptr<BallType>;
}

#endif // BALLTYPE_H

