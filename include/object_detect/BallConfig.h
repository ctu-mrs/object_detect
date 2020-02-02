#ifndef BALLCONFIG_H
#define BALLCONFIG_H

#include <object_detect/DetectionParamsConfig.h>
#include "object_detect/lut.h"

namespace object_detect
{
  using cfg_t = object_detect::DetectionParamsConfig;
  using ball_params_t = cfg_t::DEFAULT::BALL_PARAMETERS;

  struct BallConfig
  {
    ball_params_t params;
    lutss_t lutss;
  };
}

#endif // BALLCONFIG_H
