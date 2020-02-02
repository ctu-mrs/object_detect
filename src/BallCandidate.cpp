#include "object_detect/BallCandidate.h"

namespace object_detect
{
  dist_qual_t dist_qual_id(std::string name)
  {
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
    if (dist_qual2id.find(name) == std::end(dist_qual2id))
      return dist_qual_t::unknown_qual;
    else
      return dist_qual2id.at(name);
  }
  
  std::string dist_qual_name(dist_qual_t id)
  {
    for (const auto& keyval : dist_qual2id)
    {
      if (keyval.second == id)
        return keyval.first;
    }
    return "unknown";
  }

  cv::Rect circle_roi_clamped(const cv::Point2d& center, const double radius, const cv::Size& limits)
  {
    cv::Point tmp_topleft(std::floor(center.x - radius), std::floor(center.y - radius));
    cv::Point tmp_botright(std::ceil(center.x + radius), std::ceil(center.y + radius));
    // clamp the dimensions
    tmp_topleft.x = std::clamp(tmp_topleft.x, 0, limits.width);
    tmp_topleft.y = std::clamp(tmp_topleft.y, 0, limits.height);
    tmp_botright.x = std::clamp(tmp_botright.x, 0, limits.width);
    tmp_botright.y = std::clamp(tmp_botright.y, 0, limits.height);
    const cv::Rect roi(tmp_topleft, tmp_botright);
    return roi;
  }

  cv::Mat circle_mask(const cv::Point2d& center, const double radius, const cv::Rect& roi)
  {
    cv::Mat mask(cv::Size(roi.width, roi.height), CV_8UC1);
    mask.setTo(cv::Scalar(0));
    const cv::Point2d offset_center = {center.x - roi.x, center.y - roi.y};
    cv::circle(mask, offset_center, radius, cv::Scalar(255), -1);
    return mask;
  }

  cv::Mat circle_mask(const cv::Point2d& center, const double radius, const cv::Size& size)
  {
    return circle_mask(center, radius, cv::Rect(cv::Point(0, 0), cv::Point(size.width, size.height)));
  }
}
