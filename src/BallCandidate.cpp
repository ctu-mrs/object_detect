#include "object_detect/BallCandidate.h"

namespace object_detect
{
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
