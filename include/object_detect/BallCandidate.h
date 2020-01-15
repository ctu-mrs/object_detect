#ifndef BALLDETECTION_H
#define BALLDETECTION_H

#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>

namespace object_detect
{
  enum dist_qual_t
  {
    unknown_qual = -1,
    no_estimate = 0,
    blob_size = 1,
    depthmap = 2,
    both = 3,
  };

  struct BallCandidate
  {
    int type;
    cv::Point2d location;
    double radius = std::numeric_limits<double>::quiet_NaN();
    Eigen::Vector3f position = Eigen::Vector3f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
    dist_qual_t dist_qual = unknown_qual;
  };

  cv::Rect circle_roi_clamped(const cv::Point2d& center, const double radius, const cv::Size& limits);

  cv::Mat circle_mask(const cv::Point2d& center, const double radius, const cv::Rect& roi);

  cv::Mat circle_mask(const cv::Point2d& center, const double radius, const cv::Size& size);
}

#endif // BALLDETECTION_H
