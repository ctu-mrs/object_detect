#ifndef BALLCANDIDATE_H
#define BALLCANDIDATE_H

#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>
#include <map>

namespace object_detect
{

  /* helper functions for dist_qual_t //{ */
  
  static std::map<std::string, dist_qual_t> dist_qual2id =
  {
    {"no_estimate", dist_qual_t::no_estimate},
    {"blob_size", dist_qual_t::blob_size},
    {"depthmap", dist_qual_t::depthmap},
    {"both", dist_qual_t::both},
  };
  
  dist_qual_t dist_qual_id(std::string name);
  
  std::string dist_qual_name(dist_qual_t id);
  
  //}

  struct BallCandidate
  {
    std::string type;
    cv::Point2d location;
    double radius = std::numeric_limits<double>::quiet_NaN();
    Eigen::Vector3f position = Eigen::Vector3f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
    dist_qual_t dist_qual = unknown_qual;
  };

  cv::Rect circle_roi_clamped(const cv::Point2d& center, const double radius, const cv::Size& limits);

  cv::Mat circle_mask(const cv::Point2d& center, const double radius, const cv::Rect& roi);

  cv::Mat circle_mask(const cv::Point2d& center, const double radius, const cv::Size& size);
}

#endif // BALLCANDIDATE_H
