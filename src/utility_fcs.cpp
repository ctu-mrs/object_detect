#include "utility_fcs.h"

Eigen::Vector3d project(float px_x, float px_y, const image_geometry::PinholeCameraModel& camera_model)
{
  cv::Point2f det_pt(px_x, px_y);
  if (!camera_model.distortionCoeffs().empty())
    det_pt = camera_model.rectifyPoint(det_pt);  // do not forget to rectify the points! (not necessary for Realsense)
  cv::Point3d cv_vec = camera_model.projectPixelTo3dRay(det_pt);
  return Eigen::Vector3d(cv_vec.x, cv_vec.y, cv_vec.z).normalized();
}

float depth2range(float d, float u, float v, float fx, float fy, float cx, float cy)
{
  float x = (u - cx)/fx; // this is the x coordinate of the pixel, projected to a plane at d = 1m
  float y = (v - cy)/fy; // this is the y coordinate of the pixel, projected to a plane at d = 1m
  return d*sqrt(x*x + y*y + 1.0);
}

void mark_pixel(cv::Mat& img, int u, int v, uint8_t color, uint8_t amount)
{
  assert(color < img.channels());
  cv::Vec3b& c = img.at<cv::Vec3b>(v, u);
  c[color] = int(c[color]) + int(amount) > std::numeric_limits<uint8_t>::max() ? std::numeric_limits<uint8_t>::max() : c[color] + amount;
  /* img.at<cv::Vec3b>(u, v) = c; */
}
