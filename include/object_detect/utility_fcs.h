#ifndef UTILITY_FCS_H
#define UTILITY_FCS_H

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <image_geometry/pinhole_camera_model.h>
#include <opencv2/imgproc/imgproc.hpp>

/** Utility functions //{**/

Eigen::Vector3f project(float px_x, float px_y, const image_geometry::PinholeCameraModel& camera_model);

float depth2range(float d, float u, float v, float fx, float fy, float cx, float cy);

void mark_pixel(cv::Mat& img, int u, int v, uint8_t color = 0, uint8_t amount = 100);

//}

#endif // UTILITY_FCS_H
