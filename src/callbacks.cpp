#include "callbacks.h"

bool new_rgb = false;
sensor_msgs::Image last_rgb_msg;
void rgb_callback(const sensor_msgs::Image& rgb_msg)
{
  ROS_INFO_THROTTLE(3.0, "Got new RGB camera image");
  last_rgb_msg = rgb_msg;
  new_rgb = true;
}

bool got_rgb_cinfo = false;
image_geometry::PinholeCameraModel rgb_camera_model;
void rgb_cinfo_callback(const sensor_msgs::CameraInfo& rgb_cinfo_msg)
{
  if (!got_rgb_cinfo)
  {
    ROS_INFO("Got RGB camera info");
    rgb_camera_model.fromCameraInfo(rgb_cinfo_msg);
    got_rgb_cinfo = true;
  }
}

bool new_dm = false;
sensor_msgs::ImageConstPtr last_dm_msg;
void dm_callback(const sensor_msgs::ImageConstPtr& dm_msg)
{
  ROS_INFO_THROTTLE(3.0, "Got new depth camera image");
  last_dm_msg = dm_msg;
  new_dm = true;
}

bool got_dm_cinfo = false;
image_geometry::PinholeCameraModel dm_camera_model;
void dm_cinfo_callback(const sensor_msgs::CameraInfo& dm_cinfo_msg)
{
  if (!got_dm_cinfo)
  {
    ROS_INFO("Got depth camera info");
    dm_camera_model.fromCameraInfo(dm_cinfo_msg);
    got_dm_cinfo = true;
  }
}
