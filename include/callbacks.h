#ifndef CALLBACKS_H
#define CALLBACKS_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_geometry/pinhole_camera_model.h>

/* Callbacks //{ */
// Callback for the RGB camera image
void rgb_callback(const sensor_msgs::Image& rgb_msg);

// Callback for the RGB camera info
void rgb_cinfo_callback(const sensor_msgs::CameraInfo& rgb_cinfo_msg);

// Callback for the depth map
void dm_callback(const sensor_msgs::ImageConstPtr& dm_msg);

// Callback for the depthmap camera info
void dm_cinfo_callback(const sensor_msgs::CameraInfo& dm_cinfo_msg);

//}

#endif // CALLBACKS_H
