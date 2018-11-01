#include "object_detect.h"

using namespace cv;
using namespace std;
using namespace object_detect;
using namespace Eigen;
using namespace mrs_lib;

// shortcut type to the dynamic reconfigure manager template instance
typedef DynamicReconfigureMgr<object_detect::DetectionParamsConfig> drmgr_t;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "object_detect");
  ROS_INFO("Node initialized.");

  ros::NodeHandle nh = ros::NodeHandle("~");

  /** Load parameters from ROS * //{*/
  string node_name = ros::this_node::getName().c_str();
  ParamLoader pl(nh);
  // LOAD STATIC PARAMETERS
  ROS_INFO("Loading static parameters:");
  // Load the detection parameters
  double object_radius = pl.load_param2<double>("object_radius");
  double max_dist = pl.load_param2<double>("max_dist");
  double max_dist_diff = pl.load_param2<double>("max_dist_diff");
  uint16_t min_depth = pl.load_param2<int>("min_depth");
  uint16_t max_depth = pl.load_param2<int>("max_depth");

  // LOAD DYNAMIC PARAMETERS
  drmgr_t drmgr(nh);

  if (!drmgr.loaded_successfully())
  {
    ROS_ERROR("Some default values of dynamically reconfigurable parameters were not loaded successfully, ending the node");
    ros::shutdown();
    return 1;
  }

  if (!pl.loaded_successfully())
  {
    ROS_ERROR("Some compulsory parameters were not loaded successfully, ending the node");
    ros::shutdown();
    return 1;
  }
  //}

  /** Create publishers and subscribers //{**/
  tf2_ros::Buffer tf_buffer;
  // Initialize transform listener
  tf2_ros::TransformListener tf_listener(tf_buffer);
  // Initialize other subs and pubs
  SubscribeMgr smgr(nh);
  SubscribeHandlerPtr<sensor_msgs::ImageConstPtr> sh_dm = smgr.create_handler<sensor_msgs::ImageConstPtr>("dm_image", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
  SubscribeHandlerPtr<sensor_msgs::CameraInfo> sh_dm_cinfo = smgr.create_handler<sensor_msgs::CameraInfo>("dm_camera_info", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
  SubscribeHandlerPtr<sensor_msgs::ImageConstPtr> sh_rgb = smgr.create_handler<sensor_msgs::ImageConstPtr>("rgb_image", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
  SubscribeHandlerPtr<sensor_msgs::CameraInfo> sh_rgb_cinfo = smgr.create_handler<sensor_msgs::CameraInfo>("rgb_camera_info", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));

  ros::Publisher debug_pub = nh.advertise<sensor_msgs::Image&>("debug_image", 1);
  ros::Publisher pcl_pub = nh.advertise<sensor_msgs::PointCloud>("detected_objects", 10);
  //}

  cout << "----------------------------------------------------------" << std::endl;

  image_geometry::PinholeCameraModel dm_camera_model;
  image_geometry::PinholeCameraModel rgb_camera_model;

  ros::Rate r(100);
  double old_fps = 0.0l;
  while (ros::ok())
  {
    ros::spinOnce();

    if (sh_dm_cinfo->has_data() && !sh_dm_cinfo->used_data())
      dm_camera_model.fromCameraInfo(sh_dm_cinfo->get_data());
    if (sh_rgb_cinfo->has_data() && !sh_rgb_cinfo->used_data())
      rgb_camera_model.fromCameraInfo(sh_rgb_cinfo->get_data());

    bool publish_debug = debug_pub.getNumSubscribers() > 0;

    // Check if we got all required messages
    if (sh_dm->new_data() && sh_dm_cinfo->used_data() && sh_rgb->new_data() && sh_rgb_cinfo->used_data())
    {
      ros::Time start_t = ros::Time::now();
      const sensor_msgs::ImageConstPtr dm_img_msg = sh_dm->get_data();
      cv_bridge::CvImageConstPtr dm_img_ros = cv_bridge::toCvShare(dm_img_msg, sensor_msgs::image_encodings::TYPE_16UC1);
      const cv::Mat dm_img(dm_img_ros->image);
      const sensor_msgs::ImageConstPtr rgb_img_msg = sh_rgb->get_data();
      cv_bridge::CvImageConstPtr rgb_img_ros = cv_bridge::toCvShare(rgb_img_msg, sensor_msgs::image_encodings::BGR8);
      const cv::Mat rgb_img = rgb_img_ros->image;
      cv::Mat thresholded_img;

      cv::Mat dbg_img;
      if (publish_debug)
      {
        cv::cvtColor(rgb_img, dbg_img, cv::COLOR_BGR2GRAY);
        cv::cvtColor(dbg_img, dbg_img, cv::COLOR_GRAY2BGR);
      }

      BlobDetector blb_det(drmgr.config);
      const vector<Blob> blobs = blb_det.detect(rgb_img, thresholded_img);

      /* Calculate relative and absolute position of the object //{ */
      const size_t n_dets = blobs.size();
      vector<geometry_msgs::Point32> positions;
      for (size_t it = 0; it < n_dets; it++)
      {
        ROS_INFO("[%s]: Processing object %lu/%lu --------------------------", ros::this_node::getName().c_str(), it + 1, n_dets);
        bool dist_valid = true;
        const Blob& blob = blobs.at(it);
        const cv::Point center = blob.location;
        const double radius = blob.radius;

        if (publish_debug)
          cv::circle(dbg_img, center, radius, Scalar(0, 255, 0), 2);

        /* First calculate the estimated distance //{ */
        const Eigen::Vector3d l_vec = project(center.x - radius*cos(M_PI_4), center.y - radius*sin(M_PI_4), rgb_camera_model);
        const Eigen::Vector3d r_vec = project(center.x + radius*cos(M_PI_4), center.y + radius*sin(M_PI_4), rgb_camera_model);
        const double alpha = acos(l_vec.dot(r_vec)) / 2.0;
        double est_dist = object_radius * sin(M_PI_2 - alpha) * (tan(alpha) + cot(alpha));
        cout << "Estimated distance: " << est_dist << std::endl;
        if (isnan(est_dist) || est_dist < 0.0 || est_dist > max_dist)
        {
          dist_valid = false;
          est_dist = est_dist < 0.0 ? 0.0 : max_dist;
          cout << "Invalid estimated distance, cropping to " << est_dist << "m" << std::endl;
        }
        //}
        
        /* Compare the estimated distance to distance reported by the depth camera //{ */
        {
          float dm_dist = 0;
          size_t n_dm_samples = 0;
          Point2f center_in_tmp(radius, radius);
          Point tmp_topleft(center.x - radius, ceil(center.y - radius));
          Point tmp_botright(center.x + radius, ceil(center.y + radius));
          // clamp the x dimension
          if (tmp_topleft.x < 0)
          {
            center_in_tmp.x -= tmp_topleft.x;
            tmp_topleft.x = 0;
          } else if (tmp_botright.x >= dm_img.cols)
          {
            center_in_tmp.x += (tmp_botright.x - dm_img.cols);
            tmp_botright.x = dm_img.cols-1;
          }
          // clamp the y dimension
          if (tmp_topleft.y < 0)
          {
            center_in_tmp.y -= tmp_topleft.y;
            tmp_topleft.y = 0;
          } else if (tmp_botright.y >= dm_img.rows)
          {
            center_in_tmp.y += (tmp_botright.y - dm_img.rows);
            tmp_botright.y = dm_img.rows-1;
          }
          const cv::Rect roi(tmp_topleft, tmp_botright);
          const cv::Mat tmp_dm_img = dm_img(roi);
          const cv::Mat tmp_mask = thresholded_img(roi);
          cv::Mat tmp_dbg_img = publish_debug ? dbg_img(roi) : cv::Mat();
          const Size size = tmp_dm_img.size();

          // average over all pixels in the area of the detected blob
          // go through all pixels in a square of size 2*radius
          for (int x = 0; x < size.width; x++)
          {
            for (int y = 0; y < size.height; y++)
            {
              // check if this pixel is part of a detected blob
              if (!tmp_mask.at<uint8_t>(y, x))
                continue;
              const uint16_t depth = tmp_dm_img.at<uint16_t>(y, x);
              // skip invalid measurements
              if (depth <= min_depth
               || depth >= max_depth)
                continue;
              const float u = tmp_topleft.x + x;
              const float v = tmp_topleft.y + y;
              dm_dist += depth2range(depth, u, v, dm_camera_model.fx(), dm_camera_model.fy(), dm_camera_model.cx(), dm_camera_model.cy());
              n_dm_samples++;
              if (publish_debug)
                mark_pixel(tmp_dbg_img, x, y, 2);
            }
          }
          if (n_dm_samples == 0)
          {
            cout << "Estimated distance: " << est_dist << "m, depthmap distance unavailable" << std::endl;
            cout << "-- skipping object" << std::endl;
            dist_valid = false;
          } else
          {
            // calculate average from the sum
            dm_dist /= float(n_dm_samples);
            // recalculate to meters from mm
            dm_dist /= 1000.0;

            cout << "Estimated distance: " << est_dist << "m, depthmap distance: " << dm_dist << "m" << std::endl;
            if (abs(dm_dist - est_dist) > max_dist_diff)
            {
              cout << "-- distances are too different, skipping object" << std::endl;
              dist_valid = false;
            }
            // it is presumed that the depth camera offers better precision
            // distance estimation, use it
            est_dist = dm_dist;
          }
        }
        //}

        if (dist_valid)
        {
          /* Calculate the estimated position of the object //{ */
          const Eigen::Vector3d pos_vec = est_dist * (l_vec + r_vec) / 2.0;
          cout << "Estimated location (camera CS): [" << pos_vec(0) << ", " << pos_vec(1) << ", " << pos_vec(2) << "]" << std::endl;
          //}

          /* If all is OK, add the position to the vector //{ */
          geometry_msgs::Point32 cur_pos;
          cur_pos.x = pos_vec(0);
          cur_pos.y = pos_vec(1);
          cur_pos.z = pos_vec(2);
          positions.push_back(cur_pos);
          //}
        }

        ROS_INFO("[%s]: Done with object %lu/%lu ---------------------------", ros::this_node::getName().c_str(), it + 1, n_dets);
      }
      //}

      /* Publish all the calculated valid positions //{ */
      sensor_msgs::PointCloud pcl_msg;
      pcl_msg.header = rgb_img_msg->header;
      pcl_msg.points = positions;
      pcl_pub.publish(pcl_msg);
      //}

      if (publish_debug)
      {
        cv_bridge::CvImage dbg_img_ros(rgb_img_msg->header, sensor_msgs::image_encodings::BGR8, dbg_img);
        sensor_msgs::ImagePtr dbg_img_msg;
        dbg_img_msg = dbg_img_ros.toImageMsg();
        dbg_img_msg->header = rgb_img_msg->header;
        debug_pub.publish(dbg_img_msg);
      }

      cout << "Image processed" << std::endl;

      ros::Duration dur = ros::Time::now() - start_t;
      ros::Duration del = ros::Time::now() - rgb_img_msg->header.stamp;
      double cur_fps = 0.2 * (1.0 / dur.toSec()) + 0.8 * old_fps;
      cout << "processing FPS: " << cur_fps << "Hz" << std::endl;
      cout << "delay: " << del.toSec() * 1000.0 << "ms" << std::endl;
      old_fps = cur_fps;
    } else
    {
      r.sleep();
    }
  }
}
