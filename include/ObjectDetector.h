#ifndef OBJECT_DETECT_H
#define OBJECT_DETECT_H

/* Includes //{ */
// ROS includes
#include <ros/package.h>
#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <std_msgs/Float32.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <image_geometry/pinhole_camera_model.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <dynamic_reconfigure/server.h>

// Eigen includes
#include <Eigen/Dense>
#include <Eigen/Geometry>

// OpenCV includes
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// STD lib includes
#include <list>
#include <algorithm>

// MRS includes
#include <mrs_lib/ParamLoader.h>
#include <mrs_lib/SubscribeHandler.h>
#include <mrs_lib/DynamicReconfigureMgr.h>
#include <mrs_lib/Profiler.h>

// Includes from this package
#include <object_detect/DetectionParamsConfig.h>
#include <object_detect/ColorChange.h>
#include <object_detect/ColorQuery.h>
#include "SegConf.h"
#include "BlobDetector.h"
#include "utility_fcs.h"

//}

#define cot(x) tan(M_PI_2 - x)

namespace object_detect
{
  // shortcut type to the dynamic reconfigure manager template instance
  typedef mrs_lib::DynamicReconfigureMgr<object_detect::DetectionParamsConfig> drmgr_t;

  // THESE MUST CORRESPOND TO THE VALUES, SPECIFIED IN THE DYNAMIC RECONFIGURE SCRIPT (DetectionParams.cfg)!
  static std::map<std::string, std::pair<color_id_t, cv::Scalar>> colors =
    {
      {"red",    {color_id_t::red, cv::Scalar(0, 0, 128)}},
      {"green",  {color_id_t::green, cv::Scalar(0, 128, 0)}},
      {"blue",   {color_id_t::blue, cv::Scalar(128, 0, 0)}},
      {"yellow", {color_id_t::yellow, cv::Scalar(0, 128, 128)}},
      {"orange", {color_id_t::orange, cv::Scalar(0, 88, 168)}},
    };
  static std::map<std::string, bin_method_t> binname2id =
    {
      {"hsv", bin_method_t::hsv},
      {"lab", bin_method_t::lab},
    };

  /* binarization_method_id() and color_id() helper functions //{ */
  color_id_t color_id(std::string name)
  {
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
    if (colors.find(name) == std::end(colors))
      return color_id_t::unknown_color;
    else
      return colors.at(name).first;
  }

  cv::Scalar color_highlight(color_id_t id)
  {
    for (const auto& keyval : colors)
    {
      if (keyval.second.first == id)
        return keyval.second.second;
    }
    return cv::Scalar(0, 0, 128);
  }

  std::string color_name(color_id_t id)
  {
    for (const auto& keyval : colors)
    {
      if (keyval.second.first == id)
        return keyval.first;
    }
    return "unknown";
  }

  bin_method_t binarization_method_id(std::string name)
  {
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
    if (binname2id.find(name) == std::end(binname2id))
      return bin_method_t::unknown_method;
    else
      return binname2id.at(name);
  }
  //}

  /* //{ class ObjectDetector */

  class ObjectDetector : public nodelet::Nodelet
  {

    private:

      /* pos_cov_t helper struct //{ */
      struct pos_cov_t
      {
        Eigen::Vector3d position;
        Eigen::Matrix3d covariance;
      };
      //}
    
    public:
      ObjectDetector() : m_node_name("ObjectDetector") {};
      virtual void onInit();

    private:
      void main_loop([[maybe_unused]] const ros::TimerEvent& evt);
      void drcfg_update_loop([[maybe_unused]] const ros::TimerEvent& evt);
      SegConf load_segmentation_config(const drcfg_t& cfg);
      SegConf load_segmentation_config(mrs_lib::ParamLoader& pl, const std::string& cfg_name);
      std::vector<SegConf> load_color_configs(mrs_lib::ParamLoader& pl, const std::string& colors_str);
      void highlight_mask(cv::Mat& img, cv::Mat label_img);
      std::vector<SegConf> get_segmentation_configs(const std::vector<SegConf>& all_seg_confs, std::vector<int> color_ids);
      bool color_change_callback(object_detect::ColorChange::Request& req, object_detect::ColorChange::Response& resp);
      bool color_query_callback(object_detect::ColorQuery::Request& req, object_detect::ColorQuery::Response& resp);

    private:
      // --------------------------------------------------------------
      // |                ROS-related member variables                |
      // --------------------------------------------------------------

      /* Parameters, loaded from ROS //{ */
      std::string m_world_frame;
      bool m_object_radius_known;
      double m_object_radius;
      double m_max_dist;
      double m_max_dist_diff;
      double m_min_depth;
      double m_max_depth;
      std::vector<SegConf> m_seg_confs;
      std::vector<SegConf> m_active_seg_confs;
      std::mutex m_active_seg_confs_mtx;
      //}

      /* ROS related variables (subscribers, timers etc.) //{ */
      std::unique_ptr<drmgr_t> m_drmgr_ptr;
      tf2_ros::Buffer m_tf_buffer;
      std::unique_ptr<tf2_ros::TransformListener> m_tf_listener_ptr;

      mrs_lib::SubscribeHandlerPtr<sensor_msgs::ImageConstPtr> m_sh_dm;
      mrs_lib::SubscribeHandlerPtr<sensor_msgs::CameraInfo> m_sh_dm_cinfo;
      mrs_lib::SubscribeHandlerPtr<sensor_msgs::ImageConstPtr> m_sh_rgb;
      mrs_lib::SubscribeHandlerPtr<sensor_msgs::CameraInfo> m_sh_rgb_cinfo;

      ros::Publisher m_pub_pcl;
      ros::Publisher m_pub_debug;

      ros::ServiceServer m_color_change_server;
      ros::ServiceServer m_color_query_server;

      ros::Timer m_main_loop_timer;
      ros::Timer m_drcfg_update_loop_timer;
      //}

    private:
      // --------------------------------------------------------------
      // |                       Other variables                      |
      // --------------------------------------------------------------

      /* Other variables //{ */
      const std::string m_node_name;
      bool m_is_initialized;
      int m_prev_color_id;
      lut_t m_cur_lut;
      std::unique_ptr<mrs_lib::Profiler> m_profiler_ptr;
      image_geometry::PinholeCameraModel m_dm_camera_model;
      image_geometry::PinholeCameraModel m_rgb_camera_model;
      //}
  
    private:
      // --------------------------------------------------------------
      // |                       Helper methods                       |
      // --------------------------------------------------------------

      //{
      //
      void update_drcfg(const SegConf& seg_conf);
      // Checks whether a calculated distance is valid
      bool distance_valid(float distance);
      // Estimates distance of an object based on the 3D vectors pointing to its edges and known distance between those edges
      float estimate_distance_from_known_size(const Eigen::Vector3f& l_vec, const Eigen::Vector3f& r_vec, float known_size);
      // Estimates distance based on information from a depthmap, masked using the binary thresholded image, optionally marks used pixels in the debug image
      float estimate_distance_from_depthmap(const cv::Point2f& area_center, float area_radius, const cv::Mat& dm_img, const cv::Mat& label_img, lut_elem_t color_id, cv::InputOutputArray dbg_img = cv::noArray());
      //}

  };
  
  //}

}  // namespace object_detect

#endif // OBJECT_DETECT_H
