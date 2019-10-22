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
#include <mrs_lib/subscribe_handler.h>
#include <mrs_lib/DynamicReconfigureMgr.h>
#include <mrs_lib/Profiler.h>

// Includes from this package
#include <object_detect/PoseWithCovarianceArrayStamped.h>
#include <object_detect/DetectionParamsConfig.h>
#include <object_detect/ColorChange.h>
#include <object_detect/ColorQuery.h>
#include "object_detect/color_mapping.h"
#include "object_detect/SegConf.h"
#include "object_detect/BlobDetector.h"
#include "object_detect/utility_fcs.h"

//}

#define cot(x) tan(M_PI_2 - x)

namespace object_detect
{
  // shortcut type to the dynamic reconfigure manager template instance
  typedef mrs_lib::DynamicReconfigureMgr<object_detect::DetectionParamsConfig> drmgr_t;

  enum dist_qual_t
  {
    unknown_qual = -1,
    no_estimate = 0,
    blob_size = 1,
    depthmap = 2,
    both = 3,
  };

  static std::map<std::string, bin_method_t> binname2id =
    {
      {"hsv", bin_method_t::hsv},
      {"lab", bin_method_t::lab},
    };
  static std::map<std::string, dist_qual_t> dist_qual2id =
  {
    {"no_estimate", dist_qual_t::no_estimate},
    {"blob_size", dist_qual_t::blob_size},
    {"depthmap", dist_qual_t::depthmap},
    {"both", dist_qual_t::both},
  };

  /* binarization_method_id() and color_id() helper functions //{ */
  bin_method_t binarization_method_id(std::string name)
  {
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
    if (binname2id.find(name) == std::end(binname2id))
      return bin_method_t::unknown_method;
    else
      return binname2id.at(name);
  }

  std::string binarization_method_name(bin_method_t id)
  {
    for (const auto& keyval : binname2id)
    {
      if (keyval.second == id)
        return keyval.first;
    }
    return "unknown";
  }

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
  //}

  /* //{ class ObjectDetector */

  class ObjectDetector : public nodelet::Nodelet
  {

    private:

      using ros_poses_t = object_detect::PoseWithCovarianceArrayStamped::_poses_type;
      using ros_pose_t = ros_poses_t::value_type::_pose_type;
      using ros_cov_t = ros_poses_t::value_type::_covariance_type;

    public:
      ObjectDetector() : m_node_name("ObjectDetector") {};
      virtual void onInit();

    private:
      void main_loop([[maybe_unused]] const ros::TimerEvent& evt);
      void drcfg_update_loop([[maybe_unused]] const ros::TimerEvent& evt);
      ros_poses_t generate_poses(const std::vector<geometry_msgs::Point32>& positions, const std::vector<dist_qual_t>& distance_qualities);
      static ros_cov_t generate_covariance(const geometry_msgs::Point32& pos, const double xy_covariance_coeff, const double z_covariance_coeff);
      static Eigen::Matrix3d calc_position_covariance(const Eigen::Vector3d& position_sf, const double xy_covariance_coeff, const double z_covariance_coeff);
      static Eigen::Matrix3d rotate_covariance(const Eigen::Matrix3d& covariance, const Eigen::Matrix3d& rotation);
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
      std::string m_ocl_lut_kernel_file;
      std::vector<SegConf> m_seg_confs;
      std::mutex m_active_seg_confs_mtx;
      std::vector<SegConf> m_active_seg_confs;
      std::map<dist_qual_t, std::pair<double, double>> m_cov_coeffs;
      //}

      /* ROS related variables (subscribers, timers etc.) //{ */
      std::unique_ptr<drmgr_t> m_drmgr_ptr;
      tf2_ros::Buffer m_tf_buffer;
      std::unique_ptr<tf2_ros::TransformListener> m_tf_listener_ptr;

      mrs_lib::SubscribeHandlerPtr<sensor_msgs::Image> m_sh_dm;
      mrs_lib::SubscribeHandlerPtr<sensor_msgs::CameraInfo> m_sh_dm_cinfo;
      mrs_lib::SubscribeHandlerPtr<sensor_msgs::Image> m_sh_rgb;
      mrs_lib::SubscribeHandlerPtr<sensor_msgs::CameraInfo> m_sh_rgb_cinfo;

      ros::Publisher m_pub_det;
      ros::Publisher m_pub_pcl;
      image_transport::Publisher m_pub_debug;

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
      BlobDetector m_blob_det;
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