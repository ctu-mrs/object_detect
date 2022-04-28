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
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
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
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <std_srvs/Trigger.h>

#include <boost/circular_buffer.hpp>

#include <visualanalysis_msgs/TargetLocations2D.h>
#include <visualanalysis_msgs/ROI2D.h>

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
#include <mrs_msgs/PoseWithCovarianceArrayStamped.h>
#include <mrs_lib/param_loader.h>
#include <mrs_lib/subscribe_handler.h>
#include <mrs_lib/dynamic_reconfigure_mgr.h>
#include <mrs_lib/profiler.h>
#include <mrs_lib/geometry/misc.h>
#include <mrs_lib/geometry/cyclic.h>

// Includes from this package
#include <object_detect/DetectionParamsConfig.h>
#include "object_detect/utility_fcs.h"

//}

#define cot(x) tan(M_PI_2 - x)

using radians  = mrs_lib::geometry::radians;
using sradians = mrs_lib::geometry::sradians;

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

  /* //{ class ObjectDetector */

  class ObjectDetector : public nodelet::Nodelet
  {

    using ros_cov_t = mrs_msgs::PoseWithCovarianceArrayStamped::_poses_type::value_type::_covariance_type;

    public:
      ObjectDetector() : m_node_name("ObjectDetector"), m_dm_camera_model_valid(false), m_rgb_camera_model_valid(false) {};
      virtual void onInit();
      bool cbk_regenerate_lut([[maybe_unused]] std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& resp);

    private:
      void main_loop(const sensor_msgs::Image::ConstPtr& rgb_img_msg, const visualanalysis_msgs::TargetLocations2D::ConstPtr& det_msg);
      static ros_cov_t generate_covariance(const Eigen::Vector3f& pos, const double xy_covariance_coeff, const double z_covariance_coeff);
      static Eigen::Matrix3d calc_position_covariance(const Eigen::Vector3d& position_sf, const double xy_covariance_coeff, const double z_covariance_coeff);
      static Eigen::Matrix3d rotate_covariance(const Eigen::Matrix3d& covariance, const Eigen::Matrix3d& rotation);
      mrs_msgs::PoseWithCovarianceIdentified to_message(const Eigen::Vector3f& pos, const dist_qual_t dist_qual) const;

    private:
      // --------------------------------------------------------------
      // |                ROS-related member variables                |
      // --------------------------------------------------------------

      /* Parameters, loaded from ROS //{ */
      std::string m_world_frame;
      double m_max_dist;
      double m_max_dist_diff;
      double m_min_depth;
      double m_max_depth;
      std::string m_mask_filename;
      std::string m_ocl_lut_kernel_file;
      std::map<dist_qual_t, std::pair<double&, double&>> m_cov_coeffs;
      //}

      /* ROS related variables (subscribers, timers etc.) //{ */
      ros::NodeHandle m_nh;

      std::unique_ptr<drmgr_t> m_drmgr_ptr;

      using sync_policy = message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, visualanalysis_msgs::TargetLocations2D>;
      std::unique_ptr<message_filters::Subscriber<sensor_msgs::Image>> m_sub_rgb;
      std::unique_ptr<message_filters::Subscriber<visualanalysis_msgs::TargetLocations2D>> m_sub_det;
      std::unique_ptr<message_filters::Synchronizer<sync_policy>> m_sync;

      mrs_lib::SubscribeHandler<visualanalysis_msgs::TargetLocations2D> m_sh_dets;
      mrs_lib::SubscribeHandler<sensor_msgs::Image> m_sh_dm;
      mrs_lib::SubscribeHandler<sensor_msgs::Image> m_sh_rgb;
      mrs_lib::SubscribeHandler<sensor_msgs::CameraInfo> m_sh_dm_cinfo;
      mrs_lib::SubscribeHandler<sensor_msgs::CameraInfo> m_sh_rgb_cinfo;

      ros::Publisher m_pub_det;
      ros::Publisher m_pub_pcl;
      ros::Publisher m_pub_posearr;
      image_transport::Publisher m_pub_debug;
      image_transport::Publisher m_pub_seg_mask;

      ros::ServiceServer m_srv_regenerate_lut;

      //}

    private:
      // --------------------------------------------------------------
      // |                       Other variables                      |
      // --------------------------------------------------------------

      /* Other variables //{ */
      const std::string m_node_name;
      bool m_is_initialized;
      int m_prev_color_id;
      std::unique_ptr<mrs_lib::Profiler> m_profiler_ptr;
      bool m_dm_camera_model_valid;
      image_geometry::PinholeCameraModel m_dm_camera_model;
      bool m_rgb_camera_model_valid;
      image_geometry::PinholeCameraModel m_rgb_camera_model;

      boost::circular_buffer<sensor_msgs::Image::ConstPtr> m_dm_bfr;

      cv::Mat m_inv_mask;
      //}
  
    private:
      // --------------------------------------------------------------
      // |                       Helper methods                       |
      // --------------------------------------------------------------

      //{
      // Checks whether a calculated distance is valid
      bool distance_valid(float distance);
      // Estimates distance of an object based on the 3D vectors pointing to its center and one of its edges and known distance between the edges
      float estimate_distance_from_known_height(const Eigen::Vector3f& c_vec, const Eigen::Vector3f& b_vec, float physical_diameter);
      // Estimates distance based on information from a depthmap, masked using the binary thresholded image, optionally marks used pixels in the debug image
      float estimate_distance_from_depthmap(const cv::Rect& roi, const double min_valid_ratio, const cv::Mat& dm_img, const cv::Mat& seg_mask, cv::InputOutputArray dbg_img);
      //}

  };
  
  //}

}  // namespace object_detect

#endif // OBJECT_DETECT_H
