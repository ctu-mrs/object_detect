#include "object_detect/ObjectDetector.h"

using namespace cv;
using namespace std;
using namespace object_detect;
using namespace Eigen;
using namespace mrs_lib;

namespace object_detect
{

  /* main_loop() method //{ */
  void ObjectDetector::main_loop([[maybe_unused]] const ros::TimerEvent& evt)
  {
    /* Initialize the camera models //{ */
    if (m_sh_dm_cinfo.hasMsg() && !m_dm_camera_model_valid)
    {
      m_dm_camera_model.fromCameraInfo(m_sh_dm_cinfo.getMsg());
      if (m_dm_camera_model.fx() == 0.0 || m_dm_camera_model.fy() == 0.0)
        ROS_ERROR_THROTTLE(1.0, "[ObjectDetector]: Depthmap camera model is invalid (fx or fy is zero)!");
      else
        m_dm_camera_model_valid = true;
    }
    if (m_sh_rgb_cinfo.hasMsg() && !m_rgb_camera_model_valid)
    {
      m_rgb_camera_model.fromCameraInfo(m_sh_rgb_cinfo.getMsg());
      if (m_rgb_camera_model.fx() == 0.0 || m_rgb_camera_model.fy() == 0.0)
        ROS_ERROR_THROTTLE(1.0, "[ObjectDetector]: Color camera model is invalid (fx or fy is zero)!");
      else
        m_rgb_camera_model_valid = true;
    }
    //}

    /* load covariance coefficients from dynparam //{ */

    {
      const std::pair<double&, double&> cov_coeffs_no_estimate = {m_drmgr_ptr->config.cov_coeffs__xy__no_estimate, m_drmgr_ptr->config.cov_coeffs__z__no_estimate};
      const std::pair<double&, double&> cov_coeffs_blob_size = {m_drmgr_ptr->config.cov_coeffs__xy__blob_size, m_drmgr_ptr->config.cov_coeffs__z__blob_size};
      const std::pair<double&, double&> cov_coeffs_depthmap = {m_drmgr_ptr->config.cov_coeffs__xy__depthmap, m_drmgr_ptr->config.cov_coeffs__z__depthmap};
      const std::pair<double&, double&> cov_coeffs_both = {m_drmgr_ptr->config.cov_coeffs__xy__both, m_drmgr_ptr->config.cov_coeffs__z__both};
      m_cov_coeffs.insert_or_assign(dist_qual_t::no_estimate, cov_coeffs_no_estimate);
      m_cov_coeffs.insert_or_assign(dist_qual_t::blob_size, cov_coeffs_blob_size);
      m_cov_coeffs.insert_or_assign(dist_qual_t::depthmap, cov_coeffs_depthmap);
      m_cov_coeffs.insert_or_assign(dist_qual_t::both, cov_coeffs_both);
    }

    //}

    const bool rgb_ready = m_sh_rgb.newMsg() && m_rgb_camera_model_valid;
    bool dm_ready = m_sh_dm.newMsg() && m_dm_camera_model_valid;

    // Check if we got all required messages
    if (m_sh_dets.newMsg() && rgb_ready)
    {
      /* Copy values from subscribers to local variables //{ */
      const ros::WallTime start_t = ros::WallTime::now();

      const auto det_msg = m_sh_dets.getMsg();
      const auto& detections = det_msg.detections;

      cv::Mat dm_img;
      if (dm_ready)
      {
        const sensor_msgs::ImageConstPtr dm_img_msg = m_sh_dm.getMsg();
        const cv_bridge::CvImageConstPtr dm_img_ros = cv_bridge::toCvShare(dm_img_msg, sensor_msgs::image_encodings::TYPE_16UC1);
        dm_img = dm_img_ros->image;
        if (!m_inv_mask.empty() && (dm_img.cols != m_inv_mask.cols || dm_img.rows != m_inv_mask.rows))
        {
          NODELET_WARN_THROTTLE(0.5, "[ObjectDetector]: Received depthmap dimensions (%d x %d) are different from the loaded mask (%d x %d)! Cannot continue, not using depthmap.", dm_img.rows, dm_img.cols, m_inv_mask.rows, m_inv_mask.cols);
          dm_ready = false;
        }
      }

      const sensor_msgs::ImageConstPtr rgb_img_msg = m_sh_rgb.getMsg();
      const cv_bridge::CvImageConstPtr rgb_img_ros = cv_bridge::toCvShare(rgb_img_msg, sensor_msgs::image_encodings::BGR8);
      const cv::Mat rgb_img = rgb_img_ros->image;
      if (!m_inv_mask.empty() && (rgb_img.cols != m_inv_mask.cols || rgb_img.rows != m_inv_mask.rows))
      {
        NODELET_ERROR_THROTTLE(0.5, "[ObjectDetector]: Received RGB image dimensions (%d x %d) are different from the loaded mask (%d x %d)! Cannot continue, skipping image.", rgb_img.rows, rgb_img.cols, m_inv_mask.rows, m_inv_mask.cols);
        return;
      }
      //}

      /* Prepare debug image if needed //{ */
      const bool publish_debug_dm = m_drmgr_ptr->config.debug_image_source == 1;
      const bool publish_debug = (!publish_debug_dm || dm_ready) && m_pub_debug.getNumSubscribers() > 0;
      cv::Mat dbg_img;
      if (publish_debug)
      {
        if (publish_debug_dm)
        {
          cv::Mat im_8UC1;
          double min = 0;
          double max = 40000;
          cv::minMaxIdx(dm_img, &min, &max);
          dm_img.convertTo(im_8UC1, CV_8UC1, 255.0 / (max-min), -min * 255.0 / (max-min));
          cv::cvtColor(im_8UC1, dbg_img, cv::COLOR_GRAY2BGR);
          /* applyColorMap(im_8UC1, dbg_img, cv::COLORMAP_JET); */
        } else
        {
          cv::cvtColor(rgb_img, dbg_img, cv::COLOR_BGR2GRAY);
          cv::cvtColor(dbg_img, dbg_img, cv::COLOR_GRAY2BGR);
        }
      }
      //}

      /* Calculate 3D positions of the detected objects //{ */
      vector<std::pair<Eigen::Vector3d, dist_qual_t>> posests;

      for (size_t it = 0; it < detections.size(); it++)
      {
        /* cout << "[" << m_node_name << "]: Processing object " << it + 1 << "/" << detections.size() << " --------------------------" << std::endl; */

        const visualanalysis_msgs::ROI2D& object = detections.at(it);
        const cv::Point center(object.x, object.y);
        const cv::Vec2d size(object.w, object.h);
        const bool physical_dimension_known = m_drmgr_ptr->config.object__physical_height > 0.0;
        const cv::Rect roi(center-size/2.0, center+size/2.0);

        /* Calculate 3D vector pointing to left and right edges of the detected object //{ */
        const Eigen::Vector3f t_vec = project(center.x - object.h/2.0*cos(M_PI_4), center.y - object.h/2.0*sin(M_PI_4), m_rgb_camera_model);
        const Eigen::Vector3f b_vec = project(center.x + object.h/2.0*cos(M_PI_4), center.y + object.h/2.0*sin(M_PI_4), m_rgb_camera_model);
        const Eigen::Vector3f c_vec = (t_vec + b_vec) / 2.0f;
        //}

        /* Estimate distance based on known size of object if applicable //{ */
        float estimated_distance = 0.0f;
        bool estimated_distance_valid = false;
        if (physical_dimension_known)
        {
          estimated_distance = estimate_distance_from_known_height(c_vec, t_vec, m_drmgr_ptr->config.object__physical_height);
          /* cout << "Estimated distance: " << estimated_distance << endl; */
          estimated_distance_valid = distance_valid(estimated_distance);
        }
        //}

        /* Get distance from depthmap if applicable //{ */
        float depthmap_distance;
        bool depthmap_distance_valid = false;
        if (dm_ready)
        {
          depthmap_distance = estimate_distance_from_depthmap(roi, m_drmgr_ptr->config.distance_min_valid_pixels_ratio, dm_img, (publish_debug && publish_debug_dm) ? dbg_img : cv::noArray());
          /* cout << "Depthmap distance: " << depthmap_distance << endl; */
          depthmap_distance_valid = distance_valid(depthmap_distance);
        }
        //}

        /* Evaluate the resulting distance and its quality //{ */
        float resulting_distance = 0.0f;
        dist_qual_t resulting_distance_quality = dist_qual_t::no_estimate;
        if (estimated_distance_valid && depthmap_distance_valid)
        {
          if (abs(depthmap_distance - estimated_distance) < m_max_dist_diff)
          {
            resulting_distance = depthmap_distance; // assuming this is a more precise distance estimate
            resulting_distance_quality = dist_qual_t::both;
          }
        } else if (depthmap_distance_valid)
        {
          resulting_distance = depthmap_distance;
          resulting_distance_quality = dist_qual_t::depthmap;
        } else if (estimated_distance_valid)
        {
          resulting_distance = estimated_distance;
          resulting_distance_quality = dist_qual_t::blob_size;
        }
        //}

        if (publish_debug)
        {
          /* cv::circle(dbg_img, center, radius, color_highlight(2*blob.color), 2); */
          cv::rectangle(dbg_img, center, cv::Vec2i(object.w, object.h), cv::Scalar(0, 0, 255), 2);
          cv::putText(dbg_img, std::to_string(resulting_distance_quality), center+cv::Point(object.w/2.0, object.h/2.0), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255));
          cv::putText(dbg_img, std::to_string(estimated_distance)+"m", center+cv::Point(object.w/2.0, 0), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255));
          cv::putText(dbg_img, std::to_string(depthmap_distance)+"m", center+cv::Point(object.w/2.0, -object.h/2.0), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255));
        }

        /* cout << "Estimated distance used: " << physical_dimension_known */
        /*      << ", valid: " << estimated_distance_valid */
        /*      << ", depthmap distance valid: " << depthmap_distance_valid */
        /*      << ", resulting quality: " << resulting_distance_quality << std::endl; */

        const Eigen::Vector3f pos_vec = resulting_distance * c_vec.normalized();

      }
      //}

      /* Publish all the calculated valid positions as pointcloud (for debugging) //{ */
      if (m_pub_pcl.getNumSubscribers() > 0)
      {
        sensor_msgs::PointCloud2 pcl_msg;
        pcl_msg.header = rgb_img_msg->header;

        sensor_msgs::PointCloud2Modifier mdf(pcl_msg);
        mdf.resize(detections.size());
        mdf.setPointCloud2Fields(5,
            "x", 1, sensor_msgs::PointField::FLOAT32,
            "y", 1, sensor_msgs::PointField::FLOAT32,
            "z", 1, sensor_msgs::PointField::FLOAT32,
            "type", 1, sensor_msgs::PointField::INT16,
            "distance_quality", 1, sensor_msgs::PointField::INT8
            );

        sensor_msgs::PointCloud2Iterator<float> iter_x(pcl_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(pcl_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(pcl_msg, "z");
        sensor_msgs::PointCloud2Iterator<int16_t> iter_type(pcl_msg, "type");
        sensor_msgs::PointCloud2Iterator<int8_t> iter_dist_qual(pcl_msg, "distance_quality");

        for (size_t it = 0; it < detections.size(); ++it, ++iter_x, ++iter_y, ++iter_z, ++iter_type, ++iter_dist_qual)
        {
          const auto& object = detections.at(it);
          *iter_x = object.position.x();
          *iter_y = object.position.y();
          *iter_z = object.position.z();
          *iter_type = color_id(object.type);
          *iter_dist_qual = object.dist_qual;
        }

        m_pub_pcl.publish(pcl_msg);
      }
      //}

      /* Publish all the calculated valid positions as pose array (for debugging) //{ */
      if (m_pub_posearr.getNumSubscribers() > 0)
      {
        mrs_msgs::PoseWithCovarianceArrayStamped msg;
        msg.header = rgb_img_msg->header;

        for (const auto& det : det_msg.detections)
        {
          mrs_msgs::PoseWithCovarianceIdentified pose;
          pose.pose = det.pose.pose;
          pose.covariance = pose.covariance;
          msg.poses.push_back(pose);
        }

        m_pub_posearr.publish(msg);
      }
      //}

      /* Publish debug image if needed //{ */
      if (publish_debug)
      {
        cv_bridge::CvImage dbg_img_ros(rgb_img_msg->header, sensor_msgs::image_encodings::BGR8, dbg_img);
        sensor_msgs::ImagePtr dbg_img_msg;
        dbg_img_msg = dbg_img_ros.toImageMsg();
        dbg_img_msg->header = rgb_img_msg->header;
        m_pub_debug.publish(dbg_img_msg);
      }
      //}

      /* Some primitive profiling info //{ */
      ros::WallDuration dur = ros::WallTime::now() - start_t;
      ros::Duration del = ros::Time::now() - rgb_img_msg->header.stamp;
      static double fps = 1.0 / dur.toSec();
      if (dur.toSec() > 0.0)
        fps = 0.1*(1.0 / dur.toSec()) + 0.9*fps;
      ROS_INFO_STREAM_THROTTLE(1.0, "Processing FPS: " << fps << "Hz");
      ROS_INFO_STREAM_THROTTLE(1.0, "delay: " << del.toSec() * 1000.0 << "ms (processing: " << dur.toSec() * 1000.0 << "ms)");
      //}

      ROS_INFO_THROTTLE(0.5, "[%s]: Image processed", m_node_name.c_str());
    }

  }
  //}

  /* generate_covariance() method //{ */
  ObjectDetector::ros_cov_t ObjectDetector::generate_covariance(const Eigen::Vector3f& pos, const double xy_covariance_coeff, const double z_covariance_coeff)
  {
    Eigen::Matrix3d e_cov = calc_position_covariance(pos.cast<double>(), xy_covariance_coeff, z_covariance_coeff);
    ros_cov_t cov;
    for (int r = 0; r < 6; r++)
    {
      for (int c = 0; c < 6; c++)
      {
        if (r < 3 && c < 3)
          cov[r * 6 + c] = e_cov(r, c);
        else if (r == c)
          cov[r * 6 + c] = 666;
        else
          cov[r * 6 + c] = 0.0;
      }
    }
    return cov;
  }
  //}

  /* calc_position_covariance() method //{ */
  /* position_sf is position of the detection in 3D in the frame of the sensor (camera) */
  Eigen::Matrix3d ObjectDetector::calc_position_covariance(const Eigen::Vector3d& position_sf, const double xy_covariance_coeff, const double z_covariance_coeff)
  {
    /* Calculates the corresponding covariance matrix of the estimated 3D position */
    Eigen::Matrix3d pos_cov = Eigen::Matrix3d::Identity();  // prepare the covariance matrix
    pos_cov(0, 0) = pos_cov(1, 1) = xy_covariance_coeff;

    pos_cov(2, 2) = position_sf(2) * sqrt(position_sf(2)) * z_covariance_coeff;
    if (pos_cov(2, 2) < 0.33 * z_covariance_coeff)
      pos_cov(2, 2) = 0.33 * z_covariance_coeff;

    /* // Find the rotation matrix to rotate the covariance to point in the direction of the estimated position */
    const Eigen::Vector3d a(0.0, 0.0, 1.0);
    const Eigen::Vector3d b = position_sf.normalized();
    /* const auto vec_rot = mrs_lib::rotation_between(a, b); */
    const auto vec_rot = mrs_lib::geometry::rotationBetween(a, b);
    pos_cov = rotate_covariance(pos_cov, vec_rot);  // rotate the covariance to point in direction of est. position
    return pos_cov;
  }
  //}

  /* rotate_covariance() method //{ */
  Eigen::Matrix3d ObjectDetector::rotate_covariance(const Eigen::Matrix3d& covariance, const Eigen::Matrix3d& rotation)
  {
    return rotation * covariance * rotation.transpose();  // rotate the covariance to point in direction of est. position
  }
  //}

  /* distance_valid() method //{ */
  bool ObjectDetector::distance_valid(float distance)
  {
    return !(isnan(distance) || distance < 0.0 || distance > m_max_dist);
  }
  //}

  /* estimate_distance_from_known_height() method //{ */
  float ObjectDetector::estimate_distance_from_known_height(const Eigen::Vector3f& c_vec, const Eigen::Vector3f& t_vec, float physical_height)
  {
    const float dot = c_vec.dot(t_vec)/(c_vec.norm() * t_vec.norm());
    const float dist = physical_height/std::sqrt(1.0f - dot*dot)/2.0f;
    return dist;
  }
  //}

  /* estimate_distance_from_depthmap() method //{ */
  float ObjectDetector::estimate_distance_from_depthmap(const cv::Rect& roi, const double min_valid_ratio, const cv::Mat& dm_img, cv::InputOutputArray dbg_img)
  {
    bool publish_debug = dbg_img.needed();
    cv::Mat dbg_mat;
    if (publish_debug)
      dbg_mat = dbg_img.getMat();

    const cv::Mat tmp_dm_img = dm_img(roi);
    cv::Mat tmp_dbg_img = publish_debug ? dbg_mat(roi) : cv::Mat();
    const Size size = tmp_dm_img.size();
    size_t n_candidates = cv::sum(tmp_mask)[0]/255.0;

    size_t n_dm_samples = 0;
    std::vector<float> dists;
    dists.reserve(n_candidates);
    // find median of all pixels in the area of the detected blob
    // go through all pixels in a square of size 2*radius
    for (int x = 0; x < size.width; x++)
    {
      for (int y = 0; y < size.height; y++)
      {
        const uint16_t depth = tmp_dm_img.at<uint16_t>(y, x);
        // skip invalid measurements
        if (depth <= m_min_depth
         || depth >= m_max_depth)
          continue;
        const float u = roi.tl().x + x;
        const float v = roi.tl().y + y;
        const float dist = depth2range(depth, u, v, m_dm_camera_model.fx(), m_dm_camera_model.fy(), m_dm_camera_model.cx(), m_dm_camera_model.cy());
        if (std::isnan(dist))
          ROS_WARN("[ObjectDetector]: Distance is nan! Skipping.");
        else
        {
          n_dm_samples++;
          dists.push_back(dist);
          if (publish_debug)
            mark_pixel(tmp_dbg_img, x, y, 2);
        }
      }
    }
    const double valid_ratio = double(n_dm_samples)/double(n_candidates);
    if (valid_ratio > min_valid_ratio)
    {
      // order the first n elements (up to the median)
      std::nth_element(std::begin(dists), std::begin(dists) + dists.size()/2, std::end(dists));
      const float median = dists[dists.size()/2]/1000.0f;
      return median;
    } else
    {
      return std::numeric_limits<float>::quiet_NaN();
    }
  }
  //}

  /* onInit() method //{ */

  void ObjectDetector::onInit()
  {

    ROS_INFO("[%s]: Initializing ------------------------------", m_node_name.c_str());
    m_nh = nodelet::Nodelet::getMTPrivateNodeHandle();
    ros::Time::waitForValid();

    /** Load parameters from ROS * //{*/
    string node_name = ros::this_node::getName().c_str();
    ParamLoader pl(m_nh, m_node_name);

    // LOAD DYNAMIC PARAMETERS
    m_drmgr_ptr = std::make_unique<drmgr_t>(m_nh, m_node_name);

    // LOAD STATIC PARAMETERS
    ROS_INFO("[%s]: Loading static parameters:", m_node_name.c_str());
    // Load the detection parameters
    pl.loadParam("max_dist", m_max_dist);
    pl.loadParam("max_dist_diff", m_max_dist_diff);
    pl.loadParam("min_depth", m_min_depth);
    pl.loadParam("max_depth", m_max_depth);
    pl.loadParam("mask_filename", m_mask_filename, ""s);

    double loop_rate = pl.loadParam2<double>("loop_rate", 100);

    if (!m_drmgr_ptr->loaded_successfully())
    {
      ROS_ERROR("[%s]: Some default values of dynamically reconfigurable parameters were not loaded successfully, ending the node", m_node_name.c_str());
      ros::shutdown();
    }

    if (!pl.loadedSuccessfully())
    {
      ROS_ERROR("[%s]: Some compulsory parameters were not loaded successfully, ending the node", m_node_name.c_str());
      ros::shutdown();
    }
    //}

    /** Create publishers and subscribers //{**/
    // Initialize other subs and pubs
    mrs_lib::SubscribeHandlerOptions shopts;
    shopts.nh = m_nh;
    shopts.node_name = m_node_name;
    shopts.no_message_timeout = ros::Duration(5.0);
    mrs_lib::construct_object(m_sh_dets, shopts, "detections");
    mrs_lib::construct_object(m_sh_dm, shopts, "dm_image");
    mrs_lib::construct_object(m_sh_dm_cinfo, shopts, "dm_camera_info");
    mrs_lib::construct_object(m_sh_rgb, shopts, "rgb_image");
    mrs_lib::construct_object(m_sh_rgb_cinfo, shopts, "rgb_camera_info");

    image_transport::ImageTransport it(m_nh);
    m_pub_debug = it.advertise("debug_image", 1);
    m_pub_pcl = m_nh.advertise<sensor_msgs::PointCloud2>("detected_objects_pcl", 10);
    m_pub_posearr = m_nh.advertise<mrs_msgs::PoseWithCovarianceArrayStamped>("detected_objects_posearr", 10);
    //}

    m_is_initialized = true;

    /* timers  //{ */

    m_main_loop_timer = m_nh.createTimer(ros::Rate(loop_rate), &ObjectDetector::main_loop, this);

    //}

    ROS_INFO("[%s]: Initialized ------------------------------", m_node_name.c_str());
  }

  //}

}

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(object_detect::ObjectDetector, nodelet::Nodelet)
