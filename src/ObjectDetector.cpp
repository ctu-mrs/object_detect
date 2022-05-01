#include "object_detect/ObjectDetector.h"

using namespace cv;
using namespace std;
using namespace object_detect;
using namespace Eigen;
using namespace mrs_lib;

namespace object_detect
{

  /* find_msg() method //{ */
  template <typename MsgT>
  MsgT find_msg(const boost::circular_buffer<MsgT>& bfr, const ros::Time& stamp)
  {
    if (bfr.empty())
      return nullptr;
  
    MsgT last_msg = bfr.front();
    for (const auto& msg : bfr)
    {
      if (msg->header.stamp > stamp)
      {
        const double diff_last = std::abs((last_msg->header.stamp - stamp).toSec());
        const double diff_cur = std::abs((msg->header.stamp - stamp).toSec());
        if (diff_last > diff_cur)
          last_msg = msg;
        break;
      }
      last_msg = msg;
    }
    return last_msg;
  }
  //}

  /* main_loop() method //{ */
  void ObjectDetector::main_loop(const sensor_msgs::Image::ConstPtr& rgb_img_msg, const visualanalysis_msgs::TargetLocations2D::ConstPtr& det_msg)
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
      const std::pair<double&, double&> cov_coeffs_terabee = {m_drmgr_ptr->config.cov_coeffs__xy__terabee, m_drmgr_ptr->config.cov_coeffs__z__terabee};
      const std::pair<double&, double&> cov_coeffs_two = {m_drmgr_ptr->config.cov_coeffs__xy__two, m_drmgr_ptr->config.cov_coeffs__z__two};
      const std::pair<double&, double&> cov_coeffs_all = {m_drmgr_ptr->config.cov_coeffs__xy__all, m_drmgr_ptr->config.cov_coeffs__z__all};

      m_cov_coeffs.insert_or_assign(dist_qual_t::no_estimate, cov_coeffs_no_estimate);
      m_cov_coeffs.insert_or_assign(dist_qual_t::blob_size, cov_coeffs_blob_size);
      m_cov_coeffs.insert_or_assign(dist_qual_t::depthmap, cov_coeffs_depthmap);
      m_cov_coeffs.insert_or_assign(dist_qual_t::terabee, cov_coeffs_terabee);

      m_cov_coeffs.insert_or_assign(dist_qual_t::blob_depth, cov_coeffs_two);
      m_cov_coeffs.insert_or_assign(dist_qual_t::blob_tb, cov_coeffs_two);
      m_cov_coeffs.insert_or_assign(dist_qual_t::depth_tb, cov_coeffs_two);

      m_cov_coeffs.insert_or_assign(dist_qual_t::all, cov_coeffs_all);
    }

    //}

    const bool rgb_ready = m_rgb_camera_model_valid;

    // Check if we got all required messages
    if (rgb_ready)
    {
      /* Copy values from subscribers to local variables //{ */
      const ros::WallTime start_t = ros::WallTime::now();

      const auto& detections = det_msg->detections;

      bool dm_ready = m_sh_dm.hasMsg() && m_dm_camera_model_valid;
      cv::Mat dm_img;
      if (dm_ready)
      {
        const sensor_msgs::ImageConstPtr dm_img_msg = find_msg(m_dm_bfr, det_msg->header.stamp);
        if (!dm_img_msg)
        {
          NODELET_WARN_THROTTLE(0.5, "[ObjectDetector]: No depthmap in the buffer! Cannot continue, not using depthmap.");
          dm_ready = false;
        }
        else
        {
          const cv_bridge::CvImageConstPtr dm_img_ros = cv_bridge::toCvShare(dm_img_msg, sensor_msgs::image_encodings::TYPE_16UC1);
          dm_img = dm_img_ros->image;
          if (!m_inv_mask.empty() && (dm_img.cols != m_inv_mask.cols || dm_img.rows != m_inv_mask.rows))
          {
            NODELET_WARN_THROTTLE(0.5, "[ObjectDetector]: Received depthmap dimensions (%d x %d) are different from the loaded mask (%d x %d)! Cannot continue, not using depthmap.", dm_img.rows, dm_img.cols, m_inv_mask.rows, m_inv_mask.cols);
            dm_ready = false;
          }
        }
      }

      const cv_bridge::CvImageConstPtr rgb_img_ros = cv_bridge::toCvShare(rgb_img_msg, sensor_msgs::image_encodings::BGR8);
      const cv::Mat rgb_img = rgb_img_ros->image;
      if (!m_inv_mask.empty() && (rgb_img.cols != m_inv_mask.cols || rgb_img.rows != m_inv_mask.rows))
      {
        NODELET_ERROR_THROTTLE(0.5, "[ObjectDetector]: Received RGB image dimensions (%d x %d) are different from the loaded mask (%d x %d)! Cannot continue, skipping image.", rgb_img.rows, rgb_img.cols, m_inv_mask.rows, m_inv_mask.cols);
        return;
      }

      bool tb_ready = m_sh_tb.hasMsg();
      positioning_systems_ros::RtlsTrackerFrame::ConstPtr tb_msg = nullptr;
      if (tb_ready)
      {
        tb_msg = find_msg(m_tb_bfr, det_msg->header.stamp);
        if (tb_msg == nullptr || tb_msg->anchors.empty() || ros::Time::now() - tb_msg->header.stamp > m_tb_timeout)
          tb_ready = false;
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
      vector<std::pair<Eigen::Vector3f, dist_qual_t>> posests;

      for (size_t it = 0; it < detections.size(); it++)
      {
        /* cout << "[" << m_node_name << "]: Processing object " << it + 1 << "/" << detections.size() << " --------------------------" << std::endl; */

        const visualanalysis_msgs::ROI2D& object = detections.at(it);
        const cv::Point2f topleft(object.x, object.y);
        const cv::Point2f center(object.x + object.w/2.0f, object.y + object.h/2.0f);
        const cv::Size size(object.w, object.h);
        const bool physical_dimension_known = m_drmgr_ptr->config.object__physical_height > 0.0;
        const cv::Rect roi(topleft, size);

        /* Calculate 3D vector pointing to left and right edges of the detected object //{ */
        const Eigen::Vector3f t_vec = project(center.x, center.y - object.h/2.0f, m_rgb_camera_model);
        const Eigen::Vector3f b_vec = project(center.x, center.y + object.h/2.0f, m_rgb_camera_model);
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
        float depthmap_distance = 0.0f;
        bool depthmap_distance_valid = false;
        if (dm_ready)
        {
          depthmap_distance = estimate_distance_from_depthmap(roi, m_drmgr_ptr->config.distance_min_valid_pixels_ratio, dm_img, (publish_debug && publish_debug_dm) ? dbg_img : cv::noArray());
          /* cout << "Depthmap distance: " << depthmap_distance << endl; */
          depthmap_distance_valid = distance_valid(depthmap_distance);
        }
        //}

        /* Get distance from terabee if applicable //{ */
        float terabee_distance = 0.0f;
        bool terabee_distance_valid = false;
        if (tb_ready)
        {
          terabee_distance = tb_msg->anchors.at(0).distance;
          /* cout << "Terabee distance: " << terabee_distance << endl; */
          terabee_distance_valid = distance_valid(terabee_distance);
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
            resulting_distance_quality = dist_qual_t::blob_depth;
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

        if (terabee_distance_valid)
        {
          // | --------------- no other estimate available -------------- |
          if (resulting_distance_quality == dist_qual_t::no_estimate)
          {
            resulting_distance = terabee_distance; // assuming this is a more precise distance estimate
            resulting_distance_quality = dist_qual_t::terabee;
          }
          // | ------------- both other estimates available ------------- |
          else if (resulting_distance_quality == dist_qual_t::blob_depth)
          {
            if (abs(terabee_distance - estimated_distance) < m_max_dist_diff)
            {
              resulting_distance = (resulting_distance + terabee_distance)/2.0; // assuming this is a more precise distance estimate
              resulting_distance_quality = dist_qual_t::all;
            }
            else
            {
              resulting_distance_quality = dist_qual_t::no_estimate;
              ROS_INFO_THROTTLE(0.5, "[%s]: Terabee distance is different from vision! Terabee: %.2fm, vision: %.2fm", m_node_name.c_str(), terabee_distance, resulting_distance);
              resulting_distance = 0.0f;
            }
          }
          // | ---------------- depth estimate available ---------------- |
          else if (resulting_distance_quality == dist_qual_t::depthmap)
          {
            if (abs(terabee_distance - estimated_distance) < m_max_dist_diff)
            {
              resulting_distance = (resulting_distance + terabee_distance)/2.0; // assuming this is a more precise distance estimate
              resulting_distance_quality = dist_qual_t::depth_tb;
            }
            else
            {
              resulting_distance_quality = dist_qual_t::no_estimate;
              ROS_INFO_THROTTLE(0.5, "[%s]: Terabee distance is different from depthmap! Terabee: %.2fm, depthmap: %.2fm", m_node_name.c_str(), terabee_distance, resulting_distance);
              resulting_distance = 0.0f;
            }
          }
          // | ----------------- blob estimate available ---------------- |
          else if (resulting_distance_quality == dist_qual_t::blob_size)
          {
            if (abs(terabee_distance - estimated_distance) < m_max_dist_diff)
            {
              resulting_distance = terabee_distance; // assuming this is a more precise distance estimate
              resulting_distance_quality = dist_qual_t::blob_tb;
            }
            else
            {
              resulting_distance_quality = dist_qual_t::no_estimate;
              ROS_INFO_THROTTLE(0.5, "[%s]: Terabee distance is different from blob size! Terabee: %.2fm, blob: %.2fm", m_node_name.c_str(), terabee_distance, resulting_distance);
              resulting_distance = 0.0f;
            }
          }
        }

        switch (resulting_distance_quality)
        {
          case dist_qual_t::unknown_qual: ROS_ERROR_THROTTLE(0.5, "[%s]: Error in distance estimation selection!", m_node_name.c_str()); break;
          case dist_qual_t::no_estimate: ROS_WARN_THROTTLE(0.5, "[%s]: No consistent distance estimate", m_node_name.c_str()); break;
          case dist_qual_t::blob_size: ROS_INFO_THROTTLE(0.5, "[%s]: Using size-based distance estimate: %.2fm", m_node_name.c_str(), resulting_distance); break;
          case dist_qual_t::depthmap: ROS_INFO_THROTTLE(0.5, "[%s]: Using depthmap distance estimate: %.2fm", m_node_name.c_str(), resulting_distance); break;
          case dist_qual_t::terabee: ROS_INFO_THROTTLE(0.5, "[%s]: Using terabee distance estimate: %.2fm", m_node_name.c_str(), resulting_distance); break;
          case dist_qual_t::blob_depth: ROS_INFO_THROTTLE(0.5, "[%s]: Using size-based and depthmap distance estimate: %.2fm", m_node_name.c_str(), resulting_distance); break;
          case dist_qual_t::blob_tb: ROS_INFO_THROTTLE(0.5, "[%s]: Using size-based and terabee distance estimate: %.2fm", m_node_name.c_str(), resulting_distance); break;
          case dist_qual_t::depth_tb: ROS_INFO_THROTTLE(0.5, "[%s]: Using depthmap and terabee distance estimate: %.2fm", m_node_name.c_str(), resulting_distance); break;
          case dist_qual_t::all: ROS_INFO_THROTTLE(0.5, "[%s]: Using all distance estimates: %.2fm", m_node_name.c_str(), resulting_distance); break;
        }
        //}

        if (resulting_distance_quality == dist_qual_t::no_estimate)
        {
          continue;
          ROS_INFO_THROTTLE(0.5, "[%s]: No distance estimate available, skipping detection.", m_node_name.c_str());
        }

        if (publish_debug)
        {
          cv::rectangle(dbg_img, roi, cv::Scalar(0, 0, 255), 2);
          cv::putText(dbg_img, std::to_string(resulting_distance_quality), topleft, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255));
          cv::putText(dbg_img, std::to_string(estimated_distance)+"m", cv::Point2d(topleft.x+size.width, topleft.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255));
          cv::putText(dbg_img, std::to_string(depthmap_distance)+"m", cv::Point2d(topleft.x+size.width, topleft.y+size.height), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255));
        }

        /* cout << "Estimated distance used: " << physical_dimension_known */
        /*      << ", valid: " << estimated_distance_valid */
        /*      << ", depthmap distance valid: " << depthmap_distance_valid */
        /*      << ", resulting quality: " << resulting_distance_quality << std::endl; */

        const Eigen::Vector3f pos_vec = resulting_distance * c_vec.normalized();
        posests.emplace_back(pos_vec, resulting_distance_quality);
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

        for (size_t it = 0; it < posests.size(); ++it, ++iter_x, ++iter_y, ++iter_z, ++iter_type, ++iter_dist_qual)
        {
          const auto& object = posests.at(it);
          *iter_x = object.first.x();
          *iter_y = object.first.y();
          *iter_z = object.first.z();
          *iter_dist_qual = object.second;
        }

        m_pub_pcl.publish(pcl_msg);
      }
      //}

      /* Publish all the calculated valid positions as pose array (for debugging) //{ */
      if (m_pub_posearr.getNumSubscribers() > 0)
      {
        mrs_msgs::PoseWithCovarianceArrayStamped msg;
        msg.header = rgb_img_msg->header;

        for (const auto& det : posests)
          msg.poses.push_back(to_message(det.first, det.second));

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
    return !(isnan(distance) || distance <= 0.0 || distance > m_max_dist);
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

    const int tl_x = std::clamp(roi.tl().x, 0, dm_img.cols-1);
    const int tl_y = std::clamp(roi.tl().y, 0, dm_img.rows-1);
    const int br_x = std::clamp(roi.br().x, 0, dm_img.cols-1);
    const int br_y = std::clamp(roi.br().y, 0, dm_img.rows-1);
    const cv::Rect roi_clamped(cv::Point(tl_x, tl_y), cv::Point(br_x, br_y));
    if (tl_x >= br_x || tl_y >= br_y)
      return 0.0f;
    const cv::Mat tmp_dm_img = dm_img(roi_clamped);
    cv::Mat tmp_dbg_img = publish_debug ? dbg_mat(roi_clamped) : cv::Mat();
    const Size size = tmp_dm_img.size();
    size_t n_candidates = tmp_dm_img.rows*tmp_dm_img.cols;
    if (n_candidates == 0)
      return 0.0f;

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
        const float u = roi_clamped.tl().x + x;
        const float v = roi_clamped.tl().y + y;
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

  /* to_output_message() method //{ */
  mrs_msgs::PoseWithCovarianceIdentified ObjectDetector::to_message(const Eigen::Vector3f& pos, const dist_qual_t dist_qual) const
  {
    mrs_msgs::PoseWithCovarianceIdentified ret;
    mrs_msgs::PoseWithCovarianceIdentified::_pose_type& pose = ret.pose;

    pose.position.x = pos.x();
    pose.position.y = pos.y();
    pose.position.z = pos.z();
    pose.orientation.x = 0;
    pose.orientation.y = 0;
    pose.orientation.z = 0;
    pose.orientation.w = 1;

    if (m_cov_coeffs.find(dist_qual) == std::end(m_cov_coeffs))
    {
      ROS_ERROR("[ObjectDetector]: Invalid distance estimate quality %d! Skipping detection.", dist_qual);
      return ret;
    }
    auto [xy_covariance_coeff, z_covariance_coeff] = m_cov_coeffs.at(dist_qual);
    ret.covariance = generate_covariance(pos, xy_covariance_coeff, z_covariance_coeff);
    ret.id = dist_qual;

    return ret;
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
    pl.loadParam("terabee/timeout", m_tb_timeout);

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

    m_sub_rgb = std::make_unique<message_filters::Subscriber<sensor_msgs::Image>>(m_nh, "rgb_image", 30);
    m_sub_det = std::make_unique<message_filters::Subscriber<visualanalysis_msgs::TargetLocations2D>>(m_nh, "detections", 5);
    m_sync = std::make_unique<message_filters::Synchronizer<sync_policy>>(sync_policy(30), *m_sub_rgb, *m_sub_det);
    m_sync->registerCallback(boost::bind(&ObjectDetector::main_loop, this, boost::placeholders::_1, boost::placeholders::_2));

    mrs_lib::SubscribeHandlerOptions shopts;
    shopts.nh = m_nh;
    shopts.node_name = m_node_name;
    shopts.no_message_timeout = ros::Duration(5.0);
    mrs_lib::construct_object(m_sh_dets, shopts, "detections");
    mrs_lib::construct_object(m_sh_rgb, shopts, "rgb_image");
    mrs_lib::construct_object(m_sh_dm, shopts, "dm_image", [this](mrs_lib::SubscribeHandler<sensor_msgs::Image>& sh)
        {
          m_dm_bfr.push_back(sh.getMsg());
        });
    mrs_lib::construct_object(m_sh_tb, shopts, "terabee", [this](mrs_lib::SubscribeHandler<positioning_systems_ros::RtlsTrackerFrame>& sh)
        {
          m_tb_bfr.push_back(sh.getMsg());
        });
    mrs_lib::construct_object(m_sh_dm_cinfo, shopts, "dm_camera_info");
    mrs_lib::construct_object(m_sh_rgb_cinfo, shopts, "rgb_camera_info");

    image_transport::ImageTransport it(m_nh);
    m_pub_debug = it.advertise("debug_image", 1);
    m_pub_pcl = m_nh.advertise<sensor_msgs::PointCloud2>("detected_objects_pcl", 10);
    m_pub_posearr = m_nh.advertise<mrs_msgs::PoseWithCovarianceArrayStamped>("detected_objects_posearr", 10);
    //}

    m_dm_bfr.set_capacity(30);
    m_tb_bfr.set_capacity(30);
    m_is_initialized = true;

    ROS_INFO("[%s]: Initialized ------------------------------", m_node_name.c_str());
  }

  //}

}

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(object_detect::ObjectDetector, nodelet::Nodelet)
