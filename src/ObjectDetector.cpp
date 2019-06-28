#include "ObjectDetector.h"

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
    if (m_sh_dm_cinfo->has_data() && !m_sh_dm_cinfo->used_data())
      m_dm_camera_model.fromCameraInfo(m_sh_dm_cinfo->get_data());
    if (m_sh_rgb_cinfo->has_data() && !m_sh_rgb_cinfo->used_data())
      m_rgb_camera_model.fromCameraInfo(m_sh_rgb_cinfo->get_data());
    //}

    const bool rgb_ready = m_sh_rgb->new_data() && m_sh_rgb_cinfo->used_data();
    const bool dm_ready = m_sh_dm->new_data() && m_sh_dm_cinfo->used_data();

    // Check if we got all required messages
    if (rgb_ready)
    {
      /* Copy values from subscribers to local variables //{ */
      const ros::Time start_t = ros::Time::now();
      cv::Mat dm_img;
      if (dm_ready)
      {
        const sensor_msgs::ImageConstPtr dm_img_msg = m_sh_dm->get_data();
        const cv_bridge::CvImageConstPtr dm_img_ros = cv_bridge::toCvShare(dm_img_msg, sensor_msgs::image_encodings::TYPE_16UC1);
        dm_img = dm_img_ros->image;
      }
      const sensor_msgs::ImageConstPtr rgb_img_msg = m_sh_rgb->get_data();
      const cv_bridge::CvImageConstPtr rgb_img_ros = cv_bridge::toCvShare(rgb_img_msg, sensor_msgs::image_encodings::BGR8);
      const cv::Mat rgb_img = rgb_img_ros->image;
      cv::Mat label_img;
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
          const double min = 0;
          const double max = 40000;
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

      std::vector<Blob> blobs;
      {
        std::scoped_lock<std::mutex> lck(m_active_seg_confs_mtx);
        /* Detect blobs of required color in the RGB image //{ */
        if (m_prev_color_id != m_drmgr_ptr->config.segment_color)
        {
          m_active_seg_confs = get_segmentation_configs(m_seg_confs, {m_drmgr_ptr->config.segment_color});
          m_prev_color_id = m_drmgr_ptr->config.segment_color;
        }
        for (const auto& seg_conf : m_active_seg_confs)
          NODELET_INFO("[ObjectDetector]: Segmenting %s color", color_name(seg_conf.color).c_str());
        BlobDetector blob_det(m_drmgr_ptr->config);
        if (m_drmgr_ptr->config.override_settings)
        {
          if (!m_active_seg_confs.empty())
            m_active_seg_confs.at(0) = load_segmentation_config(m_drmgr_ptr->config);
          blobs = blob_det.detect(rgb_img, m_active_seg_confs, label_img);
        } else
        {
          blobs = blob_det.detect(rgb_img, m_cur_lut, m_active_seg_confs, label_img);
        }
      }
      if (publish_debug)
        highlight_mask(dbg_img, label_img);
      //}

      /* Calculate 3D positions of the detected blobs //{ */
      const size_t n_dets = blobs.size();
      vector<geometry_msgs::Point32> positions;
      vector<dist_qual_t> distance_qualities;
      positions.reserve(n_dets);

      for (size_t it = 0; it < n_dets; it++)
      {
        cout << "[" << m_node_name << "]: Processing object " << it + 1 << "/" << n_dets << " --------------------------" << std::endl;

        const Blob& blob = blobs.at(it);
        const cv::Point& center = blob.location;
        const float radius = blob.radius;

        /* Calculate 3D vector pointing to left and right edges of the detected object //{ */
        const Eigen::Vector3f l_vec = project(center.x - radius*cos(M_PI_4), center.y - radius*sin(M_PI_4), m_rgb_camera_model);
        const Eigen::Vector3f r_vec = project(center.x + radius*cos(M_PI_4), center.y + radius*sin(M_PI_4), m_rgb_camera_model);
        //}
        
        /* Estimate distance based on known size of object if applicable //{ */
        float estimated_distance = 0.0f;
        bool estimated_distance_valid = false;
        if (m_object_radius_known)
        {
          estimated_distance = estimate_distance_from_known_size(l_vec, r_vec, m_object_radius);
          cout << "Estimated distance: " << estimated_distance << endl;
          estimated_distance_valid = distance_valid(estimated_distance);
        }
        //}

        /* Get distance from depthmap if applicable //{ */
        float depthmap_distance;
        bool depthmap_distance_valid = false;
        if (dm_ready)
        {
          depthmap_distance = estimate_distance_from_depthmap(center, radius, dm_img, label_img, blob.color, (publish_debug && publish_debug_dm) ? dbg_img : cv::noArray());
          cout << "Depthmap distance: " << depthmap_distance << endl;
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
          cv::circle(dbg_img, center, radius, cv::Scalar(0, 0, 255), 2);
          cv::putText(dbg_img, std::to_string(resulting_distance_quality), center+cv::Point(radius, radius), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255));
          cv::putText(dbg_img, std::to_string(estimated_distance)+"m", center+cv::Point(radius, 0), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255));
          cv::putText(dbg_img, std::to_string(depthmap_distance)+"m", center+cv::Point(radius, -radius), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255));
        }

        cout << "Estimated distance used: " << m_object_radius_known
             << ", valid: " << estimated_distance_valid
             << ", depthmap distance valid: " << depthmap_distance_valid
             << ", resulting quality: " << resulting_distance_quality << std::endl;

        if (resulting_distance_quality > dist_qual_t::no_estimate)
        {
          /* Calculate the estimated position of the object //{ */
          const Eigen::Vector3f pos_vec = resulting_distance * (l_vec + r_vec) / 2.0;
          cout << "Estimated location (camera CS): [" << pos_vec(0) << ", " << pos_vec(1) << ", " << pos_vec(2) << "]" << std::endl;
          //}

          /* If all is OK, add the position to the vector //{ */
          geometry_msgs::Point32 cur_pos;
          cur_pos.x = pos_vec(0);
          cur_pos.y = pos_vec(1);
          cur_pos.z = pos_vec(2);
          positions.push_back(cur_pos);
          distance_qualities.push_back(resulting_distance_quality);
          //}
        }

        cout << "[" << m_node_name << "]: Done with object " << it + 1 << "/" << n_dets << " --------------------------" << std::endl;
      }
      //}

      /* Publish all the calculated valid positions //{ */
      if (m_pub_det.getNumSubscribers() > 0)
      {
        object_detect::PoseWithCovarianceArrayStamped det_msg;
  
        det_msg.header = rgb_img_msg->header;
        det_msg.poses = generate_poses(positions, distance_qualities);

        m_pub_det.publish(det_msg);
      }
      //}

      /* Publish all the calculated valid positions //{ */
      if (m_pub_pcl.getNumSubscribers() > 0)
      {
        sensor_msgs::PointCloud pcl_msg;
        pcl_msg.header = rgb_img_msg->header;
        pcl_msg.points = positions;

        sensor_msgs::ChannelFloat32 chl_msg;
        chl_msg.name = "distance_quality"s;
        chl_msg.values = std::vector<float>(std::begin(distance_qualities), std::end(distance_qualities));
        pcl_msg.channels.push_back(chl_msg);

        m_pub_pcl.publish(pcl_msg);
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
      ros::Duration dur = ros::Time::now() - start_t;
      ros::Duration del = ros::Time::now() - rgb_img_msg->header.stamp;
      static double fps = 1.0 / dur.toSec();
      fps = 0.1*(1.0 / dur.toSec()) + 0.9*fps;
      cout << "processing FPS: " << fps << "Hz" << std::endl;
      cout << "delay: " << del.toSec() * 1000.0 << "ms" << std::endl;
      //}

      ROS_INFO("[%s]: Image processed", m_node_name.c_str());
    }
    
  }
  //}

  /* drmgr_update_loop() method //{ */
  void ObjectDetector::drcfg_update_loop([[maybe_unused]] const ros::TimerEvent& evt)
  {
    if (!m_drmgr_ptr->config.override_settings)
    {
      std::vector<SegConf> active_seg_confs;
      {
        std::scoped_lock<std::mutex> lck(m_active_seg_confs_mtx);
        active_seg_confs = m_active_seg_confs;
      }
      if (!active_seg_confs.empty())
        update_drcfg(active_seg_confs.at(0));
    }
  }
  //}

  /* generate_poses() method //{ */
  ObjectDetector::ros_poses_t ObjectDetector::generate_poses(const std::vector<geometry_msgs::Point32>& positions, const std::vector<dist_qual_t>& distance_qualities)
  {
    assert(positions.size() == distance_qualities.size());
    ros_poses_t ret;
    ret.reserve(positions.size());
  
    for (unsigned it = 0; it < positions.size(); it++)
    {
      const auto qual = distance_qualities.at(it);
      const auto pos = positions.at(it);
  
      ros_pose_t pose;
      pose.position.x = pos.x;
      pose.position.y = pos.y;
      pose.position.z = pos.z;
      pose.orientation.x = 0;
      pose.orientation.y = 0;
      pose.orientation.z = 0;
      pose.orientation.w = 1;
  
      if (m_cov_coeffs.find(qual) == std::end(m_cov_coeffs))
      {
        ROS_ERROR_THROTTLE(1.0, "[ObjectDetector]: Invalid distance estimate quality %d! Skipping detection.", qual);
        continue;
      }
      auto [xy_covariance_coeff, z_covariance_coeff] = m_cov_coeffs.at(qual);
      const ros_cov_t cov = generate_covariance(pos, xy_covariance_coeff, z_covariance_coeff);
  
      geometry_msgs::PoseWithCovariance pose_with_cov;
      pose_with_cov.pose = pose;
      pose_with_cov.covariance = cov;
      ret.push_back(pose_with_cov);
    }
    return ret;
  }
  //}

  /* generate_covariance() method //{ */
  ObjectDetector::ros_cov_t ObjectDetector::generate_covariance(const geometry_msgs::Point32& pos, const double xy_covariance_coeff, const double z_covariance_coeff)
  {
    Eigen::Vector3d e_pos(pos.x, pos.y, pos.z);
    Eigen::Matrix3d e_cov = calc_position_covariance(e_pos, xy_covariance_coeff, z_covariance_coeff);
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
    const double tol = 1e-9;
    pos_cov(0, 0) = pos_cov(1, 1) = xy_covariance_coeff;

    pos_cov(2, 2) = position_sf(2) * sqrt(position_sf(2)) * z_covariance_coeff;
    if (pos_cov(2, 2) < 0.33 * z_covariance_coeff)
      pos_cov(2, 2) = 0.33 * z_covariance_coeff;

    // Find the rotation matrix to rotate the covariance to point in the direction of the estimated position
    const Eigen::Vector3d a(0.0, 0.0, 1.0);
    const Eigen::Vector3d b = position_sf.normalized();
    const Eigen::Vector3d v = a.cross(b);
    const double sin_ab = v.norm();
    const double cos_ab = a.dot(b);
    Eigen::Matrix3d vec_rot = Eigen::Matrix3d::Identity();
    if (sin_ab < tol)  // unprobable, but possible - then it is identity or 180deg
    {
      if (cos_ab + 1.0 < tol)  // that would be 180deg
      {
        vec_rot << -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0;
      }     // otherwise its identity
    } else  // otherwise just construct the matrix
    {
      Eigen::Matrix3d v_x;
      v_x << 0.0, -v(2), v(1), v(2), 0.0, -v(0), -v(1), v(0), 0.0;
      vec_rot = Eigen::Matrix3d::Identity() + v_x + (1 - cos_ab) / (sin_ab * sin_ab) * (v_x * v_x);
    }
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

  /* get_active_segmentation_configs() method //{ */
  std::vector<SegConf> ObjectDetector::get_segmentation_configs(const std::vector<SegConf>& all_seg_confs, std::vector<int> color_ids)
  {
    std::vector<SegConf> ret;
    for (const auto& color_id : color_ids)
    {
      for (const auto& seg_conf : all_seg_confs)
      {
        if (seg_conf.color == color_id)
          ret.push_back(seg_conf);
      }
    }
    return ret;
  }
  //}

  /* distance_valid() method //{ */
  bool ObjectDetector::distance_valid(float distance)
  {
    return !(isnan(distance) || distance < 0.0 || distance > m_max_dist);
  }
  //}

  /* estimate_distance_from_known_size() method //{ */
  float ObjectDetector::estimate_distance_from_known_size(const Eigen::Vector3f& l_vec, const Eigen::Vector3f& r_vec, float known_size)
  {
    const float alpha = acos(l_vec.dot(r_vec)) / 2.0;
    return known_size * sin(M_PI_2 - alpha) * (tan(alpha) + cot(alpha));
  }
  //}

  /* estimate_distance_from_depthmap() method //{ */
  float ObjectDetector::estimate_distance_from_depthmap(const cv::Point2f& area_center, float area_radius, const cv::Mat& dm_img, const cv::Mat& label_img, lut_elem_t color_id, cv::InputOutputArray dbg_img)
  {
    bool publish_debug = dbg_img.needed();
    cv::Mat dbg_mat;
    if (publish_debug)
      dbg_mat = dbg_img.getMat();
  
    float dm_dist = 0;
    size_t n_dm_samples = 0;
  
    Point tmp_topleft(area_center.x - area_radius, ceil(area_center.y - area_radius));
    Point tmp_botright(area_center.x + area_radius, ceil(area_center.y + area_radius));
    // clamp the x dimension
    if (tmp_topleft.x < 0)
    {
      tmp_topleft.x = 0;
    } else if (tmp_botright.x >= dm_img.cols)
    {
      tmp_botright.x = dm_img.cols-1;
    }
    // clamp the y dimension
    if (tmp_topleft.y < 0)
    {
      tmp_topleft.y = 0;
    } else if (tmp_botright.y >= dm_img.rows)
    {
      tmp_botright.y = dm_img.rows-1;
    }
    const cv::Rect roi(tmp_topleft, tmp_botright);
    const cv::Mat tmp_dm_img = dm_img(roi);
    cv::Mat tmp_mask;
    cv::bitwise_and(label_img(roi), cv::Scalar(color_id), tmp_mask);
    cv::Mat tmp_dbg_img = publish_debug ? dbg_mat(roi) : cv::Mat();
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
        if (depth <= m_min_depth
         || depth >= m_max_depth)
          continue;
        const float u = tmp_topleft.x + x;
        const float v = tmp_topleft.y + y;
        dm_dist += depth2range(depth, u, v, m_dm_camera_model.fx(), m_dm_camera_model.fy(), m_dm_camera_model.cx(), m_dm_camera_model.cy());
        if (std::isnan(dm_dist))
          ROS_WARN("[ObjectDetector]: Distance is nan! Skipping.");
        else
        {
          n_dm_samples++;
          if (publish_debug)
            mark_pixel(tmp_dbg_img, x, y, 2);
        }
      }
    }
    if (n_dm_samples > 0)
    {
      // calculate average from the sum
      dm_dist /= float(n_dm_samples);
      // recalculate to meters from mm
      dm_dist /= 1000.0;
    } else
    {
      dm_dist = std::numeric_limits<float>::quiet_NaN();
    }
    return dm_dist;
  }
  //}

  /* highlight_mask() method //{ */
  void ObjectDetector::highlight_mask(cv::Mat& img, cv::Mat label_img)
  {
    assert(img.size() == label_img.size());
    assert(img.channels() == 3);
    assert(label_img.channels() == 1);
    Size size = img.size();
    if (img.isContinuous() && label_img.isContinuous())
    {
      size.width *= size.height;
      size.height = 1;
    }
    for (int i = 0; i < size.height; i++)
    {
      const uint8_t* sptr = label_img.ptr<uint8_t>(i);
      uint8_t* dptr = img.ptr<uint8_t>(i);
      for (int j = 0; j < size.width; j++)
      {
        if (sptr[j])
        {
          const cv::Scalar color = color_highlight(color_id_t(sptr[j]));
          uint8_t& b = dptr[3*j + 0];
          uint8_t& g = dptr[3*j + 1];
          uint8_t& r = dptr[3*j + 2];
          r = std::clamp(int(r+color(2)), 0, 255);
          g = std::clamp(int(g+color(1)), 0, 255);
          b = std::clamp(int(b+color(0)), 0, 255);
        }
      }
    }
  }
  //}

  /* load_segmentation_config() method //{ */
  SegConf ObjectDetector::load_segmentation_config(const drcfg_t& cfg)
  {
    SegConf ret;
  
    ret.active = true;
    ret.color = color_id_t(cfg.segment_color);
    ret.color_name = color_name(ret.color);
    ret.method = bin_method_t(cfg.binarization_method);
  
    // Load HSV thresholding params
    ret.hue_center = cfg.hue_center;
    ret.hue_range = cfg.hue_range;
    ret.sat_center = cfg.sat_center;
    ret.sat_range = cfg.sat_range;
    ret.val_center = cfg.val_center;
    ret.val_range = cfg.val_range;
  
    // Load L*a*b* thresholding params
    ret.l_center = cfg.l_center;
    ret.l_range = cfg.l_range;
    ret.a_center = cfg.a_center;
    ret.a_range = cfg.a_range;
    ret.b_center = cfg.b_center;
    ret.b_range = cfg.b_range;
  
    return ret;
  }
  //}

  /* load_segmentation_config() method //{ */
  SegConf ObjectDetector::load_segmentation_config(mrs_lib::ParamLoader& pl, const std::string& cfg_name)
  {
    SegConf ret;
  
    ret.active = true;
    ret.color = color_id(cfg_name);
    ret.color_name = cfg_name;
    std::transform(ret.color_name.begin(), ret.color_name.end(), ret.color_name.begin(), ::tolower);
  
    std::string bin_method;
    pl.load_param(cfg_name + "/binarization_method", bin_method);
    ret.method = binarization_method_id(bin_method);
  
    // Load HSV thresholding params
    pl.load_param(cfg_name + "/hsv/hue_center", ret.hue_center);
    pl.load_param(cfg_name + "/hsv/hue_range", ret.hue_range);
    pl.load_param(cfg_name + "/hsv/sat_center", ret.sat_center);
    pl.load_param(cfg_name + "/hsv/sat_range", ret.sat_range);
    pl.load_param(cfg_name + "/hsv/val_center", ret.val_center);
    pl.load_param(cfg_name + "/hsv/val_range", ret.val_range);
  
    // Load L*a*b* thresholding params
    pl.load_param(cfg_name + "/lab/l_center", ret.l_center);
    pl.load_param(cfg_name + "/lab/l_range", ret.l_range);
    pl.load_param(cfg_name + "/lab/a_center", ret.a_center);
    pl.load_param(cfg_name + "/lab/a_range", ret.a_range);
    pl.load_param(cfg_name + "/lab/b_center", ret.b_center);
    pl.load_param(cfg_name + "/lab/b_range", ret.b_range);
  
    return ret;
  }
  //}

  /* update_drcfg() method //{ */
  void ObjectDetector::update_drcfg(const SegConf& seg_conf)
  {
    drcfg_t cfg = m_drmgr_ptr->config;
  
    cfg.segment_color = seg_conf.color;
    cfg.binarization_method = seg_conf.method;
  
    // Load HSV thresholding params
    cfg.hue_center = seg_conf.hue_center;
    cfg.hue_range = seg_conf.hue_range;
    cfg.sat_center = seg_conf.sat_center;
    cfg.sat_range = seg_conf.sat_range;
    cfg.val_center = seg_conf.val_center;
    cfg.val_range = seg_conf.val_range;
  
    // Load L*a*b* thresholding params
    cfg.l_center = seg_conf.l_center;
    cfg.l_range = seg_conf.l_range;
    cfg.a_center = seg_conf.a_center;
    cfg.a_range = seg_conf.a_range;
    cfg.b_center = seg_conf.b_center;
    cfg.b_range = seg_conf.b_range;
  
    m_drmgr_ptr->update_config(cfg);
  }
  //}

  /* load_color_configs() method //{ */
  std::vector<SegConf> ObjectDetector::load_color_configs(mrs_lib::ParamLoader& pl, const std::string& colors_str)
  {
    std::vector<SegConf> seg_confs;
    std::stringstream ss(colors_str);
    std::string color_name;
    while (std::getline(ss, color_name))
    {
      // remove whitespaces
      color_name.erase(std::remove_if(color_name.begin(), color_name.end(), ::isspace), color_name.end());
      // skip empty color_names (such as the last one)
      if (!color_name.empty())
      {
        SegConf seg_conf = load_segmentation_config(pl, color_name);
        seg_confs.push_back(seg_conf);
      }
    }
    return seg_confs;
  }
  //}

  /* onInit() method //{ */

  void ObjectDetector::onInit()
  {

    ROS_INFO("[%s]: Initializing ------------------------------", m_node_name.c_str());
    ros::NodeHandle nh = nodelet::Nodelet::getMTPrivateNodeHandle();
    ros::Time::waitForValid();

    /** Load parameters from ROS * //{*/
    string node_name = ros::this_node::getName().c_str();
    ParamLoader pl(nh, m_node_name);

    // LOAD DYNAMIC PARAMETERS
    m_drmgr_ptr = std::make_unique<drmgr_t>(nh, m_node_name);

    // LOAD STATIC PARAMETERS
    ROS_INFO("[%s]: Loading static parameters:", m_node_name.c_str());
    // Load the detection parameters
    pl.load_param("object_radius", m_object_radius, -1.0);
    m_object_radius_known = m_object_radius > 0;
    pl.load_param("max_dist", m_max_dist);
    pl.load_param("max_dist_diff", m_max_dist_diff);
    pl.load_param("min_depth", m_min_depth);
    pl.load_param("max_depth", m_max_depth);

    /* load covariance coefficients //{ */
    
    // admittedly a pretty fcking overcomplicated way to do this, but I'm home bored, so whatever
    {
      std::map<std::string, double> cov_coeffs_xy = pl.load_param2<std::map<std::string, double>>("cov_coeffs/xy");
      std::map<std::string, double> cov_coeffs_z = pl.load_param2<std::map<std::string, double>>("cov_coeffs/z");
      std::map<dist_qual_t, double> loaded_vals_xy;
      for (const auto& keyval : cov_coeffs_xy)
      {
        int val;
        if ((val = dist_qual_id(keyval.first)) == dist_qual_t::unknown_qual)
        {
          ROS_ERROR("[%s]: Unknwown distance estimation quality key: '%s'! Skipping.", m_node_name.c_str(), keyval.first.c_str());
          continue;
        }
        dist_qual_t enval = (dist_qual_t)val;
        loaded_vals_xy.insert({enval, keyval.second});
      }
      std::map<dist_qual_t, double> loaded_vals_z;
      for (const auto& keyval : cov_coeffs_z)
      {
        int val;
        if ((val = dist_qual_id(keyval.first)) == dist_qual_t::unknown_qual)
        {
          ROS_ERROR("[%s]: Unknwown distance estimation quality key: '%s'! Skipping.", m_node_name.c_str(), keyval.first.c_str());
          continue;
        }
        dist_qual_t enval = (dist_qual_t)val;
        loaded_vals_z.insert({enval, keyval.second});
      }
      for (const auto& keyval : loaded_vals_xy)
      {
        auto it = std::end(loaded_vals_z);
        if ((it = loaded_vals_z.find(keyval.first)) == std::end(loaded_vals_z))
        {
          ROS_ERROR("[%s]: Distance estimation quality key '%s' was specified for xy but not for z! Skipping.", m_node_name.c_str(), dist_qual_name(keyval.first).c_str());
          continue;
        }
        m_cov_coeffs.insert({keyval.first, {keyval.second, it->second}});
        ROS_INFO("[%s]: Inserting xyz coeffs: [%.2f, %.2f, %.2f].", m_node_name.c_str(), keyval.second, keyval.second, it->second);
      }
      for (const auto& keyval : loaded_vals_z)
        if (loaded_vals_xy.find(keyval.first) == std::end(loaded_vals_xy))
          ROS_ERROR("[%s]: Distance estimation quality key '%s' was specified for xy but not for z! Skipping.", m_node_name.c_str(), dist_qual_name(keyval.first).c_str());
          // just warn the user, not much else we can do
    }
    
    //}

    double loop_rate = pl.load_param2<double>("loop_rate", 100);
    std::string colors_str;
    pl.load_param("colors", colors_str);
    m_seg_confs = load_color_configs(pl, colors_str);
    // load the desired segmentation color as text and convert to ID
    std::string segment_color_text = pl.load_param2<std::string>("segment_color_text");
    drcfg_t drcfg = m_drmgr_ptr->config;
    drcfg.segment_color = color_id(segment_color_text);
    m_drmgr_ptr->update_config(drcfg);

    if (!m_drmgr_ptr->loaded_successfully())
    {
      ROS_ERROR("[%s]: Some default values of dynamically reconfigurable parameters were not loaded successfully, ending the node", m_node_name.c_str());
      ros::shutdown();
    }

    if (!pl.loaded_successfully())
    {
      ROS_ERROR("[%s]: Some compulsory parameters were not loaded successfully, ending the node", m_node_name.c_str());
      ros::shutdown();
    }
    //}
    
    /** Create publishers and subscribers //{**/
    // Initialize transform listener
    m_tf_listener_ptr = std::make_unique<tf2_ros::TransformListener>(m_tf_buffer, m_node_name);
    // Initialize other subs and pubs
    SubscribeMgr smgr(nh);
    m_sh_dm = smgr.create_handler_threadsafe<sensor_msgs::ImageConstPtr>("dm_image", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
    m_sh_dm_cinfo = smgr.create_handler_threadsafe<sensor_msgs::CameraInfo>("dm_camera_info", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
    m_sh_rgb = smgr.create_handler_threadsafe<sensor_msgs::ImageConstPtr>("rgb_image", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
    m_sh_rgb_cinfo = smgr.create_handler_threadsafe<sensor_msgs::CameraInfo>("rgb_camera_info", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
  
    m_pub_debug = nh.advertise<sensor_msgs::Image&>("debug_image", 1);
    m_pub_pcl = nh.advertise<sensor_msgs::PointCloud>("detected_objects_pcl", 10);
    m_pub_det = nh.advertise<object_detect::PoseWithCovarianceArrayStamped>("detected_objects", 10);

    if (!smgr.loaded_successfully())
    {
      ROS_ERROR("Unable to subscribe to some topics, ending the node");
      ros::shutdown();
    }

    m_color_change_server = nh.advertiseService("change_colors", &ObjectDetector::color_change_callback, this);
    m_color_query_server = nh.advertiseService("query_colors", &ObjectDetector::color_query_callback, this);
    //}

    /* profiler //{ */

    m_profiler_ptr = std::make_unique<mrs_lib::Profiler>(nh, m_node_name, false);

    //}

    m_active_seg_confs = get_segmentation_configs(m_seg_confs, {m_drmgr_ptr->config.segment_color});
    m_prev_color_id = m_drmgr_ptr->config.segment_color;
    {
      const auto lut_start_time = ros::WallTime::now();
      ROS_INFO("[%s]: Generating lookup table", m_node_name.c_str());
      generate_lut(m_cur_lut, m_seg_confs);
      const auto lut_end_time = ros::WallTime::now();
      const auto lut_dur = lut_end_time - lut_start_time;
      ROS_INFO("[%s]: Lookup table generated in %fs", m_node_name.c_str(), lut_dur.toSec());
    }
    m_is_initialized = true;

    /* timers  //{ */

    m_main_loop_timer = nh.createTimer(ros::Rate(loop_rate), &ObjectDetector::main_loop, this);
    m_drcfg_update_loop_timer = nh.createTimer(ros::Rate(2.0), &ObjectDetector::drcfg_update_loop, this);

    //}

    ROS_INFO("[%s]: Initialized ------------------------------", m_node_name.c_str());
  }

  //}

  /* ObjectDetector::color_change_callback() method //{ */
  
  bool ObjectDetector::color_change_callback(object_detect::ColorChange::Request& req, object_detect::ColorChange::Response& resp)
  {
    std::vector<std::string> known_colornames;
    std::vector<SegConf> seg_confs;
    std::vector<std::string> unknown_colornames;
    for (const auto& color_name : req.detectColors)
    {
      std::string lowercase = color_name;
      std::transform(lowercase.begin(), lowercase.end(), lowercase.begin(), ::tolower);
      std::vector<SegConf> cur_seg_confs;
      for (const auto& seg_conf : m_seg_confs)
      {
        if (seg_conf.color_name == lowercase)
          cur_seg_confs.push_back(seg_conf);
      }
      if (cur_seg_confs.empty())
      {
        unknown_colornames.push_back(lowercase);
      } else
      {
        known_colornames.push_back(lowercase);
        seg_confs.insert(std::end(seg_confs), std::begin(cur_seg_confs), std::end(cur_seg_confs));
      }
    }
  
    if (unknown_colornames.empty())
    {
      m_active_seg_confs = seg_confs;
      std::stringstream ss;
      ss << "Detecting colors: [";
      for (size_t it = 0; it < known_colornames.size(); it++)
      {
        ss << known_colornames.at(it);
        if (it < known_colornames.size()-1)
          ss << ", ";
      }
      ss << "]";
      resp.message = ss.str();
      resp.success = true;
      NODELET_INFO("[%s]: %s", m_node_name.c_str(), resp.message.c_str());
    } else
    {
      std::stringstream ss;
      ss << "Unknown colors: [";
      for (size_t it = 0; it < unknown_colornames.size(); it++)
      {
        ss << unknown_colornames.at(it);
        if (it < unknown_colornames.size()-1)
          ss << ", ";
      }
      ss << "]";
      resp.message = ss.str();
      resp.success = false;
      NODELET_WARN("[%s]: %s", m_node_name.c_str(), resp.message.c_str());
    }
    return true;
  }
  
  //}

  /* ObjectDetector::color_query_callback() method //{ */
  
  bool ObjectDetector::color_query_callback(object_detect::ColorQuery::Request& req, object_detect::ColorQuery::Response& resp)
  {
    resp.colors.reserve(m_seg_confs.size());
    for (const auto& seg_conf : m_seg_confs)
    {
      resp.colors.push_back(seg_conf.color_name);
    }
    return true;
  }
  
  //}

}

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(object_detect::ObjectDetector, nodelet::Nodelet)
