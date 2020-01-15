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
      const ros::WallTime start_t = ros::WallTime::now();
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

      /* get ball candidates //{ */
      std::vector<BallCandidate> balls;
      {
        NODELET_INFO("[ObjectDetector]: Segmenting %s color (by %s)",
            color_name(color_id_t(m_drmgr_ptr->config.segment_color)).c_str(),
            binarization_method_name(bin_method_t(m_drmgr_ptr->config.binarization_method)).c_str());
        m_blob_det.set_drcfg(m_drmgr_ptr->config);
        balls = m_blob_det.detect_candidates(rgb_img, label_img);
      }
      if (publish_debug)
        highlight_mask(dbg_img, label_img);
      //}

      /* Calculate 3D positions of the detected balls //{ */
      vector<dist_qual_t> distance_qualities;

      for (size_t it = 0; it < balls.size(); it++)
      {
        cout << "[" << m_node_name << "]: Processing object " << it + 1 << "/" << balls.size() << " --------------------------" << std::endl;

        BallCandidate& ball = balls.at(it);
        const cv::Point& center = ball.location;
        const float radius = ball.radius;
        const bool physical_radius_known = m_drmgr_ptr->config.ball_physical_diameter > 0.0;
        std::string name_upper = color_name(color_id_t(ball.type));
        std::transform(name_upper.begin(), name_upper.end(), name_upper.begin(), ::toupper);
        cout << "object classified as " << name_upper << " ball" << std::endl;

        /* Calculate 3D vector pointing to left and right edges of the detected object //{ */
        const Eigen::Vector3f l_vec = project(center.x - radius*cos(M_PI_4), center.y - radius*sin(M_PI_4), m_rgb_camera_model);
        const Eigen::Vector3f r_vec = project(center.x + radius*cos(M_PI_4), center.y + radius*sin(M_PI_4), m_rgb_camera_model);
        //}
        
        /* Estimate distance based on known size of object if applicable //{ */
        float estimated_distance = 0.0f;
        bool estimated_distance_valid = false;
        if (physical_radius_known)
        {
          estimated_distance = estimate_distance_from_known_diameter(l_vec, r_vec, m_drmgr_ptr->config.ball_physical_diameter);
          cout << "Estimated distance: " << estimated_distance << endl;
          estimated_distance_valid = distance_valid(estimated_distance);
        }
        //}

        /* Get distance from depthmap if applicable //{ */
        float depthmap_distance;
        bool depthmap_distance_valid = false;
        if (dm_ready)
        {
          depthmap_distance = estimate_distance_from_depthmap(center, radius, m_drmgr_ptr->config.distance_min_valid_pixels_ratio, dm_img, (publish_debug && publish_debug_dm) ? dbg_img : cv::noArray());
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
          const Eigen::Vector3f pos_vec = resulting_distance * ((l_vec + r_vec) / 2.0).normalized();
          cout << "Estimated location (camera CS): [" << pos_vec(0) << ", " << pos_vec(1) << ", " << pos_vec(2) << "]" << std::endl;
          ball.position = pos_vec;
          /* cout << "Estimated location (camera CS): [" << ball.position(0) << ", " << ball.position(1) << ", " << ball.position(2) << "]" << std::endl; */
          //}
        }
        ball.dist_qual = resulting_distance_quality;

        cout << "[" << m_node_name << "]: Done with object " << it + 1 << "/" << balls.size() << " --------------------------" << std::endl;
      }
      //}

      /* Publish all the calculated valid positions //{ */
      if (m_pub_det.getNumSubscribers() > 0)
      {
        const object_detect::BallDetections det_msg = to_output_message(balls, rgb_img_msg->header);
        m_pub_det.publish(det_msg);
      }
      //}

      /* Publish all the calculated valid positions //{ */
      if (m_pub_pcl.getNumSubscribers() > 0)
      {
        sensor_msgs::PointCloud2 pcl_msg;
        pcl_msg.header = rgb_img_msg->header;

        sensor_msgs::PointCloud2Modifier mdf(pcl_msg);
        mdf.resize(balls.size());
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

        for (size_t it = 0; it < balls.size(); ++it, ++iter_x, ++iter_y, ++iter_z, ++iter_type, ++iter_dist_qual)
        {
          const auto& ball = balls.at(it);
          cerr << "adding position to pcl: " << ball.position << std::endl;
          *iter_x = ball.position.x();
          *iter_y = ball.position.y();
          *iter_z = ball.position.z();
          *iter_type = ball.type;
          *iter_dist_qual = ball.dist_qual;
        }

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
      ros::WallDuration dur = ros::WallTime::now() - start_t;
      ros::Duration del = ros::Time::now() - rgb_img_msg->header.stamp;
      static double fps = 1.0 / dur.toSec();
      if (dur.toSec() > 0.0)
        fps = 0.1*(1.0 / dur.toSec()) + 0.9*fps;
      cout << "processing FPS: " << fps << "Hz" << std::endl;
      cout << "delay: " << del.toSec() * 1000.0 << "ms (processing: " << dur.toSec() * 1000.0 << "ms)" << std::endl;
      //}

      ROS_INFO("[%s]: Image processed", m_node_name.c_str());
    }
    
  }
  //}

  /* to_output_message() method //{ */
  object_detect::BallDetections ObjectDetector::to_output_message(const std::vector<BallCandidate>& balls, const std_msgs::Header& header) const
  {
    object_detect::BallDetections ret;
    ret.detections.reserve(balls.size());
  
    for (const auto& ball : balls)
    {
      object_detect::BallDetection det;
      const auto qual = ball.dist_qual;
      const auto pos = ball.position;
  
      ros_pose_t pose;
      pose.position.x = pos.x();
      pose.position.y = pos.y();
      pose.position.z = pos.z();
      pose.orientation.x = 0;
      pose.orientation.y = 0;
      pose.orientation.z = 0;
      pose.orientation.w = 1;
  
      if (m_cov_coeffs.find(qual) == std::end(m_cov_coeffs))
      {
        ROS_ERROR("[ObjectDetector]: Invalid distance estimate quality %d! Skipping detection.", qual);
        continue;
      }
      auto [xy_covariance_coeff, z_covariance_coeff] = m_cov_coeffs.at(qual);
      const ros_cov_t cov = generate_covariance(pos, xy_covariance_coeff, z_covariance_coeff);
  
      geometry_msgs::PoseWithCovariance pose_with_cov;
      pose_with_cov.pose = pose;
      pose_with_cov.covariance = cov;
      det.pose = pose_with_cov;
      det.type = ball.type;
      ret.detections.push_back(det);
    }
    ret.header = header;
    return ret;
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

  /* distance_valid() method //{ */
  bool ObjectDetector::distance_valid(float distance)
  {
    return !(isnan(distance) || distance < 0.0 || distance > m_max_dist);
  }
  //}

  /* estimate_distance_from_known_diameter() method //{ */
  float ObjectDetector::estimate_distance_from_known_diameter(const Eigen::Vector3f& l_vec, const Eigen::Vector3f& r_vec, float physical_diameter)
  {
    const Eigen::Vector3f c_vec = (l_vec + r_vec) / 2.0f;
    const float dot = c_vec.dot(l_vec)/(c_vec.norm() * l_vec.norm());
    const float dist = physical_diameter/std::sqrt(1.0f - dot*dot)/2.0f;
    return dist;
  }
  //}

  /* estimate_distance_from_depthmap() method //{ */
  float ObjectDetector::estimate_distance_from_depthmap(const cv::Point2f& area_center, const float area_radius, const double min_valid_ratio, const cv::Mat& dm_img, cv::InputOutputArray dbg_img)
  {
    bool publish_debug = dbg_img.needed();
    cv::Mat dbg_mat;
    if (publish_debug)
      dbg_mat = dbg_img.getMat();
  
    const cv::Rect roi = circle_roi_clamped(area_center, area_radius, dm_img.size());
    const cv::Mat tmp_mask = circle_mask(area_center, area_radius, roi);
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
        // check if this pixel is part of a detected blob
        if (!tmp_mask.at<uint8_t>(y, x))
          continue;
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
    pl.load_param("max_dist", m_max_dist);
    pl.load_param("max_dist_diff", m_max_dist_diff);
    pl.load_param("min_depth", m_min_depth);
    pl.load_param("max_depth", m_max_depth);
    std::string ocl_lut_kernel_file = pl.load_param2<std::string>("ocl_lut_kernel_file");
    bool use_ocl = pl.load_param2<bool>("use_ocl");

    /* load covariance coefficients //{ */
    
    // admittedly a pretty fcking overcomplicated way to do this, but I'm home bored, so whatever
    std::pair<double&, double&> cov_coeffs_no_estimate = {m_drmgr_ptr->config.cov_coeffs__xy__no_estimate, m_drmgr_ptr->config.cov_coeffs__z__no_estimate};
    std::pair<double&, double&> cov_coeffs_blob_size = {m_drmgr_ptr->config.cov_coeffs__xy__blob_size, m_drmgr_ptr->config.cov_coeffs__z__blob_size};
    std::pair<double&, double&> cov_coeffs_depthmap = {m_drmgr_ptr->config.cov_coeffs__xy__depthmap, m_drmgr_ptr->config.cov_coeffs__z__depthmap};
    std::pair<double&, double&> cov_coeffs_both = {m_drmgr_ptr->config.cov_coeffs__xy__both, m_drmgr_ptr->config.cov_coeffs__z__both};
    m_cov_coeffs.insert({dist_qual_t::no_estimate, cov_coeffs_no_estimate});
    m_cov_coeffs.insert({dist_qual_t::blob_size, cov_coeffs_blob_size});
    m_cov_coeffs.insert({dist_qual_t::depthmap, cov_coeffs_depthmap});
    m_cov_coeffs.insert({dist_qual_t::both, cov_coeffs_both});
    
    //}

    double loop_rate = pl.load_param2<double>("loop_rate", 100);

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
    // Initialize other subs and pubs
    SubscribeMgr smgr(nh);
    m_sh_dm = smgr.create_handler<sensor_msgs::Image>("dm_image", ros::Duration(5.0));
    m_sh_dm_cinfo = smgr.create_handler<sensor_msgs::CameraInfo>("dm_camera_info", ros::Duration(5.0));
    m_sh_rgb = smgr.create_handler<sensor_msgs::Image>("rgb_image", ros::Duration(5.0));
    m_sh_rgb_cinfo = smgr.create_handler<sensor_msgs::CameraInfo>("rgb_camera_info", ros::Duration(5.0));
  
    image_transport::ImageTransport it(nh);
    m_pub_debug = it.advertise("debug_image", 1);
    m_pub_pcl = nh.advertise<sensor_msgs::PointCloud2>("detected_balls_pcl", 10);
    m_pub_det = nh.advertise<object_detect::BallDetections>("detected_balls", 10);
    //}

    /* profiler //{ */

    m_profiler_ptr = std::make_unique<mrs_lib::Profiler>(nh, m_node_name, false);

    //}

    m_prev_color_id = m_drmgr_ptr->config.segment_color;
    {
      const auto lut_start_time = ros::WallTime::now();
      ROS_INFO("[%s]: Generating lookup table", m_node_name.c_str());
      const auto lut = generate_lut(m_drmgr_ptr->config);
      const auto lut_end_time = ros::WallTime::now();
      const auto lut_dur = lut_end_time - lut_start_time;
      ROS_INFO("[%s]: Lookup table generated in %fs", m_node_name.c_str(), lut_dur.toSec());

      if (use_ocl)
      {
        m_blob_det = BlobDetector(ocl_lut_kernel_file, lut);
        ROS_INFO("[%s]: Using OpenCL HW acceleration.", m_node_name.c_str());
      } else
      {
        m_blob_det = BlobDetector(lut);
      }
    }

    m_is_initialized = true;

    /* timers  //{ */

    m_main_loop_timer = nh.createTimer(ros::Rate(loop_rate), &ObjectDetector::main_loop, this);

    //}

    ROS_INFO("[%s]: Initialized ------------------------------", m_node_name.c_str());
  }

  //}

}

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(object_detect::ObjectDetector, nodelet::Nodelet)
