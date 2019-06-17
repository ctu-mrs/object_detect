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
      cv::Mat thresholded_img;
      //}

      /* Prepare debug image if needed //{ */
      bool publish_debug = m_pub_debug.getNumSubscribers() > 0;
      cv::Mat dbg_img;
      if (publish_debug)
      {
        cv::cvtColor(rgb_img, dbg_img, cv::COLOR_BGR2GRAY);
        cv::cvtColor(dbg_img, dbg_img, cv::COLOR_GRAY2BGR);
      }
      //}

      /* Detect blobs of required color in the RGB image //{ */
      std::vector<SegConf> active_seg_confs = get_active_segmentation_configs(m_seg_confs, m_drmgr_ptr->config.segment_color);
      for (const auto& seg_conf : active_seg_confs)
        NODELET_INFO("[ObjectDetector]: Segmenting %s color", color_name(seg_conf.color).c_str());
      BlobDetector blob_det(m_drmgr_ptr->config);
      const vector<Blob> blobs = blob_det.detect(rgb_img, active_seg_confs, thresholded_img);
      if (publish_debug)
        highlight_mask(dbg_img, thresholded_img, cv::Scalar(0, 0, 128));
      //}

      /* Calculate 3D positions of the detected blobs //{ */
      const size_t n_dets = blobs.size();
      vector<geometry_msgs::Point32> positions;
      vector<float> distance_qualities;
      positions.reserve(n_dets);

      for (size_t it = 0; it < n_dets; it++)
      {
        cout << "[" << m_node_name << "]: Processing object " << it + 1 << "/" << n_dets << " --------------------------" << std::endl;

        const Blob& blob = blobs.at(it);
        const cv::Point& center = blob.location;
        const float radius = blob.radius;

        if (publish_debug)
          cv::circle(dbg_img, center, radius, Scalar(0, 255, 0), 2);

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
          depthmap_distance = estimate_distance_from_depthmap(center, radius, dm_img, thresholded_img);
          cout << "Depthmap distance: " << depthmap_distance << endl;
          depthmap_distance_valid = distance_valid(depthmap_distance);
        }
        //}

        /* Evaluate the resulting distance and its quality //{ */
        float resulting_distance = 0.0f;
        int resulting_distance_quality = 0;
        if (estimated_distance_valid && depthmap_distance_valid)
        {
          if (abs(depthmap_distance - estimated_distance) < m_max_dist_diff)
          {
            resulting_distance = depthmap_distance; // assuming this is a more precise distance estimate
            resulting_distance_quality  = 3;
          }
        } else if (depthmap_distance_valid)
        {
          resulting_distance = depthmap_distance;
          resulting_distance_quality  = 2;
        } else if (estimated_distance_valid)
        {
          resulting_distance = estimated_distance;
          resulting_distance_quality  = 1;
        }
        //}

        cout << "Estimated distance used: " << m_object_radius_known
             << ", valid: " << estimated_distance_valid
             << ", depthmap distance valid: " << depthmap_distance_valid
             << ", resulting quality: " << resulting_distance_quality << std::endl;

        if (resulting_distance_quality > 0)
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
      {
        sensor_msgs::PointCloud pcl_msg;
        pcl_msg.header = rgb_img_msg->header;
        pcl_msg.points = positions;

        sensor_msgs::ChannelFloat32 chl_msg;
        chl_msg.name = "distance_quality"s;
        chl_msg.values = distance_qualities;
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

  std::vector<SegConf> ObjectDetector::get_active_segmentation_configs(const std::vector<SegConf>& all_seg_confs, int color_id)
  {
    if (color_id < 0)
      return all_seg_confs;

    std::vector<SegConf> ret;
    for (const auto& seg_conf : all_seg_confs)
    {
      if (seg_conf.color == color_id)
        ret.push_back(seg_conf);
    }
    return ret;
  }

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
  float ObjectDetector::estimate_distance_from_depthmap(const cv::Point2f& area_center, float area_radius, const cv::Mat& dm_img, const cv::Mat& binary_img, cv::InputOutputArray dbg_img)
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
    const cv::Mat tmp_mask = binary_img(roi);
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
        n_dm_samples++;
        if (publish_debug)
          mark_pixel(tmp_dbg_img, x, y, 2);
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
  void ObjectDetector::highlight_mask(cv::Mat& img, cv::Mat mask, cv::Scalar color)
  {
    assert(img.size() == mask.size());
    assert(img.channels() == 3);
    assert(mask.channels() == 1);
    Size size = img.size();
    if (img.isContinuous() && mask.isContinuous())
    {
      size.width *= size.height;
      size.height = 1;
    }
    for (int i = 0; i < size.height; i++)
    {
      const uint8_t* sptr = mask.ptr<uint8_t>(i);
      uint8_t* dptr = img.ptr<uint8_t>(i);
      for (int j = 0; j < size.width; j++)
      {
        if (sptr[j])
        {
          uint8_t& r = dptr[3*j + 0];
          uint8_t& g = dptr[3*j + 1];
          uint8_t& b = dptr[3*j + 2];
          r = std::clamp(int(r+color(0)), 0, 255);
          g = std::clamp(int(g+color(1)), 0, 255);
          b = std::clamp(int(b+color(2)), 0, 255);
        }
      }
    }
  }
  //}

  /* load_segmentation_config() method //{ */
  SegConf ObjectDetector::load_segmentation_config(mrs_lib::ParamLoader& pl, const std::string& cfg_name)
  {
    SegConf ret;
  
    ret.active = true;
    ret.color = color_id(cfg_name);
  
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
    // LOAD STATIC PARAMETERS
    ROS_INFO("Loading static parameters:");
    // Load the detection parameters
    pl.load_param("object_radius", m_object_radius, -1.0);
    m_object_radius_known = m_object_radius > 0;
    pl.load_param("max_dist", m_max_dist);
    pl.load_param("max_dist_diff", m_max_dist_diff);
    pl.load_param("min_depth", m_min_depth);
    pl.load_param("max_depth", m_max_depth);
    double loop_rate = pl.load_param2<double>("loop_rate", 100);
    std::string colors_str;
    pl.load_param("colors", colors_str);
    m_seg_confs = load_color_configs(pl, colors_str);

    // LOAD DYNAMIC PARAMETERS
    m_drmgr_ptr = std::make_unique<drmgr_t>(nh, m_node_name);

    if (!m_drmgr_ptr->loaded_successfully())
    {
      ROS_ERROR("Some default values of dynamically reconfigurable parameters were not loaded successfully, ending the node");
      ros::shutdown();
    }

    if (!pl.loaded_successfully())
    {
      ROS_ERROR("Some compulsory parameters were not loaded successfully, ending the node");
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
    m_pub_pcl = nh.advertise<sensor_msgs::PointCloud>("detected_objects", 10);

    if (!smgr.loaded_successfully())
    {
      ROS_ERROR("Unable to subscribe to some topics, ending the node");
      ros::shutdown();
    }

    //}

    /* profiler //{ */

    m_profiler_ptr = std::make_unique<mrs_lib::Profiler>(nh, m_node_name, false);

    //}

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
