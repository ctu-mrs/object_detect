#include <list>

#include <ros/ros.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <image_geometry/pinhole_camera_model.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <mrs_lib/ParamLoader.h>
#include <mrs_lib/subscribe_handler.h>
#include "SegConf.h"

using namespace cv;
using namespace std;
using namespace object_detect;

/* helper functions etc //{ */

bool show_distance = true;
bool show_dist_quality = true;
bool show_correction_delay = true;
struct Option {const int key; bool& option; const std::string txt, op1, op2;};
static const std::vector<Option> options =
{
  {'d', show_distance, "showing distance", "not ", ""},
  {'q', show_dist_quality, "showing distance estimate quality", "not ", ""},
  {'t', show_correction_delay, "showing time since last correction", "not ", ""},
};
void print_options()
{
  ROS_INFO("Options (change by selecting the OpenCV window and pressing the corresponding key)");
  std::cout << "key:\ttoggles:" << std::endl;
  std::cout << "----------------------------" << std::endl;
  for (const auto& opt : options)
  {
    std::cout << ' ' << char(opt.key) << '\t' << opt.txt << std::endl;
  }
}
void eval_keypress(int key)
{
  for (const auto& opt : options)
  {
    if (key == opt.key)
    {
      ROS_INFO(("%s" + opt.txt).c_str(), opt.option?opt.op1.c_str():opt.op2.c_str());
      opt.option = !opt.option;
    }
  }
}

std::string to_str_prec(double num, unsigned prec = 3)
{
  std::stringstream strstr;
  strstr << std::fixed << std::setprecision(prec);
  strstr << num;
  return strstr.str();
}

template <typename T>
void add_to_buffer(T img, std::list<T>& bfr)
{
  bfr.push_back(img);
  if (bfr.size() > 100)
    bfr.pop_front();
}

template <class T>
T find_closest(ros::Time stamp, std::list<T>& bfr)
{
  T closest;
  double closest_diff;
  bool closest_set = false;

  for (auto& imptr : bfr)
  {
    double cur_diff = abs((imptr->header.stamp - stamp).toSec());

    if (!closest_set || cur_diff < closest_diff)
    {
      closest = imptr;
      closest_diff = cur_diff;
      closest_set = true;
    }
  }
  return closest;
}

int get_dist_qual_channel(sensor_msgs::PointCloudConstPtr pc)
{
  int dist_qual_channel = -1;
  for (size_t it = 0; it < pc->channels.size(); it++)
  {
    if (pc->channels[it].name == "distance_quality")
      dist_qual_channel = it;
  }
  return dist_qual_channel;
}

//}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "backproject_display");
  ROS_INFO("Node initialized.");

  ros::NodeHandle nh = ros::NodeHandle("~");

  {
    std::cout << "Waiting for valid time..." << std::endl;
    ros::Rate r(10);
    while (!ros::Time::isValid())
    {
      r.sleep();
      ros::spinOnce();
    }
  }

  mrs_lib::ParamLoader pl(nh);
  const double object_radius = pl.load_param2<double>("object_radius");

  mrs_lib::SubscribeMgr smgr(nh, "backproject_display");
  auto sh_pc = smgr.create_handler<sensor_msgs::PointCloud>("detections", ros::Duration(5.0));
  auto sh_img = smgr.create_handler<sensor_msgs::Image>("image_rect", ros::Duration(5.0));
  auto sh_cinfo = smgr.create_handler<sensor_msgs::CameraInfo>("camera_info", ros::Duration(5.0));
  auto sh_ball = smgr.create_handler<geometry_msgs::PoseWithCovarianceStamped>("chosen_balloon", ros::Duration(5.0));

  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener(tf_buffer);

  print_options();

  int window_flags = WINDOW_AUTOSIZE | WINDOW_KEEPRATIO | WINDOW_GUI_NORMAL;
  std::string window_name = "backprojected_localization";
  cv::namedWindow(window_name, window_flags);
  image_geometry::PinholeCameraModel camera_model;
  ros::Rate r(100);

  std::list<sensor_msgs::ImageConstPtr> img_buffer;

  while (ros::ok())
  {
    ros::spinOnce();

    if (sh_cinfo->has_data() && !sh_cinfo->used_data())
      camera_model.fromCameraInfo(sh_cinfo->get_data());

    if (sh_img->new_data())
      add_to_buffer(sh_img->get_data(), img_buffer);

    if (sh_img->has_data() && sh_cinfo->used_data())
    {
      cv::Mat img;
      if (sh_pc->has_data())
      {
        ros::Time cur_det_t = sh_pc->get_data()->header.stamp;
        sensor_msgs::PointCloudConstPtr dets_msg = sh_pc->get_data();
        sensor_msgs::ImageConstPtr img_ros = find_closest(cur_det_t, img_buffer);

        geometry_msgs::TransformStamped transform;
        try
        {
          transform = tf_buffer.lookupTransform(img_ros->header.frame_id, dets_msg->header.frame_id, dets_msg->header.stamp, ros::Duration(1.0));
        } catch (tf2::TransformException& ex)
        {
          ROS_WARN("Error during transform from \"%s\" frame to \"%s\" frame.\n\tMSG: %s", dets_msg->header.frame_id.c_str(), img_ros->header.frame_id.c_str(), ex.what());
          continue;
        }

        const cv_bridge::CvImagePtr img_ros2 = cv_bridge::toCvCopy(img_ros, "bgr8");
        img = img_ros2->image;
        /* cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); */
        /* cv::cvtColor(img, img, cv::COLOR_GRAY2BGR); */
        size_t channel_quality_it = get_dist_qual_channel(dets_msg);
        if (channel_quality_it < 0)
          ROS_WARN("distance_quality channel not found");

        for (size_t it = 0; it < dets_msg->points.size(); it++)
        {
          cv::Point prev_pt2d;
          cv::Point3d pt3d;
          cv::Scalar color;

          const geometry_msgs::Point32& point_float = dets_msg->points[it];
          geometry_msgs::Point point_orig;
          point_orig.x = point_float.x; point_orig.y = point_float.y; point_orig.z = point_float.z;
          geometry_msgs::Point point_transformed;
          tf2::doTransform(point_orig, point_transformed, transform);
        
          pt3d.x = point_transformed.x;
          pt3d.y = point_transformed.y;
          pt3d.z = point_transformed.z;
          const cv::Point pt2d = camera_model.project3dToPixel(pt3d);
        
          /* color = get_color(hyp_msg.position_sources[it]); */
          color = cv::Scalar(255, 0, 0);
          /* const int thickness = is_main ? 2 : 1; */
          const int thickness = -1;
          const int size = 10;
        
          cv::circle(img, pt2d, size, color, thickness);
          prev_pt2d = pt2d;

          // display info
          {
            const double dist = sqrt(pt3d.x*pt3d.x + pt3d.y*pt3d.y + pt3d.z*pt3d.z);
            int li = 0;        // line iterator
            const int ls = 15; // line step
            const cv::Point lo = prev_pt2d + cv::Point(45, -45);
            if (show_distance)
              cv::putText(img, "distance:   " + to_str_prec(dist) + "m", lo+cv::Point(0, li++*ls), FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            if (show_dist_quality)
              cv::putText(img, "dist. qual: " + to_str_prec(dets_msg->channels[channel_quality_it].values.at(it)), lo+cv::Point(0, li++*ls), FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            /* if (show_dist_quality) */
            /*   cv::putText(img, "distance: " + to_str_prec(dist) + "m", lo+cv::Point(0, li++*ls), FONT_HERSHEY_SIMPLEX, 0.5, color, 1); */
          }
        } // for (const auto& hyp_msg : hyps_msg.hypotheses)

        if (sh_ball->new_data())
        {
          geometry_msgs::PoseWithCovarianceStamped pt = *(sh_ball->get_data());
          geometry_msgs::TransformStamped transform;
          bool got_transform = true;
          try
          {
            transform = tf_buffer.lookupTransform(img_ros->header.frame_id, pt.header.frame_id, pt.header.stamp, ros::Duration(1.0));
          } catch (tf2::TransformException& ex)
          {
            ROS_WARN("Error during transform from \"%s\" frame to \"%s\" frame.\n\tMSG: %s", pt.header.frame_id.c_str(), img_ros->header.frame_id.c_str(), ex.what());
            got_transform = false;
          }
          if (got_transform)
          {
            geometry_msgs::Point point_transformed;
            tf2::doTransform(pt.pose.pose.position, point_transformed, transform);

            cv::Point prev_pt2d;
            cv::Point3d pt3d;
            pt3d.x = point_transformed.x;
            pt3d.y = point_transformed.y;
            pt3d.z = point_transformed.z;
            const cv::Point pt2d = camera_model.project3dToPixel(pt3d);
            cv::circle(img, pt2d, 5, cv::Scalar(0, 0, 255), -1);
          }
        }
      } else // if (sh_pc->has_data())
      {
        sensor_msgs::ImageConstPtr img_ros = img_buffer.back();
        cv_bridge::CvImageConstPtr img_ros2 = cv_bridge::toCvShare(img_ros, "bgr8");
        img = img_ros2->image;
      }

      if (img.empty())
        continue;
      cv::imshow(window_name, img);
      eval_keypress(cv::waitKey(1));
    }

    r.sleep();
  }
}

