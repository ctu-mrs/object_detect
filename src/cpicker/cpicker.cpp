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
#include "object_detect/lut.h"

using namespace cv;
using namespace std;
using namespace object_detect;

std::mutex global_mtx;
cv::Mat global_image;
bool global_image_valid = false;

std::mutex global_hist_mtx;
cv::Mat global_hist;
bool global_hist_valid = false;
cv::Mat global_lut;

/* helper functions etc //{ */

bool pause_img = false;
bool show_correction_delay = true;
bool clear_colors = false;
struct Option {const int key; bool& option; const std::string txt, op1, op2;};
static const std::vector<Option> options =
{
  {' ', pause_img, "pausing", "not ", ""},
  {'c', clear_colors, "clearing colors", "not ", ""},
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

cv::Mat segment_img()
{
  cv::Mat img, lut;
  {
    std::scoped_lock lck(global_mtx, global_hist_mtx);
    global_image.copyTo(img);
    global_lut.copyTo(lut);
  }
  cv::cvtColor(img, img, COLOR_BGR2HSV);
  for (auto it = img.begin<Vec3b>(); it != img.end<Vec3b>(); it++)
  {
    const uint8_t h = (*it)(0);
    const uint8_t s = (*it)(1);
    const uint8_t lutval = lut.at<uint8_t>(h/2, s/2);
    if (lutval)
      *it = Vec3b(255, 255, 255);
    else
      *it = Vec3b(0, 0, 0);
  }
  return img;
}

/* recalc_hist_hsv() //{ */

void recalc_hist_hsv(const std::vector<cv::Vec3b>& colors)
{
  cv::Mat hsv;
  cv::cvtColor(colors, hsv, COLOR_BGR2HSV);
  int hbins = 90, sbins = 128;
  int histSize[] = {hbins, sbins};
  float hranges[] = { 0, 180 };
  float sranges[] = { 0, 256 };
  const float* ranges[] = { hranges, sranges };
  int channels[] = {0, 1};
  std::scoped_lock lck(global_hist_mtx);
  calcHist( &hsv, 1, channels, Mat(), // do not use mask
           global_hist, 2, histSize, ranges,
           true, // the histogram is uniform
           true  // accumulate
           );
  global_hist_valid = true;
  std::cout << "recalculated histogram" << std::endl;
  if (global_lut.empty())
    global_lut = cv::Mat(global_hist.size(), CV_8UC1, Scalar(0));
}

//}

/* clear_hist() //{ */

void clear_hist()
{
  std::scoped_lock lck(global_hist_mtx);
  global_hist.setTo(Scalar(0.0));
  global_lut.setTo(Scalar(0));
  clear_colors = false;
}

//}

/* color mouse callback etc //{ */

void add_color_selection(cv::Point selection_start, cv::Point selection_end)
{
  std::scoped_lock lck(global_mtx);
  if (!global_image_valid)
  {
    std::cerr << "no image received yet, cannot add selection" << std::endl;
    return;
  }
  selection_start.x = std::clamp(selection_start.x, 0, global_image.cols-1);
  selection_start.y = std::clamp(selection_start.y, 0, global_image.rows-1);
  selection_end.x = std::clamp(selection_end.x, 0, global_image.cols-1);
  selection_end.y = std::clamp(selection_end.y, 0, global_image.rows-1);
  cv::Rect sel(selection_start, selection_end);

  std::vector<cv::Vec3b> colors;
  if (sel.br() == sel.tl())
  {
    colors.push_back(global_image.at<cv::Vec3b>(sel.br()));
  }
  else if (sel.br().x == sel.tl().x)
  {
    for (int it = sel.tl().x; it < sel.br().x; it++)
      colors.push_back(global_image.at<cv::Vec3b>(sel.br().y, it));
  }
  else if (sel.br().y == sel.tl().y)
  {
    for (int it = sel.tl().y; it < sel.br().y; it++)
      colors.push_back(global_image.at<cv::Vec3b>(it, sel.br().x));
  }
  else
  {
    cv::Mat roid = global_image(sel);
    for (auto it = roid.begin<cv::Vec3b>(); it != roid.end<cv::Vec3b>(); it++)
      colors.push_back(*it);
  }

  std::cout << "added " << colors.size() << " points to histogram" << std::endl;
  recalc_hist_hsv(colors);
}

cv::Point cursor_pos;
cv::Point selection_start;
cv::Point selection_end;
bool prev_lmouse = false;
void color_mouse_callback([[maybe_unused]] int event, int x, int y, [[maybe_unused]]int flags, [[maybe_unused]]void* userdata)
{
  cursor_pos = cv::Point(x, y);
  const bool lmouse = flags & EVENT_FLAG_LBUTTON;
  if (lmouse)
  {
    selection_end = cursor_pos;
  }
  else
  {
    if (prev_lmouse)
      add_color_selection(selection_start, selection_end);
    selection_start = cursor_pos;
  }
  prev_lmouse = lmouse;
}

//}

/* hist mouse callback etc //{ */

void add_hist_selection(cv::Point selection_start, cv::Point selection_end)
{
  std::scoped_lock lck(global_hist_mtx);
  if (!global_hist_valid)
  {
    std::cerr << "no histogram valid yet, cannot add selection" << std::endl;
    return;
  }
  std::cout << "adding rectangle to lut" << std::endl;
  cv::rectangle(global_lut, selection_start, selection_end, Scalar(255), -1);
}

void remove_hist_selection(cv::Point selection_start, cv::Point selection_end)
{
  std::scoped_lock lck(global_hist_mtx);
  if (!global_hist_valid)
  {
    std::cerr << "no histogram valid yet, cannot remove selection" << std::endl;
    return;
  }
  std::cout << "removing rectangle from lut" << std::endl;
  cv::rectangle(global_lut, selection_start, selection_end, Scalar(0), -1);
}

cv::Point hist_cursor_pos;

cv::Point hist_lselection_start;
cv::Point hist_lselection_end;
bool hist_prev_lmouse = false;

cv::Point hist_rselection_start;
cv::Point hist_rselection_end;
bool hist_prev_rmouse = false;
void hist_mouse_callback([[maybe_unused]] int event, int x, int y, [[maybe_unused]]int flags, [[maybe_unused]]void* userdata)
{
  hist_cursor_pos = cv::Point(x, y);
  const bool lmouse = flags & EVENT_FLAG_LBUTTON;
  const bool rmouse = flags & EVENT_FLAG_RBUTTON;
  if (lmouse)
  {
    hist_lselection_end = hist_cursor_pos;
  }
  else
  {
    if (hist_prev_lmouse)
      add_hist_selection(hist_lselection_start, hist_lselection_end);
    hist_lselection_start = hist_cursor_pos;
  }
  hist_prev_lmouse = lmouse;

  if (rmouse)
  {
    hist_rselection_end = hist_cursor_pos;
  }
  else
  {
    if (hist_prev_rmouse)
      remove_hist_selection(hist_rselection_start, hist_rselection_end);
    hist_rselection_start = hist_cursor_pos;
  }
  hist_prev_rmouse = rmouse;
}

//}

std::string to_str_prec(double num, unsigned prec = 3)
{
  std::stringstream strstr;
  strstr << std::fixed << std::setprecision(prec);
  strstr << num;
  return strstr.str();
}

template<int N>
void SetChannel(Mat &img, unsigned char newVal) {   
    for(int x=0;x<img.cols;x++) {
        for(int y=0;y<img.rows;y++) {
            *(img.data + (y * img.cols + x) * img.channels() + N) = newVal;
        }
    }
}

//}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "cpicker");
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
  /* const double object_radius = pl.load_param2<double>("object_radius"); */

  mrs_lib::SubscribeMgr smgr(nh, "cpicker");
  auto sh_img = smgr.create_handler<sensor_msgs::Image>("image_in", ros::Duration(5.0));
  /* auto sh_cinfo = smgr.create_handler<sensor_msgs::CameraInfo>("camera_info", ros::Duration(5.0)); */

  print_options();

  /* int window_flags = WINDOW_AUTOSIZE | WINDOW_KEEPRATIO | WINDOW_GUI_NORMAL; */
  int window_flags = WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED;
  std::string window_name = "cpicker";
  cv::namedWindow(window_name, window_flags);
  cv::setMouseCallback(window_name, color_mouse_callback, NULL);

  std::string hist_window_name = "hist";
  cv::namedWindow(hist_window_name, window_flags);
  cv::setMouseCallback(hist_window_name, hist_mouse_callback, NULL);

  std::string seg_window_name = "seg";
  cv::namedWindow(seg_window_name, window_flags);
  ros::Rate r(100);

  while (ros::ok())
  {
    ros::spinOnce();

    if (sh_img->has_data())
    {
      if (clear_colors)
        clear_hist();
      sensor_msgs::ImageConstPtr img_ros = sh_img->get_data();
      const cv_bridge::CvImagePtr img_ros2 = cv_bridge::toCvCopy(img_ros, "bgr8");
      if (!pause_img)
      {
        std::scoped_lock lck(global_mtx);
        global_image = img_ros2->image;
        global_image_valid = true;
      }

      {
        cv::Mat img;
        global_image.copyTo(img);
        if (prev_lmouse)
          cv::rectangle(img, selection_start, selection_end, Scalar(0,0,255), 2);
        cv::imshow(window_name, img);
        eval_keypress(cv::waitKey(1));
      }

      if (global_hist_valid)
      {
        cv::Mat hist_img;
        cv::Mat lut_img;
        {
          std::scoped_lock lck(global_hist_mtx);
          cv::cvtColor(global_hist, hist_img, cv::COLOR_GRAY2BGR);
          cv::cvtColor(global_lut, lut_img, cv::COLOR_GRAY2BGR);
        }
        SetChannel<1>(lut_img, 0);
        SetChannel<2>(lut_img, 0);

        cv::log(hist_img, hist_img);
        double min, max;
        cv::minMaxLoc(hist_img, &min, &max); 

        hist_img.convertTo(hist_img, CV_8U, 255/(max-min), -255*min/(max-min));
        cv::Mat show_img;
        cv::addWeighted(hist_img, 0.7, lut_img, 0.3, 0.0, show_img);

        if (hist_prev_lmouse)
          cv::rectangle(show_img, hist_lselection_start, hist_lselection_end, Scalar(255,0,0), 1);
        if (hist_prev_rmouse)
          cv::rectangle(show_img, hist_rselection_start, hist_rselection_end, Scalar(0,0,255), 1);
        cv::imshow(hist_window_name, show_img);
        eval_keypress(cv::waitKey(1));

        cv::Mat seg_img = segment_img();
        cv::imshow(seg_window_name, seg_img);
        eval_keypress(cv::waitKey(1));
      }
    }

    r.sleep();
  }
}


