#include "object_detect/BlobDetector.h"

using namespace cv;
using namespace std;
using namespace object_detect;

/* helper OCL methods //{ */

std::string get_ocl_vendor(const cv::ocl::Device& ocl_device)
{
  return ocl_device.vendorName();
}

std::string get_ocl_type(const cv::ocl::Device& ocl_device)
{
  std::string ret = "unknown";
  switch (ocl_device.type())
  {
    case cv::ocl::Device::TYPE_DEFAULT:
      ret = "DEFAULT";
      break;
    case cv::ocl::Device::TYPE_CPU:
      ret = "CPU";
      break;
    case cv::ocl::Device::TYPE_GPU:
      ret = "GPU";
      break;
    case cv::ocl::Device::TYPE_ACCELERATOR:
      ret = "ACCELERATOR";
      break;
    case cv::ocl::Device::TYPE_DGPU:
      ret = "DGPU";
      break;
    case cv::ocl::Device::TYPE_IGPU:
      ret = "IGPU";
      break;
    case cv::ocl::Device::TYPE_ALL:
      ret = "ALL";
      break;
  }
  return ret;
}

//}

/* BlobDetector() constructor //{ */
BlobDetector::BlobDetector()
  :
    m_use_ocl(false)
{
}

BlobDetector::BlobDetector(cv::InputArray lut)
  :
    m_use_ocl(false)
{
  lut.copyTo(m_lut);
  lut.copyTo(m_ulut);
}

BlobDetector::BlobDetector(const std::string& ocl_kernel_filename, cv::InputArray lut)
  :
    m_use_ocl(true)
{
  lut.copyTo(m_lut);
  lut.copyTo(m_ulut);

  const std::string ocl_options = "";
  m_ocl_seg_kernel = load_ocl_kernel(ocl_kernel_filename, "segmentation", ocl_options);
  m_ocl_bitwise_and_kernel = load_ocl_kernel(ocl_kernel_filename, "bitwise_and", ocl_options);
  if (m_ocl_seg_kernel.empty())
  {
    ROS_ERROR("[BlobDetector]: Disabling OpenCL (not all required kernels were loaded).");
    m_use_ocl = false;
    return;
  }

  auto ocl_context = cv::ocl::Context::getDefault(false);
  auto ocl_device = cv::ocl::Device::getDefault();
  const std::string ocl_version = ocl_device.OpenCLVersion();
  const std::string name = ocl_device.name();
  const std::string type = get_ocl_type(ocl_device);
  const std::string vendor = get_ocl_vendor(ocl_device);
  const std::string driver = ocl_device.driverVersion();
  ROS_INFO("[BlobDetector]: Initialized OpenCL device:\n\tName:\t%s:\n\tType:\t%s\n\tVendor:\t%s\n\tDriver:\t%s\n\tOpenCL version:\t%s", name.c_str(), type.c_str(), vendor.c_str(), driver.c_str(), ocl_version.c_str());

  m_main_queue.create(ocl_context, ocl_device);
  m_thread_count = ocl_device.maxWorkGroupSize();
}
//}

/* set_drcfg() method //{ */
void BlobDetector::set_drcfg(const drcfg_t& drcfg)
{
  m_drcfg = drcfg;
}
//}

/* BlobDetector::load_ocl_kernel() method //{ */

cv::ocl::Kernel BlobDetector::load_ocl_kernel(const std::string& filename, const std::string& kernel_name, const std::string& options)
{
  ROS_INFO_STREAM("[BlobDetector]: Loading OpenCL kernel file \" " << filename << "\"");
  std::ifstream ifstr(filename, std::ios::binary);
  if (!ifstr.is_open())
  {
    ROS_ERROR("[BlobDetector]: Failed to open OpenCL kernel file!");
    return cv::ocl::Kernel();
  }
  std::string str((std::istreambuf_iterator<char>(ifstr)),
                   std::istreambuf_iterator<char>());
  cv::ocl::Kernel ret(kernel_name.c_str(), cv::ocl::ProgramSource(str.c_str()), options.c_str());
  if (ret.empty())
    ROS_ERROR("[BlobDetector]: Failed to load OpenCL kernel \"%s\".", kernel_name.c_str());
  return ret;
}

//}

/* BlobDetector::find_blobs() method //{ */
std::vector<Blob> BlobDetector::find_blobs(const cv::Mat binary_image, const lut_elem_t color_label) const
{
  std::vector<Blob> blobs_ret;
  std::vector<std::vector<Point>> contours;
  findContours(binary_image, contours, RETR_LIST, CHAIN_APPROX_NONE);
  blobs_ret.reserve(contours.size());

  for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++)
  {
    Blob blob;
    blob.confidence = 1;
    Moments moms = moments(Mat(contours[contourIdx]));
    if (moms.m00 == 0.0)
      continue;
    blob.location = Point2d(moms.m10 / moms.m00, moms.m01 / moms.m00);

    // ignore empty contours (not a blob)
    if (!binary_image.at<uint8_t>(blob.location))
      continue;

    blob.area = moms.m00;

    if (m_drcfg.filter_by_area)
    {
      const double area = moms.m00;
      if (area < m_drcfg.min_area || area >= m_drcfg.max_area)
        continue;
    }

    if (m_drcfg.filter_by_circularity)
    {
      const double area = moms.m00;
      const double perimeter = arcLength(Mat(contours[contourIdx]), true);
      const double ratio = 4 * CV_PI * area / (perimeter * perimeter);
      blob.circularity = ratio;
      if (ratio < m_drcfg.min_circularity || ratio >= m_drcfg.max_circularity)
        continue;
    }

    /* Filter by orientation //{ */
    if (m_drcfg.filter_by_orientation)
    {
      constexpr double eps = 1e-3;
      double angle = 0;
      if (abs(moms.mu20 - moms.mu02) > eps)
        angle = abs(0.5 * atan2((2 * moms.mu11), (moms.mu20 - moms.mu02)));
      blob.angle = angle;
      if (m_drcfg.filter_by_orientation && (angle < m_drcfg.min_angle || angle > m_drcfg.max_angle))
        continue;
    }
    //}

    if (m_drcfg.filter_by_inertia)
    {
      const double denominator = std::sqrt(std::pow(2 * moms.mu11, 2) + std::pow(moms.mu20 - moms.mu02, 2));
      const double eps = 1e-2;
      double ratio;
      if (denominator > eps)
      {
        const double cosmin = (moms.mu20 - moms.mu02) / denominator;
        const double sinmin = 2 * moms.mu11 / denominator;
        const double cosmax = -cosmin;
        const double sinmax = -sinmin;

        const double imin = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmin - moms.mu11 * sinmin;
        const double imax = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmax - moms.mu11 * sinmax;
        ratio = imin / imax;
      } else
      {
        ratio = 1;
      }

      blob.inertia = ratio;
      if (ratio < m_drcfg.min_inertia_ratio || ratio >= m_drcfg.max_inertia_ratio)
        continue;

      blob.confidence = ratio * ratio;
    }

    if (m_drcfg.filter_by_convexity)
    {
      std::vector<Point> hull;
      convexHull(Mat(contours[contourIdx]), hull);
      const double area = contourArea(Mat(contours[contourIdx]));
      const double hullArea = contourArea(Mat(hull));
      const double ratio = area / hullArea;
      blob.convexity = ratio;
      if (ratio < m_drcfg.min_convexity || ratio >= m_drcfg.max_convexity)
        continue;
    }

    // compute blob radius
    switch (m_drcfg.blob_radius_method)
    {
      case 0:
      // the radius is estimated as maximal distance between points on the contour and its center
        {
          double max_dist = 0.0;
          for (size_t pointIdx = 0; pointIdx < contours[contourIdx].size(); pointIdx++)
          {
            const Point2d pt = contours[contourIdx][pointIdx];
            const double cur_dist = norm(blob.location - pt);
            if (cur_dist > max_dist)
              max_dist = cur_dist;
          }
          blob.radius = max_dist;
          break;
        }
      case 1:
      // the radius is estimated as median distance between points on the contour and its center
        {
          std::vector<double> dists;
          for (size_t pointIdx = 0; pointIdx < contours[contourIdx].size(); pointIdx++)
          {
            Point2d pt = contours[contourIdx][pointIdx];
            dists.push_back(norm(blob.location - pt));
          }
          std::sort(dists.begin(), dists.end());
          blob.radius = (dists[(dists.size() - 1) / 2] + dists[dists.size() / 2]) / 2.;
        }
    }

    blob.color_id = color_label;
    blob.contours.push_back(contours[contourIdx]);

    blobs_ret.push_back(blob);
  }

  return blobs_ret;
}
//}

/* BlobDetector::preprocess_image() method //{ */
void BlobDetector::preprocess_image(cv::Mat& inout_img) const
{
  // dilate the image if requested
  {
    cv::Mat element = cv::getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
    cv::dilate(inout_img, inout_img, element, Point(-1, -1), m_drcfg.dilate_iterations);
  }
  // blur it if requested
  if (m_drcfg.gaussianblur_size % 2 == 1)
  {
    cv::GaussianBlur(inout_img, inout_img, cv::Size(m_drcfg.gaussianblur_size, m_drcfg.gaussianblur_size), 0);
  }
  // blur it if requested
  if (m_drcfg.medianblur_size % 2 == 1)
  {
    cv::medianBlur(inout_img, inout_img, m_drcfg.medianblur_size);
  }
}
//}

/* BlobDetector::detect_candidates() method //{ */
std::vector<BallCandidate> BlobDetector::detect_candidates(cv::Mat in_img, cv::OutputArray p_labels_img)
{
#ifdef ENABLE_PROFILING
  ros::WallTime t_start = ros::WallTime::now();
#endif
  preprocess_image(in_img);
#ifdef ENABLE_PROFILING
  ros::WallTime t_preprocess = ros::WallTime::now();
#endif

  cv::Mat seg_img;
  if (m_drcfg.override_settings)
    seg_img = segment_ranges(in_img, p_labels_img);
  else
    seg_img = segment_lut(in_img, p_labels_img);

  const uint8_t label = (uint8_t)m_drcfg.segment_color;
  postprocess_binary_image(seg_img);
  const std::vector<Blob> blobs = find_blobs(seg_img, label);
  const auto balls = blobs_to_balls(blobs);
  if (p_labels_img.needed())
  {
    const cv::Mat tmp_img = seg_img/255*label;
    cv::bitwise_or(p_labels_img, tmp_img, p_labels_img);
  }

#ifdef ENABLE_PROFILING
  ros::WallTime t_finish = ros::WallTime::now();
  const std::string ocl_str = ocl_success ? " (with OpenCL)" : "";
  const double preproc_ms = (t_preprocess - t_start).toSec()*1000.0;
  /* const double segment_ms = (t_segment - t_preprocess).toSec()*1000.0; */
  const double fblobs_ms = (t_findblobs - t_finish).toSec()*1000.0;
  const double total_ms = (m_t_finish - t_start).toSec()*1000.0;
  ROS_INFO("[BlobDetector]: profiling%s\n\tpreproc:\t%.2f\n\tfblobs:\t%.2f\n\tfinish:%.2f\n\ttotal:%.2f", ocl_str.c_str(), preproc_ms, segment_ms, finish_ms, total_ms);
#endif
  return balls;
}
//}

/* BlobDetector::segment_lut() method //{ */
cv::Mat BlobDetector::segment_lut(cv::Mat in_img, cv::OutputArray p_labels_img)
{
  const uint8_t label = (uint8_t)m_drcfg.segment_color;
  std::vector<uint8_t> labels = {label};
  cv::Mat bin_imgs;

  bool ocl_success = false;
  if (m_use_ocl)
    ocl_success = segment_image_lut_ocl(in_img, m_ulut, labels, bin_imgs, p_labels_img);
  if (!ocl_success)
    segment_image_lut_cpu(in_img, m_lut, labels, bin_imgs, p_labels_img);

  cv::Mat binary_img = bin_imgs(in_img).reshape(0, in_img.rows);
  return binary_img;
}
//}

/* BlobDetector::segment_ranges() method //{ */
cv::Mat BlobDetector::segment_ranges(cv::Mat in_img, cv::OutputArray p_labels_img)
{
  p_labels_img.create(in_img.size(), CV_8UC1);
  p_labels_img.setTo(0);

  cv::Mat seg_img;
  if (m_drcfg.binarization_method == bin_method_t::hsv)
  {
    cv::Mat prep_img;
    cv::cvtColor(in_img, prep_img, cv::COLOR_BGR2HSV);
    seg_img = threshold_hsv(prep_img);
  }
  else
  {
    cv::Mat prep_img;
    cv::cvtColor(in_img, prep_img, cv::COLOR_BGR2Lab);
    seg_img = threshold_lab(prep_img);
  }
  return seg_img;
}
//}

/* postprocess_binary_image() method //{ */
void BlobDetector::postprocess_binary_image(cv::Mat binary_img) const
{
  // fill holes in the image
  switch (m_drcfg.fill_holes)
  {
    case 0:  // none
      break;
    case 1:  // using findContours
    {
      vector<vector<cv::Point>> contours;
      cv::findContours(binary_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
      cv::drawContours(binary_img, contours, -1, cv::Scalar(255), CV_FILLED, LINE_8);
      break;
    }
    case 2:  // using convexHull
    {
      vector<vector<cv::Point>> contours;
      cv::findContours(binary_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
      for (const auto& contour : contours)
      {
        vector<vector<cv::Point>> cx_hull(1);
        cv::convexHull(contour, cx_hull.at(0), false, true);
        cv::drawContours(binary_img, cx_hull, 0, cv::Scalar(255), CV_FILLED, LINE_8);
      }
      break;
    }
  }
}
//}

/* BlobDetector::threshold_hsv() method //{ */
cv::Mat BlobDetector::threshold_hsv(cv::Mat hsv_img) const
{
  double hue_lower = m_drcfg.hsv__hue_center - m_drcfg.hsv__hue_range / 2.0;
  double hue_higher = m_drcfg.hsv__hue_center + m_drcfg.hsv__hue_range / 2.0;
  bool overflow;
  /* calculate the correct bounds for the pixel HSV values //{ */
  {
    overflow = false;
    if (hue_lower < 0)
    {
      hue_lower += 180;
      overflow = true;
    }
    if (hue_higher > 179)
    {
      hue_higher -= 180;
      overflow = true;
    }
  }
  //}

  double sat_lower = m_drcfg.hsv__sat_center - m_drcfg.hsv__sat_range / 2.0;
  double sat_higher = m_drcfg.hsv__sat_center + m_drcfg.hsv__sat_range / 2.0;

  double val_lower = m_drcfg.hsv__val_center - m_drcfg.hsv__val_range / 2.0;
  double val_higher = m_drcfg.hsv__val_center + m_drcfg.hsv__val_range / 2.0;

  cv::Mat binary_img;
  // filter the HSV image by color
  {
    Size size = hsv_img.size();
    binary_img.create(size, CV_8UC1);
    // here is the idiom: check the arrays for continuity and,
    // if this is the case,
    // treat the arrays as 1D vectors
    if (hsv_img.isContinuous() && binary_img.isContinuous())
    {
      size.width *= size.height;
      size.height = 1;
    }
    for (int i = 0; i < size.height; i++)
    {
      // when the arrays are continuous,
      // the outer loop is executed only once
      const uint8_t* sptr = hsv_img.ptr<uint8_t>(i);
      uint8_t* dptr = binary_img.ptr<uint8_t>(i);
      for (int j = 0; j < size.width; j++)
      {
        const uint8_t cur_h = sptr[3 * j + 0];
        const uint8_t cur_s = sptr[3 * j + 1];
        const uint8_t cur_v = sptr[3 * j + 2];
        const bool h_ok = (!overflow && cur_h > hue_lower && cur_h < hue_higher) || (overflow && (cur_h > hue_lower || cur_h < hue_higher));
        const bool s_ok = cur_s > sat_lower && cur_s < sat_higher;
        const bool v_ok = cur_v > val_lower && cur_v < val_higher;
        if (h_ok && s_ok && v_ok)
          dptr[j] = 255;
        else
          dptr[j] = 0;
      }
    }
  }
  return binary_img;
}
//}

/* BlobDetector::threshold_lab() method //{ */
cv::Mat BlobDetector::threshold_lab(cv::Mat lab_img) const
{
  double l_lower = m_drcfg.lab__l_center - m_drcfg.lab__l_range / 2.0;
  double l_higher = m_drcfg.lab__l_center + m_drcfg.lab__l_range / 2.0;

  double a_lower = m_drcfg.lab__a_center - m_drcfg.lab__a_range / 2.0;
  double a_higher = m_drcfg.lab__a_center + m_drcfg.lab__a_range / 2.0;

  double b_lower = m_drcfg.lab__b_center - m_drcfg.lab__b_range / 2.0;
  double b_higher = m_drcfg.lab__b_center + m_drcfg.lab__b_range / 2.0;

  cv::Mat binary_img;
  // filter the Lab image by color
  {
    Size size = lab_img.size();
    binary_img.create(size, CV_8UC1);
    // here is the idiom: check the arrays for continuity and,
    // if this is the case,
    // treat the arrays as 1D vectors
    if (lab_img.isContinuous() && binary_img.isContinuous())
    {
      size.width *= size.height;
      size.height = 1;
    }
    for (int i = 0; i < size.height; i++)
    {
      // when the arrays are continuous,
      // the outer loop is executed only once
      const uint8_t* sptr = lab_img.ptr<uint8_t>(i);
      uint8_t* dptr = binary_img.ptr<uint8_t>(i);
      for (int j = 0; j < size.width; j++)
      {
        const uint8_t cur_l = sptr[3 * j + 0];
        const uint8_t cur_a = sptr[3 * j + 1];
        const uint8_t cur_b = sptr[3 * j + 2];
        const bool l_ok = cur_l > l_lower && cur_l < l_higher;
        const bool a_ok = cur_a > a_lower && cur_a < a_higher;
        const bool b_ok = cur_b > b_lower && cur_b < b_higher;
        if (l_ok && a_ok && b_ok)
          dptr[j] = 255;
        else
          dptr[j] = 0;
      }
    }
  }
  return binary_img;
}
//}

/* class parallelSegment //{ */

class parallelSegment : public ParallelLoopBody
{
  public:
      parallelSegment(const uint8_t* sptr, uint8_t* dptr, cv::InputArray lut)
        : sptr(sptr), dptr(dptr), lut(lut)
      {
      }

      void operator()(const Range &range) const
      {
        for (int j = range.start; j < range.end; j++)
        {
          const uint8_t cur_b = sptr[3 * j + 0];
          const uint8_t cur_g = sptr[3 * j + 1];
          const uint8_t cur_r = sptr[3 * j + 2];
          dptr[j] = lookup_lut(lut, cur_r, cur_g, cur_b);
        }
      }

  private:
      const uint8_t* sptr;
      uint8_t* dptr;
      cv::InputArray lut;
};

//}

/* BlobDetector::segment_image_lut_cpu() method //{ */
bool BlobDetector::segment_image_lut_cpu(cv::InputArray p_in_img, cv::InputArray p_lut, cv::InputArray p_labels, cv::OutputArray p_bin_imgs, cv::OutputArray p_labels_img)
{
  const cv::Mat in_img = p_in_img.getMat();
  const int in_img_len = in_img.cols*in_img.rows;
  p_labels_img.create(in_img.size(), CV_8UC1);
  cv::Mat labels_img = p_labels_img.getMat();
  Size size = in_img.size();
  // here is the idiom: check the arrays for continuity and,
  // if this is the case,
  // treat the arrays as 1D vectors
  if (in_img.isContinuous() && labels_img.isContinuous())
  {
    size.width *= size.height;
    size.height = 1;
  }
  for (int i = 0; i < size.height; i++)
  {
    // when the arrays are continuous,
    // the outer loop is executed only once
    const uint8_t* const sptr_row = in_img.ptr<uint8_t>(i);
    uint8_t* dptr = labels_img.ptr<uint8_t>(i);
    parallel_for_(Range(0, size.width), parallelSegment(sptr_row, dptr, p_lut));
    /* for (int j = 0; j < size.width; j++) */
    /* { */
    /*   const uint8_t cur_b = sptr_row[3 * j + 0]; */
    /*   const uint8_t cur_g = sptr_row[3 * j + 1]; */
    /*   const uint8_t cur_r = sptr_row[3 * j + 2]; */
    /*   dptr[j] = lookup_lut(p_lut, cur_r, cur_g, cur_b); */
    /* } */
  }

  cv::Mat labels = p_labels.getMat();
  const int labels_len = labels.cols;

  const cv::Size out_size(in_img_len*labels_len, 1);
  p_bin_imgs.create(out_size, CV_8UC1);
  cv::Mat bin_imgs = p_bin_imgs.getMat();

  for (int it = 0; it < labels_len; it++)
  {
    uint8_t color_id = labels.at<uint8_t>(it);
    const cv::Rect roi(it*in_img_len, 0, in_img_len, 1);
    cv::Mat binary_img = bin_imgs(roi).reshape(0, in_img.rows);
    bool ocl_success = false;
    if (m_use_ocl)
      ocl_success = bitwise_and_ocl(color_id, p_labels_img, binary_img);
    if (!ocl_success)
      cv::bitwise_and(p_labels_img, cv::Scalar(color_id), binary_img);
  }

  return true;
}
//}

/* BlobDetector::bitwise_and_ocl() method //{ */
bool BlobDetector::bitwise_and_ocl(uint8_t value, cv::InputArray p_in_img, cv::OutputArray p_out_img)
{
  cv::UMat in_img = p_in_img.getUMat();
  if (p_out_img.empty())
    p_out_img.create(in_img.size(), in_img.type());
  cv::UMat out_img = p_out_img.getUMat();
  // ensure all matrices are continuous
  if (!in_img.isContinuous())
  {
    ROS_ERROR("[BlobDetector::bitwise_and_ocl]: Input image must be continuous! Skipping.");
    return false;
  }

  int ki = 0;
  ki = m_ocl_bitwise_and_kernel.set(ki, value);
  ki = m_ocl_bitwise_and_kernel.set(ki, cv::ocl::KernelArg::PtrReadOnly(in_img));
  if (out_img.empty()) // avoid pointer problems by providing a dummy 1x1 matrix (which is not used)
    ki = m_ocl_seg_kernel.set(ki, cv::ocl::KernelArg::PtrWriteOnly(cv::UMat(cv::Size(1, 1), CV_8UC1)));
  else
    ki = m_ocl_seg_kernel.set(ki, cv::ocl::KernelArg::PtrWriteOnly(out_img));
  
  constexpr int dims = 1;
  size_t globalsize[dims] = {(size_t)in_img.cols*(size_t)in_img.rows};

  bool success = m_ocl_bitwise_and_kernel.run(dims, globalsize, nullptr, true, m_main_queue);
  if (!success)
    ROS_ERROR("[BlobDetector]: Failed running kernel \"bitwise_and\"!");
  return success;
}
//}

/* blobs_to_balls() method //{ */
std::vector<BallCandidate> BlobDetector::blobs_to_balls(const std::vector<Blob>& blobs)
{
  std::vector<BallCandidate> ret;
  ret.reserve(blobs.size());
  for (const auto& blob : blobs)
  {
    BallCandidate ball;
    ball.type = m_drcfg.segment_color;
    ball.location = blob.location;
    ball.radius = blob.radius;
    ret.push_back(ball);
  }
  return ret;
}
//}

/* BlobDetector::segment_image_lut_ocl() method //{ */
bool BlobDetector::segment_image_lut_ocl(cv::InputArray p_in_img, cv::InputArray p_lut, cv::InputArray p_labels, cv::OutputArray p_bin_imgs, cv::OutputArray p_labels_img)
{
  cv::UMat labels = p_labels.getUMat();
  const int labels_len = labels.cols;

  cv::UMat in_img = p_in_img.getUMat();
  const int in_img_len = in_img.cols*in_img.rows;

  cv::UMat lut = p_lut.getUMat();

  p_labels_img.create(in_img.size(), CV_8UC1);
  cv::UMat labels_img = p_labels_img.getUMat();

  const cv::Size out_size(in_img_len*labels_len, 1);
  p_bin_imgs.create(out_size, CV_8UC1);
  cv::UMat bin_imgs = p_bin_imgs.getUMat();

  // ensure all matrices are continuous
  if (!in_img.isContinuous())
  {
    ROS_ERROR("[BlobDetector::segment_image_ocl]: Input image must be continuous! Skipping.");
    return false;
  }
  if (!lut.isContinuous())
  {
    ROS_ERROR("[BlobDetector::segment_image_ocl]: LUT must be continuous! Skipping.");
    return false;
  }

  int ki = 0;
  if (labels.empty()) // avoid pointer problems by providing a dummy 1x1 matrix (which is not used)
    ki = m_ocl_seg_kernel.set(ki, cv::ocl::KernelArg::PtrReadOnly(cv::UMat(cv::Size(1, 1), CV_8UC1)));
  else
    ki = m_ocl_seg_kernel.set(ki, cv::ocl::KernelArg::PtrReadOnly(labels));
  ki = m_ocl_seg_kernel.set(ki, (int)labels_len);
  ki = m_ocl_seg_kernel.set(ki, cv::ocl::KernelArg::PtrReadOnly(in_img));
  ki = m_ocl_seg_kernel.set(ki, (int)in_img_len);
  ki = m_ocl_seg_kernel.set(ki, cv::ocl::KernelArg::PtrReadOnly(lut));
  ki = m_ocl_seg_kernel.set(ki, (int)lut_dim);
  ki = m_ocl_seg_kernel.set(ki, cv::ocl::KernelArg::PtrWriteOnly(labels_img));
  if (bin_imgs.empty()) // avoid pointer problems by providing a dummy 1x1 matrix (which is not used)
    ki = m_ocl_seg_kernel.set(ki, cv::ocl::KernelArg::PtrWriteOnly(cv::UMat(cv::Size(1, 1), CV_8UC1)));
  else
    ki = m_ocl_seg_kernel.set(ki, cv::ocl::KernelArg::PtrWriteOnly(bin_imgs));
  
  constexpr int dims = 1;
  size_t globalsize[dims] = {(size_t)in_img_len};
  /* size_t localsize[dims] = {m_thread_count, m_thread_count}; */

#ifdef ENABLE_PROFILING
  ros::WallTime start = ros::WallTime::now();
#endif
  bool success = m_ocl_seg_kernel.run(dims, globalsize, nullptr, true, m_main_queue);
#ifdef ENABLE_PROFILING
  ros::WallTime end = ros::WallTime::now();
  ROS_INFO("[BlobDetector]: OpenCL runtime: %.2f", (end-start).toSec()*1000.0);
#endif
  if (!success)
    ROS_ERROR("[BlobDetector]: Failed running kernel \"segmentation\"!");

  return success;
}
//}

