#include "BlobDetector.h"

using namespace cv;
using namespace std;
using namespace object_detect;

BlobDetector::BlobDetector()
  :
    m_use_ocl(false)
{
}

std::string get_ocl_vendor(const cv::ocl::Device& ocl_device)
{
  return ocl_device.vendorName();
  /* std::string ret = "unknown"; */
  /* if (ocl_device.isIntel()) */
  /*   ret = "Intel"; */
  /* if (ocl_device.isAMD()) */
  /*   ret = "AMD"; */
  /* if (ocl_device.isNVidia()) */
  /*   ret = "NVidia"; */
  /* return ret; */
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

BlobDetector::BlobDetector(const std::string& ocl_kernel_filename)
  :
    m_use_ocl(true)
{
  const std::string ocl_options = "";
  m_ocl_lut_kernel = load_ocl_kernel(ocl_kernel_filename, "ocl_lut_kernel", ocl_options);
  m_ocl_bitwise_and_kernel = load_ocl_kernel(ocl_kernel_filename, "ocl_bitwise_and_kernel", ocl_options);
  m_ocl_seg_kernel = load_ocl_kernel(ocl_kernel_filename, "ocl_seg_kernel", ocl_options);
  /* if (m_ocl_seg_kernel.empty()) */
  /* { */
  /*   ROS_ERROR("[BlobDetector]: Disabling OpenCL (not all required kernels were loaded)."); */
  /*   m_use_ocl = false; */
  /* } */

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

void BlobDetector::set_drcfg(const drcfg_t& drcfg)
{
  m_drcfg = drcfg;
}

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

    blob.color = color_label;
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

/* BlobDetector::detect() method //{ */
std::vector<Blob> BlobDetector::detect(cv::Mat in_img, const lut_t& lut, const std::vector<SegConf>& seg_confs, cv::OutputArray p_labels_img)
{
  m_t_start = ros::WallTime::now();
  std::vector<Blob> blobs;
  preprocess_image(in_img);
  m_t_preprocess = ros::WallTime::now();
  std::vector<uint8_t> labels;
  labels.reserve(seg_confs.size());
  for (const auto& seg_conf : seg_confs)
    labels.push_back(seg_conf.color);
  std::vector<cv::Mat> bin_imgs;

  /* bool ocl_success = segment_image_ocl2(in_img, lut, labels, bin_imgs, p_labels_img); */
  bool ocl_success = false;
  cv::Mat labels_img;
  if (m_use_ocl)
    ocl_success = segment_image_ocl2(in_img, lut, labels, bin_imgs, labels_img);
    /* ocl_success = segment_image_ocl(in_img, lut, labels_img); */
  if (!ocl_success)
    segment_image(in_img, lut, labels_img);

  m_t_segment = ros::WallTime::now();

  for (size_t it = 0; it < seg_confs.size(); it++)
  {
    const auto& seg_conf = seg_confs.at(it);
    const cv::Mat binary_img = binarize_image(labels_img, seg_conf);
    /* const cv::Mat binary_img = bin_imgs.at(it); */
    const std::vector<Blob> tmp_blobs = find_blobs(binary_img, seg_conf.color);
    blobs.insert(std::end(blobs), std::begin(tmp_blobs), std::end(tmp_blobs)); 
  }
  m_t_findblobs = ros::WallTime::now();
  if (p_labels_img.needed())
    labels_img.copyTo(p_labels_img);
  m_t_finish = ros::WallTime::now();
  const std::string ocl_str = ocl_success ? " (with OpenCL)" : "";
  const double preproc_ms = (m_t_preprocess - m_t_start).toSec()*1000.0;
  const double segment_ms = (m_t_segment - m_t_preprocess).toSec()*1000.0;
  const double fblobs_ms = (m_t_findblobs - m_t_segment).toSec()*1000.0;
  const double finish_ms = (m_t_finish - m_t_findblobs).toSec()*1000.0;
  const double total_ms = (m_t_finish - m_t_start).toSec()*1000.0;
  ROS_INFO("[BlobDetector]: profiling%s\n\tpreproc:\t%.2f\n\tsegment:\t%.2f\n\tfblobs:\t%.2f\n\tfinish:%.2f\n\ttotal:%.2f", ocl_str.c_str(), preproc_ms, segment_ms, fblobs_ms, finish_ms, total_ms);
  return blobs;
}
//}

/* BlobDetector::detect() method //{ */
std::vector<Blob> BlobDetector::detect(cv::Mat in_img, const std::vector<SegConf>& seg_confs, cv::OutputArray label_img)
{
  std::vector<Blob> blobs;
  preprocess_image(in_img);
  const cv::Mat labels_img = cv::Mat(in_img.size(), CV_8UC1, cv::Scalar(0));

  cv::Mat hsv_img;
  {
    bool use_hsv = false;
    for (const auto& seg_conf : seg_confs)
      use_hsv = use_hsv || seg_conf.method == bin_method_t::hsv;
    if (use_hsv)
      cv::cvtColor(in_img, hsv_img, cv::COLOR_BGR2HSV);
  }

  cv::Mat lab_img;
  {
    bool use_lab = false;
    for (const auto& seg_conf : seg_confs)
      use_lab = use_lab || seg_conf.method == bin_method_t::lab;
    if (use_lab)
    cv::cvtColor(in_img, lab_img, cv::COLOR_BGR2Lab);
  }

  for (const auto& seg_conf : seg_confs)
  {
    const cv::Mat binary_img = binarize_image(hsv_img, lab_img, seg_conf);
    if (binary_img.empty())
      continue;
    const std::vector<Blob> tmp_blobs = find_blobs(binary_img, seg_conf.color);
    blobs.insert(std::end(blobs), std::begin(tmp_blobs), std::end(tmp_blobs)); 
    if (label_img.needed())
    {
      const cv::Mat tmp_img = binary_img/255*seg_conf.color;
      cv::bitwise_or(labels_img, tmp_img, labels_img);
    }
  }
  if (label_img.needed())
    labels_img.copyTo(label_img);
  return blobs;
}
//}

/* BlobDetector::binarize_image() method //{ */
cv::Mat BlobDetector::binarize_image(cv::Mat label_img, const SegConf& seg_conf)
{
  cv::Scalar color_label(seg_conf.color);
  cv::Mat binary_img;
  bool ocl_success = false;
  if (m_use_ocl)
    ocl_success = bitwise_and_ocl(seg_conf.color, label_img, binary_img);
  if (!ocl_success)
    cv::bitwise_and(label_img, color_label, binary_img);

  /* Preprocess the binary image (fill holes) //{ */
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
  //}

  return binary_img;
}
//}

/* BlobDetector::binarize_image() method //{ */
cv::Mat BlobDetector::binarize_image(cv::Mat hsv_img, cv::Mat lab_img, const SegConf& seg_conf) const
{
  cv::Mat binary_img;
  switch (seg_conf.method)
  {
    case bin_method_t::hsv:
      binary_img = threshold_hsv(hsv_img, seg_conf);
      break;
    case bin_method_t::lab:
      binary_img = threshold_lab(lab_img, seg_conf);
      break;
    case bin_method_t::unknown_method:
      ROS_ERROR("[BlobDetector]: unknown binarization method selected - cannot perform detection!");
      return binary_img;
  }

  /* Preprocess the binary image (fill holes) //{ */
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
  //}

  return binary_img;
}
//}

/* BlobDetector::threshold_hsv() method //{ */
cv::Mat BlobDetector::threshold_hsv(cv::Mat hsv_img, const SegConf& seg_conf) const
{
  double hue_lower = seg_conf.hue_center - seg_conf.hue_range / 2.0;
  double hue_higher = seg_conf.hue_center + seg_conf.hue_range / 2.0;
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

  double sat_lower = seg_conf.sat_center - seg_conf.sat_range / 2.0;
  double sat_higher = seg_conf.sat_center + seg_conf.sat_range / 2.0;

  double val_lower = seg_conf.val_center - seg_conf.val_range / 2.0;
  double val_higher = seg_conf.val_center + seg_conf.val_range / 2.0;

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
cv::Mat BlobDetector::threshold_lab(cv::Mat lab_img, const SegConf& seg_conf) const
{
  double l_lower = seg_conf.l_center - seg_conf.l_range / 2.0;
  double l_higher = seg_conf.l_center + seg_conf.l_range / 2.0;

  double a_lower = seg_conf.a_center - seg_conf.a_range / 2.0;
  double a_higher = seg_conf.a_center + seg_conf.a_range / 2.0;

  double b_lower = seg_conf.b_center - seg_conf.b_range / 2.0;
  double b_higher = seg_conf.b_center + seg_conf.b_range / 2.0;

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
      parallelSegment(const uint8_t* sptr, uint8_t* dptr, const lut_t& lut)
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
      const lut_t& lut;
};

//}

/* BlobDetector::segment_image() method //{ */
bool BlobDetector::segment_image(cv::InputArray p_in_img, cv::InputArray p_lut, cv::OutputArray p_labels_img) const
{
  const cv::Mat in_img = p_in_img.getMat();
  const cv::Mat lut = p_lut.getMat();
  cv::Mat labels_img(in_img.size(), CV_8UC1);
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
    parallel_for_(Range(0, size.width), parallelSegment(sptr_row, dptr, lut));
    /* for (int j = 0; j < size.width; j++) */
    /* { */
    /*   const uint8_t cur_b = sptr[3 * j + 0]; */
    /*   const uint8_t cur_g = sptr[3 * j + 1]; */
    /*   const uint8_t cur_r = sptr[3 * j + 2]; */
    /*   dptr[j] = lookup_lut(lut, cur_r, cur_g, cur_b); */
    /* } */
  }
  labels_img.copyTo(p_labels_img);
  return true;
}
//}

/* BlobDetector::segment_image_ocl() method //{ */
bool BlobDetector::segment_image_ocl(cv::InputArray p_in_img, cv::InputArray p_lut, cv::OutputArray p_labels_img)
{
  cv::UMat in_img = p_in_img.getUMat();
  cv::UMat lut = p_lut.getUMat();
  /* cv::UMat label_img_umat = label_img.getUMat(); */
  cv::UMat label_img(in_img.size(), CV_8UC1);
  // ensure all matrices are continuous
  if (!in_img.isContinuous())
    in_img = in_img.clone();
  if (!lut.isContinuous())
    lut = lut.clone();
  if (!label_img.isContinuous())
    label_img = label_img.clone();

  int ki = 0;
  ki = m_ocl_lut_kernel.set(ki, (int)lut_dim);
  ki = m_ocl_lut_kernel.set(ki, cv::ocl::KernelArg::PtrReadOnly(in_img));
  ki = m_ocl_lut_kernel.set(ki, cv::ocl::KernelArg::PtrReadOnly(lut));
  ki = m_ocl_lut_kernel.set(ki, cv::ocl::KernelArg::PtrWriteOnly(label_img));
  
  constexpr int dims = 1;
  size_t globalsize[dims] = {(size_t)in_img.cols*(size_t)in_img.rows};
  /* size_t localsize[dims] = {m_thread_count, m_thread_count}; */

  bool success = m_ocl_lut_kernel.run(dims, globalsize, nullptr, true, m_main_queue);
  if (!success)
    ROS_ERROR("[BlobDetector]: Failed running kernel!");
  label_img.copyTo(p_labels_img);
  return success;
}
//}

/* BlobDetector::bitwise_and_ocl() method //{ */
bool BlobDetector::bitwise_and_ocl(uint8_t value, cv::InputArray p_in_img, cv::OutputArray p_out_img)
{
  cv::UMat in_img = p_in_img.getUMat();
  cv::UMat out_img(in_img.size(), in_img.type());
  // ensure all matrices are continuous
  if (!in_img.isContinuous())
    in_img = in_img.clone();
  if (!out_img.isContinuous())
    out_img = out_img.clone();

  int ki = 0;
  ki = m_ocl_bitwise_and_kernel.set(ki, value);
  ki = m_ocl_bitwise_and_kernel.set(ki, cv::ocl::KernelArg::PtrReadOnly(in_img));
  ki = m_ocl_bitwise_and_kernel.set(ki, cv::ocl::KernelArg::PtrWriteOnly(out_img));
  
  constexpr int dims = 1;
  size_t globalsize[dims] = {(size_t)in_img.cols*(size_t)in_img.rows};

  bool success = m_ocl_bitwise_and_kernel.run(dims, globalsize, nullptr, true, m_main_queue);
  if (!success)
    ROS_ERROR("[BlobDetector]: Failed running kernel!");
  out_img.copyTo(p_out_img);
  return success;
}
//}

/* BlobDetector::segment_image_ocl2() method //{ */
bool BlobDetector::segment_image_ocl2(cv::InputArray p_in_img, cv::InputArray p_lut, cv::InputArray p_labels, std::vector<cv::Mat>& p_bin_imgs, cv::OutputArray p_labels_img)
{
  cv::UMat labels = p_labels.getUMat();
  const int labels_len = labels.cols;

  cv::UMat in_img = p_in_img.getUMat();
  const int in_img_len = in_img.cols*in_img.rows;

  cv::UMat lut = p_lut.getUMat();

  const cv::Size out_size(in_img_len*labels_len, 1);
  cv::UMat bin_imgs(out_size, CV_8UC1);

  cv::UMat labels_img(in_img.size(), CV_8UC1);
  // ensure all matrices are continuous
  if (!in_img.isContinuous())
    in_img = in_img.clone();
  if (!lut.isContinuous())
    lut = lut.clone();
  if (!labels_img.isContinuous())
    labels_img = labels_img.clone();

  int ki = 0;
  ki = m_ocl_seg_kernel.set(ki, cv::ocl::KernelArg::PtrReadOnly(labels));
  ki = m_ocl_seg_kernel.set(ki, (int)labels_len);
  ki = m_ocl_seg_kernel.set(ki, cv::ocl::KernelArg::PtrReadOnly(in_img));
  ki = m_ocl_seg_kernel.set(ki, (int)in_img_len);
  ki = m_ocl_seg_kernel.set(ki, cv::ocl::KernelArg::PtrReadOnly(lut));
  ki = m_ocl_seg_kernel.set(ki, (int)lut_dim);
  ki = m_ocl_seg_kernel.set(ki, cv::ocl::KernelArg::PtrWriteOnly(bin_imgs));
  ki = m_ocl_seg_kernel.set(ki, cv::ocl::KernelArg::PtrWriteOnly(labels_img));
  
  constexpr int dims = 1;
  size_t globalsize[dims] = {(size_t)in_img.cols*(size_t)in_img.rows};
  /* size_t localsize[dims] = {m_thread_count, m_thread_count}; */

  bool success = m_ocl_seg_kernel.run(dims, globalsize, nullptr, true, m_main_queue);
  if (!success)
    ROS_ERROR("[BlobDetector]: Failed running kernel!");
  labels_img.copyTo(p_labels_img);
  return success;

  /* cv::UMat in_img = p_in_img.getUMat(); */
  /* cv::UMat lut = p_lut.getUMat(); */
  /* cv::UMat labels = p_labels.getUMat(); */
  /* const int in_img_len = in_img.cols*in_img.rows; */
  /* const int labels_len = labels.cols; */
  /* /1* cv::UMat label_img_umat = label_img.getUMat(); *1/ */
  /* const cv::Size out_size(in_img_len*labels_len, 1); */
  /* cv::UMat bin_imgs(out_size, CV_8UC1); */
  /* cv::UMat labels_img(in_img.size(), CV_8UC1); */
  /* // ensure all matrices are continuous */
  /* if (!in_img.isContinuous()) */
  /*   in_img = in_img.clone(); */
  /* if (!lut.isContinuous()) */
  /*   lut = lut.clone(); */

  /* int ki = 0; */
  /* ki = m_ocl_seg_kernel.set(ki, (int)666); */
  /* /1* ki = m_ocl_seg_kernel.set(ki, cv::ocl::KernelArg::PtrReadOnly(labels)); *1/ */
  /* /1* ki = m_ocl_seg_kernel.set(ki, (int)labels_len); *1/ */
  /* /1* ki = m_ocl_seg_kernel.set(ki, cv::ocl::KernelArg::PtrReadOnly(in_img)); *1/ */
  /* /1* ki = m_ocl_seg_kernel.set(ki, (int)in_img_len); *1/ */
  /* /1* ki = m_ocl_seg_kernel.set(ki, cv::ocl::KernelArg::PtrReadOnly(lut)); *1/ */
  /* /1* ki = m_ocl_seg_kernel.set(ki, (int)lut_dim); *1/ */
  /* /1* ki = m_ocl_seg_kernel.set(ki, cv::ocl::KernelArg::PtrWriteOnly(bin_imgs)); *1/ */
  /* /1* ki = m_ocl_seg_kernel.set(ki, cv::ocl::KernelArg::PtrWriteOnly(labels_img)); *1/ */
  
  /* constexpr int dims = 1; */
  /* /1* size_t globalsize[dims] = {(size_t)in_img_len}; *1/ */
  /* size_t globalsize[dims] = {1}; */
  /* /1* size_t localsize[dims] = {m_thread_count, m_thread_count}; *1/ */

  /* bool success = m_ocl_seg_kernel.run(dims, globalsize, nullptr, true, m_main_queue); */
  /* if (!success) */
  /*   ROS_ERROR("[BlobDetector]: Failed running kernel ocl_segment_kernel!"); */

  /* if (p_labels_img.needed()) */
  /*   labels_img.copyTo(p_labels_img); */

  /* p_bin_imgs.reserve(labels_len); */
  /* for (int it = 0; it < labels_len; it++) */
  /* { */
  /*   cv::Mat bin_img; */
  /*   cv::UMat ubin_img; */
  /*   cv::Rect roi(it*in_img_len, 0, in_img_len, 1); */
  /*   ubin_img = bin_imgs(roi).reshape(1, in_img.rows); */
  /*   ubin_img.copyTo(bin_img); */
  /*   p_bin_imgs.push_back(bin_img); */
  /* } */
  /* return success; */
}
//}

