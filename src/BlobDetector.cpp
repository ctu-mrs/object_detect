#include "BlobDetector.h"

using namespace cv;
using namespace std;
using namespace object_detect;

BlobDetector::BlobDetector(const drcfg_t& dr_config)
  : m_params(dr_config)
{
  m_drcfg = dr_config;
}

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

    if (m_params.filter_by_area)
    {
      const double area = moms.m00;
      if (area < m_params.min_area || area >= m_params.max_area)
        continue;
    }

    if (m_params.filter_by_circularity)
    {
      const double area = moms.m00;
      const double perimeter = arcLength(Mat(contours[contourIdx]), true);
      const double ratio = 4 * CV_PI * area / (perimeter * perimeter);
      blob.circularity = ratio;
      if (ratio < m_params.min_circularity || ratio >= m_params.max_circularity)
        continue;
    }

    /* Filter by orientation //{ */
    if (m_params.filter_by_orientation)
    {
      constexpr double eps = 1e-3;
      double angle = 0;
      if (abs(moms.mu20 - moms.mu02) > eps)
        angle = abs(0.5 * atan2((2 * moms.mu11), (moms.mu20 - moms.mu02)));
      blob.angle = angle;
      if (m_params.filter_by_orientation && (angle < m_params.min_angle || angle > m_params.max_angle))
        continue;
    }
    //}

    if (m_params.filter_by_inertia)
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
      if (ratio < m_params.min_inertia_ratio || ratio >= m_params.max_inertia_ratio)
        continue;

      blob.confidence = ratio * ratio;
    }

    if (m_params.filter_by_convexity)
    {
      std::vector<Point> hull;
      convexHull(Mat(contours[contourIdx]), hull);
      const double area = contourArea(Mat(contours[contourIdx]));
      const double hullArea = contourArea(Mat(hull));
      const double ratio = area / hullArea;
      blob.convexity = ratio;
      if (ratio < m_params.min_convexity || ratio >= m_params.max_convexity)
        continue;
    }

    // compute blob radius
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
std::vector<Blob> BlobDetector::detect(cv::Mat in_img, const lut_t& lut, const std::vector<SegConf>& seg_confs, cv::OutputArray label_img)
{
  std::vector<Blob> blobs;
  preprocess_image(in_img);
  const cv::Mat labels_img = segment_image(in_img, lut);

  for (const auto& seg_conf : seg_confs)
  {
    const cv::Mat binary_img = binarize_image(labels_img, seg_conf);
    const std::vector<Blob> tmp_blobs = find_blobs(binary_img, seg_conf.color);
    blobs.insert(std::end(blobs), std::begin(tmp_blobs), std::end(tmp_blobs)); 
  }
  if (label_img.needed())
    labels_img.copyTo(label_img);
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
cv::Mat BlobDetector::binarize_image(cv::Mat label_img, const SegConf& seg_conf) const
{
  cv::Scalar color_label(seg_conf.color);
  cv::Mat binary_img;
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
      cv::findContours(binary_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);  // this line sometimes throws error :(
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
      cv::findContours(binary_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);  // this line sometimes throws error :(
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

/* BlobDetector::segment_image() method //{ */
cv::Mat BlobDetector::segment_image(cv::Mat in_img, const lut_t& lut) const
{
  cv::Mat binary_img;
  Size size = in_img.size();
  binary_img.create(size, CV_8UC1);
  // here is the idiom: check the arrays for continuity and,
  // if this is the case,
  // treat the arrays as 1D vectors
  if (in_img.isContinuous() && binary_img.isContinuous())
  {
    size.width *= size.height;
    size.height = 1;
  }
  for (int i = 0; i < size.height; i++)
  {
    // when the arrays are continuous,
    // the outer loop is executed only once
    const uint8_t* const sptr_row = in_img.ptr<uint8_t>(i);
    uint8_t* dptr = binary_img.ptr<uint8_t>(i);
    parallel_for_(Range(0, size.width), [&](const Range &range)
      {
        const uint8_t * sptr = sptr_row + 3*range.start;
        for (int j = range.start; j < range.end; j++)
        {
          const uint8_t cur_b = sptr[0];
          const uint8_t cur_g = sptr[1];
          const uint8_t cur_r = sptr[2];
          dptr[j] = lookup_lut(lut, cur_r, cur_g, cur_b);
          sptr += 3;
        }
      }
    );
    /* for (int j = 0; j < size.width; j++) */
    /* { */
    /*   const uint8_t cur_b = sptr[3 * j + 0]; */
    /*   const uint8_t cur_g = sptr[3 * j + 1]; */
    /*   const uint8_t cur_r = sptr[3 * j + 2]; */
    /*   dptr[j] = lookup_lut(lut, cur_r, cur_g, cur_b); */
    /* } */
  }
  return binary_img;
}
//}

/* Params constructor //{ */
Params::Params(drcfg_t cfg)
{
  // Filter by area
  filter_by_area = cfg.filter_by_area;
  min_area = cfg.min_area;
  max_area = cfg.max_area;
  // Filter by circularity
  filter_by_circularity = cfg.filter_by_circularity;
  min_circularity = cfg.min_circularity;
  max_circularity = cfg.max_circularity;
  // Filter by orientation
  filter_by_orientation = cfg.filter_by_orientation;
  min_angle = cfg.min_angle;
  max_angle = cfg.max_angle;
  // Filter by convexity
  filter_by_convexity = cfg.filter_by_convexity;
  min_convexity = cfg.min_convexity;
  max_convexity = cfg.max_convexity;
  // Filter by inertia
  filter_by_inertia = cfg.filter_by_inertia;
  min_inertia_ratio = cfg.min_inertia_ratio;
  max_inertia_ratio = cfg.max_inertia_ratio;
  // Other filtering criterions
  min_dist_between = cfg.min_dist_between;
  min_repeatability = cfg.min_repeatability;
}
//}

