#include "BlobDetector.h"

using namespace cv;
using namespace std;
using namespace object_detect;

BlobDetector::BlobDetector(const drcfg_t& dr_config)
  : m_params(dr_config)
{
  m_drcfg = dr_config;
}

/* BlobDetector::detect_blobs() method //{ */
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
    blob.area = moms.m00;

    if (m_params.filter_by_area)
    {
      double area = moms.m00;
      if (area < m_params.min_area || area >= m_params.max_area)
        continue;
    }

    if (m_params.filter_by_circularity)
    {
      double area = moms.m00;
      double perimeter = arcLength(Mat(contours[contourIdx]), true);
      double ratio = 4 * CV_PI * area / (perimeter * perimeter);
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
      double denominator = std::sqrt(std::pow(2 * moms.mu11, 2) + std::pow(moms.mu20 - moms.mu02, 2));
      const double eps = 1e-2;
      double ratio;
      if (denominator > eps)
      {
        double cosmin = (moms.mu20 - moms.mu02) / denominator;
        double sinmin = 2 * moms.mu11 / denominator;
        double cosmax = -cosmin;
        double sinmax = -sinmin;

        double imin = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmin - moms.mu11 * sinmin;
        double imax = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmax - moms.mu11 * sinmax;
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
      double area = contourArea(Mat(contours[contourIdx]));
      double hullArea = contourArea(Mat(hull));
      double ratio = area / hullArea;
      blob.convexity = ratio;
      if (ratio < m_params.min_convexity || ratio >= m_params.max_convexity)
        continue;
    }

    if (moms.m00 == 0.0)
      continue;
    blob.location = Point2d(moms.m10 / moms.m00, moms.m01 / moms.m00);

    // compute blob radius
    {
      double max_dist = 0.0;
      for (size_t pointIdx = 0; pointIdx < contours[contourIdx].size(); pointIdx++)
      {
        Point2d pt = contours[contourIdx][pointIdx];
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

/* BlobDetector::detect() method //{ */
std::vector<Blob> BlobDetector::detect(cv::Mat in_img, const lut_t& lut, const std::vector<SegConf>& seg_confs, cv::OutputArray label_img)
{
  std::vector<Blob> blobs;

  /* Preprocess the input image //{ */

  // dilate the image if requested
  {
    cv::Mat element = cv::getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
    cv::dilate(in_img, in_img, element, Point(-1, -1), m_drcfg.dilate_iterations);
  }
  // blur it if requested
  if (m_drcfg.gaussianblur_size % 2 == 1)
  {
    cv::GaussianBlur(in_img, in_img, cv::Size(m_drcfg.gaussianblur_size, m_drcfg.gaussianblur_size), 0);
  }
  // blur it if requested
  if (m_drcfg.medianblur_size % 2 == 1)
  {
    cv::medianBlur(in_img, in_img, m_drcfg.medianblur_size);
  }
  //}

  const cv::Mat labels_img = segment_image(in_img, lut);

  for (const auto& seg_conf : seg_confs)
  {
    cv::Mat tmp_img;
    std::vector<Blob> tmp_blobs;
    tmp_blobs = detect_blobs(labels_img, seg_conf);
    blobs.insert(std::end(blobs), std::begin(tmp_blobs), std::end(tmp_blobs)); 
  }
  if (label_img.needed())
    labels_img.copyTo(label_img);
  return blobs;
}
//}

/* BlobDetector::detect_blobs() method //{ */
std::vector<Blob> BlobDetector::detect_blobs(cv::Mat label_img, const SegConf& seg_conf) const
{
  cv::Scalar color_label(seg_conf.color);
  cv::Mat binary_img;
  cv::bitwise_and(label_img, color_label, binary_img);

  /* Preprocess the binary image //{ */
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

  std::vector<Blob> blobs = find_blobs(binary_img, seg_conf.color);
  return blobs;
}
//}

class parallelSegment : public ParallelLoopBody
{
public:
    parallelSegment(const uint8_t* sptr, uint8_t* dptr, const lut_t& lut)
      : sptr(sptr), dptr(dptr), lut(lut)
    {
    }

    void operator()(const Range &range) const CV_OVERRIDE
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

/* BlobDetector::segment_image() method //{ */
cv::Mat BlobDetector::segment_image(cv::Mat in_img, const lut_t& lut)
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
    const uint8_t* sptr = in_img.ptr<uint8_t>(i);
    uint8_t* dptr = binary_img.ptr<uint8_t>(i);
    parallel_for_(Range(0, size.width), parallelSegment(sptr, dptr, lut));
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

