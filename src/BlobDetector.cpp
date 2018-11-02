#include "BlobDetector.h"

using namespace cv;
using namespace std;
using namespace object_detect;

BlobDetector::BlobDetector(const drcfg_t& dr_config)
  : m_params(dr_config)
{
  m_drcfg = dr_config;
}

/* BlobDetector::findBlobs() method //{ */
std::vector<Blob> BlobDetector::findBlobs(cv::Mat binary_image) const
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
      std::vector<double> dists;
      for (size_t pointIdx = 0; pointIdx < contours[contourIdx].size(); pointIdx++)
      {
        Point2d pt = contours[contourIdx][pointIdx];
        dists.push_back(norm(blob.location - pt));
      }
      std::sort(dists.begin(), dists.end());
      blob.radius = (dists[(dists.size() - 1) / 2] + dists[dists.size() / 2]) / 2.;
    }

    blob.contours.push_back(contours[contourIdx]);

    blobs_ret.push_back(blob);
  }

  return blobs_ret;
}
//}

/* BlobDetector::detect_blobs() method //{ */
std::vector<Blob> BlobDetector::detect_blobs(cv::Mat binary_image) const
{
  // these blobs will be grouped, filtered, and only some of them will be returned
  std::vector<std::vector<Blob> > blobs;
  std::vector<Blob> cur_blobs = findBlobs(binary_image);
  std::vector<std::vector<Blob> > new_blobs;
  new_blobs.reserve(cur_blobs.size());

  for (size_t i = 0; i < cur_blobs.size(); i++)
  {
    bool isNew = true;
    for (size_t j = 0; j < blobs.size(); j++)
    {
      double dist = norm(blobs[j][blobs[j].size() / 2].location - cur_blobs[i].location);
      isNew = dist >= m_params.min_dist_between && dist >= blobs[j][blobs[j].size() / 2].radius && dist >= cur_blobs[i].radius;
      if (!isNew)
      {
        blobs[j].push_back(cur_blobs[i]);

        size_t k = blobs[j].size() - 1;
        while (k > 0 && blobs[j][k].radius < blobs[j][k - 1].radius)
        {
          blobs[j][k] = blobs[j][k - 1];
          k--;
        }
        blobs[j][k] = cur_blobs[i];

        break;
      }
    }
    if (isNew)
      new_blobs.push_back(std::vector<Blob>(1, cur_blobs[i]));
  }
  std::copy(new_blobs.begin(), new_blobs.end(), std::back_inserter(blobs));

  std::vector<Blob> blobs_ret;
  blobs_ret.reserve(blobs.size());
  for (const vector<Blob>& cur_blobs : blobs)
  {
    if (cur_blobs.size() < (size_t)m_params.min_repeatability)
      continue;
    Point2d sumPoint(0, 0);
    double normalizer = 0;
    vector<vector<Point> > contours;
    contours.reserve(cur_blobs.size());
    for (size_t j = 0; j < cur_blobs.size(); j++)
    {
      sumPoint += cur_blobs[j].confidence * cur_blobs[j].location;
      normalizer += cur_blobs[j].confidence;
      contours.push_back(cur_blobs[j].contours[0]);
    }
    sumPoint *= (1. / normalizer);
    Blob result_blob;
    result_blob.confidence = normalizer / cur_blobs.size();
    result_blob.location = sumPoint;
    result_blob.radius = cur_blobs[cur_blobs.size() / 2].radius;
    result_blob.avg_depth = cur_blobs[cur_blobs.size() / 2].avg_depth;
    result_blob.convexity = cur_blobs[cur_blobs.size() / 2].convexity;
    result_blob.angle = cur_blobs[cur_blobs.size() / 2].angle;
    result_blob.area = cur_blobs[cur_blobs.size() / 2].area;
    result_blob.circularity = cur_blobs[cur_blobs.size() / 2].circularity;
    result_blob.inertia = cur_blobs[cur_blobs.size() / 2].inertia;
    result_blob.contours = contours;
    blobs_ret.push_back(result_blob);
  }
  return blobs_ret;
}
//}

/* BlobDetector::detect() method //{ */
std::vector<Blob> BlobDetector::detect(cv::Mat in_img, cv::OutputArray thresholded_img)
{

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

  // calculate the correct bounds for the pixel values
  double color_lower = m_drcfg.color_center - m_drcfg.color_range / 2.0;
  double color_higher = m_drcfg.color_center + m_drcfg.color_range / 2.0;
  bool overflow = false;
  if (color_lower < 0)
  {
    color_lower += 180;
    overflow = true;
  }
  if (color_higher > 179)
  {
    color_higher -= 180;
    overflow = true;
  }

  cv::Mat binary_img;
  // filter the HSV image by color
  {
    cv::Mat hsv_img;
    // convert the input image to HSV for better filtering
    cv::cvtColor(in_img, hsv_img, cv::COLOR_BGR2HSV);
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
      const uchar* sptr = hsv_img.ptr<uint8_t>(i);
      uchar* dptr = binary_img.ptr<uint8_t>(i);
      for (int j = 0; j < size.width; j++)
      {
        uint8_t cur_h = sptr[3 * j + 0];
        uint8_t cur_s = sptr[3 * j + 1];
        uint8_t cur_v = sptr[3 * j + 2];
        bool h_ok = (!overflow && cur_h > color_lower && cur_h < color_higher) || (overflow && (cur_h > color_lower || cur_h < color_higher));
        if (h_ok && cur_s > m_drcfg.min_saturation && cur_v > m_drcfg.min_brightness)
          dptr[j] = 255;
        else
          dptr[j] = 0;
      }
    }
  }

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
  
  std::vector<Blob> blobs = detect_blobs(binary_img);

  if (thresholded_img.needed())
  {
    binary_img.copyTo(thresholded_img);
  }

  return blobs;
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

