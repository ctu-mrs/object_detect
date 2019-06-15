#ifndef BLOBDETECTOR_H
#define BLOBDETECTOR_H

// OpenCV includes
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>

// local includes
#include <object_detect/DetectionParamsConfig.h>
#include "SegConf.h"

namespace object_detect
{
  
  // shortcut type to the dynamic reconfigure manager template instance
  typedef object_detect::DetectionParamsConfig drcfg_t;

  struct Blob
  {
    double confidence;
    cv::Point2d location;
    double radius;
    double avg_depth;
    double circularity;
    double convexity;
    double angle;
    uint32_t area;
    double inertia;
    std::vector<std::vector<cv::Point> > contours;
  };

  struct Params
  {
    // Filter by area
    bool filter_by_area;
    int min_area;
    int max_area;
    // Filter by circularity
    bool filter_by_circularity;
    double min_circularity;
    double max_circularity;
    // Filter by convexity
    bool filter_by_convexity;
    double min_convexity;
    double max_convexity;
    // Filter by orientation
    bool filter_by_orientation;
    double min_angle;
    double max_angle;
    // Filter by inertia
    bool filter_by_inertia;
    double min_inertia_ratio;
    double max_inertia_ratio;
    // Other filtering criterions
    double min_dist_between;
    size_t min_repeatability;

    Params(drcfg_t cfg);
  };


  class BlobDetector
  {
    public:
      BlobDetector(const drcfg_t& dr_config);
      std::vector<Blob> detect(cv::Mat in_img, const std::vector<SegConf>& seg_confs, cv::OutputArray thresholded_img = cv::noArray());

    private:
      drcfg_t m_drcfg;
      Params m_params;

    private:
      std::vector<Blob> findBlobs(cv::Mat binary_image) const;
      std::vector<Blob> detect_blobs(cv::Mat binary_image) const;
      cv::Mat threshold_hsv(cv::Mat in_img);
      cv::Mat threshold_lab(cv::Mat in_img);

  };

} //namespace object_detect

#endif // BLOBDETECTOR_H
