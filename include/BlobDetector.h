#ifndef BLOBDETECTOR_H
#define BLOBDETECTOR_H

// OpenCV includes
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/ocl.hpp>
#include <CL/cl.h>

#include <fstream>

// local includes
#include <object_detect/DetectionParamsConfig.h>
#include "SegConf.h"

namespace object_detect
{
  
  // shortcut type to the dynamic reconfigure manager template instance
  typedef object_detect::DetectionParamsConfig drcfg_t;

  struct Blob
  {
    lut_elem_t color;
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
    int blob_radius_method;
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
      BlobDetector(const drcfg_t& dr_config, const std::string& ocl_lut_kernel_filename);
      std::vector<Blob> detect(cv::Mat in_img, const lut_t& lut, const std::vector<SegConf>& seg_confs, cv::OutputArray thresholded_img = cv::noArray());
      std::vector<Blob> detect(cv::Mat in_img, const std::vector<SegConf>& seg_confs, cv::OutputArray thresholded_img = cv::noArray());

    private:
      drcfg_t m_drcfg;
      Params m_params;

    private:
      size_t m_thread_count;
      cv::ocl::Queue m_main_queue;
      cv::ocl::Kernel m_ocl_lut_kernel;

    private:
      void preprocess_image(cv::Mat& inout_img) const;
      cv::Mat binarize_image(cv::Mat hsv_img, cv::Mat lab_img, const SegConf& seg_conf) const;
      cv::Mat binarize_image(cv::Mat label_img, const SegConf& seg_conf) const;
      std::vector<Blob> find_blobs(const cv::Mat binary_image, const lut_elem_t color_label) const;
      cv::Mat segment_image(cv::Mat in_img, const lut_t& lut) const;
      bool segment_image_ocl(cv::InputArray in_img, cv::InputArray lut, cv::OutputArray label_img);
      cv::Mat threshold_hsv(cv::Mat hsv_img, const SegConf& seg_conf) const;
      cv::Mat threshold_lab(cv::Mat lab_img, const SegConf& seg_conf) const;

    private:
      cv::ocl::Kernel load_ocl_kernel(const std::string& filename, const std::string& kernel_name, const std::string& options);

  };

} //namespace object_detect

#endif // BLOBDETECTOR_H
