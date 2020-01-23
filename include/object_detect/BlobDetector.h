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
#include "object_detect/BallCandidate.h"
#include "object_detect/lut.h"

namespace object_detect
{
  
  // shortcut type to the dynamic reconfigure manager template instance
  typedef object_detect::DetectionParamsConfig drcfg_t;

  struct Blob
  {
    lut_elem_t color_id;
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

  class BlobDetector
  {
    public:
      BlobDetector();
      BlobDetector(cv::InputArray lut);
      BlobDetector(const std::string& ocl_kernel_filename, cv::InputArray lut);
      void set_drcfg(const drcfg_t& drcfg);
      void set_lut(cv::InputArray lut);
      std::vector<BallCandidate> detect_candidates(cv::Mat in_img, cv::OutputArray p_labels_img = cv::noArray());

    private:
      drcfg_t m_drcfg;
      cv::Mat m_lut;
      cv::UMat m_ulut;

    private:
      bool m_use_ocl;
      size_t m_thread_count;
      cv::ocl::Queue m_main_queue;
      cv::ocl::Kernel m_ocl_seg_kernel;
      cv::ocl::Kernel m_ocl_bitwise_and_kernel;

    private:
      cv::Mat segment_lut(cv::Mat in_img, cv::OutputArray p_labels_img = cv::noArray());
      cv::Mat segment_ranges(cv::Mat in_img, cv::OutputArray p_labels_img = cv::noArray());

    private:
      void preprocess_image(cv::Mat& inout_img) const;
      void postprocess_binary_image(cv::Mat binary_img) const;
      std::vector<Blob> find_blobs(const cv::Mat binary_image, const lut_elem_t color_label) const;
      bool segment_image_lut_cpu(cv::InputArray p_in_img, cv::InputArray p_lut, cv::InputArray p_labels, cv::OutputArray p_bin_imgs, cv::OutputArray p_labels_img);
      bool segment_image_lut_ocl(cv::InputArray p_in_img, cv::InputArray p_lut, cv::InputArray p_labels, cv::OutputArray p_bin_imgs, cv::OutputArray p_labels_img);
      cv::Mat threshold_hsv(cv::Mat hsv_img) const;
      cv::Mat threshold_lab(cv::Mat lab_img) const;
      bool bitwise_and_ocl(uint8_t value, cv::InputArray in_img, cv::OutputArray out_img);

      std::vector<BallCandidate> blobs_to_balls(const std::vector<Blob>& blobs);

    private:
      cv::ocl::Kernel load_ocl_kernel(const std::string& filename, const std::string& kernel_name, const std::string& options);

  };

} //namespace object_detect

#endif // BLOBDETECTOR_H
