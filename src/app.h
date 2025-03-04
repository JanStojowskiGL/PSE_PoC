#pragma once

#include "line_detection.h"
#include "yolo_detection.h"
#include <opencv2/core/mat.hpp>
#include <vector>

struct App {
  void predict(const cv::Mat &frame);

  YoloDetection detector{};
  LineDetector line_detector{};
};