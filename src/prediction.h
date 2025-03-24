#pragma once

#include "line_detection.h"
#include "yolo_detection.h"
#include <vector>

struct Prediction {
  std::vector<Detection> boxes;
  std::vector<Line> lines;
  cv::Mat frame;

  float heading; // Float with average line heading angle

  float boxes_ms;
  float lines_ms;
};