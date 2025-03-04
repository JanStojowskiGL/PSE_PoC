#pragma once

#include <opencv2/core/mat.hpp>

struct Line {
    cv::Point2f a;
    cv::Point2f b;
};

struct LineDetector {
  std::vector<Line> predict(const cv::Mat &frame);

  std::vector<Line> detect_lines(const cv::Mat &frame);
  std::vector<Line> detect_lines_hugh(const cv::Mat &frame);

  std::vector<Line> filter_lines(const std::vector<Line> &input);
};