#pragma once

#include <cmath>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/dnn.hpp>

struct Line {
    cv::Point2f a;
    cv::Point2f b;
    cv::Point2f midpoint() const;
    float length() const;
    float angle() const;
    int color{0};
};

struct SegDetection
{
    int class_id{0};
    std::string className{};
    float confidence{0.0};
    cv::Scalar color{};
    cv::Rect box{};
    float mask[32];
};

struct LineDetector {
  LineDetector();
  std::vector<Line> predict(const cv::Mat &frame);

  std::vector<Line> detect_lines_yolo(const cv::Mat &frame);

  std::vector<Line> detect_lines(const cv::Mat &frame);
  std::vector<Line> detect_lines2(const cv::Mat &frame);

  std::vector<Line> detect_lines3(const cv::Mat &frame);
  std::vector<Line> detect_lines4(const cv::Mat &frame);
  std::vector<Line> detect_lines_hugh(const cv::Mat &frame);

  std::vector<Line> filter_lines(const std::vector<Line> &input);

  cv::dnn::Net net;
  cv::Size input_size{640, 640};
};