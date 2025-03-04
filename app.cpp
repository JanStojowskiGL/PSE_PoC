#include "app.h"
#include "fmt/base.h"
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stitching.hpp>

void App::predict(const cv::Mat &frame) {
  // cv::imshow("Frame", frame);

  const std::vector<Detection> boxes = detector.predict(frame);
  const std::vector<Line> lines = line_detector.predict(frame);

  cv::Mat annotated = frame.clone();
  for (const auto &detection : boxes) {
    cv::rectangle(annotated, detection.box, cv::Scalar{0, 255, 0}, 4);
  }

  for (const auto &line : lines) {
    cv::line(annotated, line.a, line.b, cv::Scalar{255, 0, 255}, 2);
  }
  cv::Mat resized;
  cv::resize(annotated, resized, annotated.size() / 2);
  cv::imshow("Annotated", resized);
}

