#pragma once

#include "async_queue.h"
#include "fmt/base.h"
#include "fmt/format.h"
#include "line_detection.h"
#include "prediction.h"
#include "yolo_detection.h"
#include <filesystem>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <sys/types.h>
#include <thread>
#include <vector>

struct App {
  App(const std::filesystem::path video_path);
  ~App();

  void predict(const cv::Mat &frame);
  std::vector<Line> filter_lines(const std::vector<Line> &lines, const std::vector<Detection> &detections) const;
  float get_heading(const std::vector<Line> &lines) const;

  std::pair<std::vector<Detection>, std::vector<Detection>>
  match_detections(const std::vector<Detection> &detections);

  std::vector<Detection> previous_detections{};
  u_int32_t detection_ix{0};

  YoloDetection detector{};
  LineDetector line_detector{};

  AsyncQueue<Prediction> *prediction_queue;

  cv::VideoCapture cap;

  bool worker_should_run = true;
  std::thread worker;
  void do_work();

  void start_worker();
};