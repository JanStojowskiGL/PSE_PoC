#include "app.h"
#include "fmt/base.h"
#include "fmt/format.h"
#include "line_detection.h"
#include "prediction.h"
#include "utils.h"
#include "yolo_detection.h"
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stitching.hpp>
#include <vector>

template <typename T>
std::function<T(void)> time_wrapper(std::function<T(void)> f, float &ms) {
  auto fun = [&]() {
    const auto start = std::chrono::steady_clock::now();
    T ret = f();
    ms = since(start);
    return ret;
  };
  return fun;
}

void App::predict(const cv::Mat &frame) {
  #if 1
  const auto boxes_start = std::chrono::steady_clock::now();
  const std::vector<Detection> boxes = detector.predict(frame);
  float boxes_ms = since(boxes_start);

  const auto lines_start = std::chrono::steady_clock::now();
  std::vector<Line> lines = line_detector.predict(frame);
  float lines_ms = since(lines_start);

  #else
  float boxes_ms, lines_ms;
  // Do both at once
  auto boxes_future =
      std::async(std::launch::async,
                 time_wrapper<std::vector<Detection>>(
                     [&]() { return detector.predict(frame); }, boxes_ms));

  auto lines_future =
      std::async(std::launch::async,
                 time_wrapper<std::vector<Line>>(
                     [&]() { return line_detector.predict(frame); }, lines_ms));

  boxes_future.wait();
  lines_future.wait();

  const std::vector<Detection> boxes = boxes_future.get();
  std::vector<Line> lines = lines_future.get();
  #endif

  lines = filter_lines(lines, boxes);

  const float heading = get_heading(lines);

  /* Deal with detections */
  auto [matched_old, matched_new] = match_detections(boxes);
  matched_old.insert(matched_old.end(), matched_new.begin(), matched_new.end());
  previous_detections = matched_old;

  Prediction prediction{matched_old, lines, frame, heading, boxes_ms, lines_ms};
  prediction_queue->push(std::move(prediction));
}

App::App(const std::filesystem::path video_path) : cap{video_path} {
}

void App::do_work() {
  while (worker_should_run) {
    cv::Mat frame;
    cap.read(frame);
    if (frame.empty()) {
      fmt::println("Warning: frame was empty");
      return;
    }
    static int counter{0};
    counter++;
    // if (counter > 1000) {
    predict(frame);
    // }
  }
}
void App::start_worker() {
  worker_should_run = true;
  worker = std::thread([&]() { do_work(); });
}

App::~App() {
  worker_should_run = false;
  worker.join();
}

std::vector<Line> filter_by_angle(const std::vector<Line> &lines,
                                  float max_deviation_deg) {
  double total_distance{0};
  double mean_x{0};
  double mean_angle{0};
  for (const Line &line : lines) {
    mean_x += line.midpoint().x * line.length();
    total_distance += line.length();
    mean_angle += line.angle() * line.length();
  }
  mean_x /= total_distance;
  mean_angle /= total_distance;

  const double max_deviation = max_deviation_deg * M_PI / 180.0;

  std::vector<Line> ret;

  for (const Line &line : lines) {
    float angle = line.angle();
    if (std::abs(angle - mean_angle) < max_deviation) {
      ret.push_back(line);
    }
  }

  return ret;
}

double length(const std::vector<Line> &lines) {
  double ret{0};

  for (const Line &line : lines) {
    ret += line.length();
  }

  return ret;
}

std::vector<Line>
remove_lines_in_detections(const std::vector<Line> &lines,
                           const std::vector<Detection> &detections) {
  if (detections.empty()) {
    return lines;
  }

  auto is_in_box = [&](const Line &line) -> bool {
    for (const Detection &detection : detections) {
      bool is_contained =
          detection.box.contains(line.a) && detection.box.contains(line.b);
      if (is_contained) {
        return true;
      }
    }
    return false;
  };
  std::vector<Line> ret;
  for (const Line &line : lines) {
    if (is_in_box(line)) {
      continue;
    }
    ret.push_back(line);
  }

  return ret;
}

std::vector<Line>
find_lines_on_edges(const std::vector<Line> &lines,
                    const std::vector<Detection> &detections) {
  std::vector<Line> ret = lines;
  const cv::Size image_size{1920 * 2, 1080 * 2};
  const cv::Point2f tl = cv::Point2f{image_size} * 0.05;
  const cv::Point2f br = cv::Point2f{image_size} * 0.95;
  const cv::Rect2f image_center{tl, br};

  for (Line &line : ret) {
    if (!image_center.contains(line.a) || !image_center.contains(line.b)) {
      line.color = 1;
    }
  }

  return ret;
}

// std::vector<Line>
// rotate(const std::vector<Line> &lines) {
//   float total_distance{0};
//   float mean_angle{0};
//   for (const Line &line : lines) {
//     double length = line.length();
//     total_distance += length;
//     mean_angle += line.angle() * length;
//   }
//   mean_angle /= total_distance;
//   // mean_angle /= 2;

//   float cx = std::cos(mean_angle) - std::sin(mean_angle);
//   float cy = std::cos(mean_angle) + std::sin(mean_angle);

//   auto rot_point = [&](cv::Point2f p) {
//     cv::Point2f middle = cv::Point2f{1920, 1080};
//     cv::Point2f diff = p - middle;
//     cv::Point2f p1{diff.x * std::cos(mean_angle) - diff.y *
//     std::sin(mean_angle), diff.x * std::sin(mean_angle) + diff.y *
//     std::cos(mean_angle)}; return p1 + middle;
//   };

//   std::vector<Line> ret{};
//   for (const Line& line : lines) {
//     ret.push_back(line);

//     Line rotated{rot_point(line.a), rot_point(line.b)};
//     rotated.color = 2;
//     ret.push_back(rotated);
//   }

//   return ret;
// }

float distancePointToLine(const cv::Point2f &a, const cv::Point2f &b,
                          const cv::Point2f &c) {
  cv::Point2f ab = b - a;
  cv::Point2f ac = c - a;
  // The magnitude of the cross product gives the area of the parallelogram
  // formed by ab and ac. Dividing by the length of ab gives the height
  // (distance) from c to the line.
  float crossProduct = std::abs(ab.x * ac.y - ab.y * ac.x);
  float abLength = cv::norm(ab);
  return crossProduct / abLength;
}

std::vector<Line> cluster_lines(const std::vector<Line> &lines) {
  std::vector<Line> ret = lines;
  for (Line &line : ret) {
    if (line.a.y > line.b.y) {
      std::swap(line.a, line.b);
    }
  }

  for (size_t i = 0; i < lines.size(); i++) {
    const Line &a = lines.at(i);
    float best_distance = INFINITY;
    for (size_t j = i + 1; j < lines.size(); j++) {
      const Line &b = lines.at(j);

      const float distance_between_points = cv::norm(a.b - b.a);

      const float distance_to_line = distancePointToLine(a.a, a.b, b.a);

      if (distance_to_line < 10.0 && distance_between_points < 200.0) {
        Line c{a.b, b.a};
        c.color = 2;
        ret.push_back(c);
      }
    }
  }

  return ret;
}

std::vector<Line> remove_non_vertical_lines(const std::vector<Line> &lines) {
  std::vector<Line> ret;

  const float angle_thres = 30 * M_PI / 180;

  for (const Line &line : lines) {
    float angle = line.angle();
    if (std::abs(angle - M_PI_2) < angle_thres) {
      ret.push_back(line);
    }
  }

  return ret;
}

// mean x, mean angle
std::pair<double, double> calc_angle(const std::vector<Line> &lines) {
  double total_distance{0};
  double mean_x{0};
  double mean_angle{0};
  for (const Line &line : lines) {
    mean_x += line.midpoint().x * line.length();
    total_distance += line.length();
    mean_angle += line.angle() * line.length();
  }
  mean_x /= total_distance;
  mean_angle /= total_distance;

  return {mean_x, mean_angle};
}

std::vector<Line>
remove_lines_far_from_towers(const std::vector<Line> &lines,
                             const std::vector<Detection> &detections,
                             float angle) {
  if (detections.empty()) {
    return lines;
  }
  std::vector<cv::Point2f> box_centers;
  for (const Detection &detection : detections) {
    cv::Point2f center =
        (cv::Point2f(detection.box.tl()) + cv::Point2f(detection.box.br())) /
        2.0;
    if (center.y < 100) {
      continue;
    }
    box_centers.push_back(center);
  }
  if (box_centers.size() < 2) {
    return lines;
  }
  size_t leftmost_ix{0};
  size_t rightmost_ix{0};
  float leftmost_x = INFINITY;
  float rightmost_x = -INFINITY;
  for (size_t i = 0; i < box_centers.size(); i++) {
    float x = box_centers.at(i).x;
    if (x < leftmost_x) {
      leftmost_x = x;
      leftmost_ix = i;
    }
    if (x > rightmost_x) {
      rightmost_x = x;
      rightmost_ix = i;
    }
  }

  leftmost_x -= detections.at(leftmost_ix).box.width;
  rightmost_x += detections.at(rightmost_ix).box.width;
  cv::Point2f delta{std::cos(angle), std::sin(angle)};

  const float line_spread = rightmost_x - leftmost_x;

  std::vector<Line> ret{};

  for (const Line &line : lines) {
    cv::Point2f leftmost1 =
        box_centers.at(leftmost_ix) -
        cv::Point2f{(float)detections.at(leftmost_ix).box.width, 0};
    cv::Point2f leftmost2 = leftmost1 + delta;

    cv::Point2f rightmost1 =
        box_centers.at(rightmost_ix) -
        cv::Point2f{(float)detections.at(rightmost_ix).box.width, 0};
    cv::Point2f rightmost2 = rightmost1 + delta;

    float a_dist_to_left = distancePointToLine(leftmost1, leftmost2, line.a);
    float a_dist_to_right = distancePointToLine(rightmost1, rightmost2, line.a);

    if (std::max(a_dist_to_left, a_dist_to_right) > line_spread) {
      continue;
    }
    ret.push_back(line);
  }

  return ret;
}

std::vector<Line>
App::filter_lines(const std::vector<Line> &lines,
                  const std::vector<Detection> &detections) const {
  std::vector<Line> filtered;

  float total_distance{0};
  float mean_x{0};
  float mean_angle{0};
  for (const Line &line : lines) {
    double length = line.length();
    length *= length;
    mean_x += line.midpoint().x * length;
    total_distance += length;
    mean_angle += line.angle() * length;
  }
  mean_x /= total_distance;
  mean_angle /= total_distance;

  cv::Point2f middle{mean_x, 1080};
  cv::Point2f other =
      middle + cv::Point2f{std::cos(mean_angle), std::sin(mean_angle)} * 1000;

  Line magic{middle, other};
  magic.color = 4;

  // filtered.push_back(Line{middle, other});
  filtered = remove_lines_in_detections(lines, detections);
  filtered = remove_non_vertical_lines(filtered);
  filtered = filter_by_angle(filtered, 30);
  filtered = filter_by_angle(filtered, 10);
  filtered = find_lines_on_edges(filtered, detections);
  filtered = cluster_lines(filtered);
  float most_angle = calc_angle(filtered).second;
  filtered = remove_lines_far_from_towers(filtered, detections, most_angle);

  {
    auto [mean_x, mean_angle] = calc_angle(filtered);
    cv::Point2f middle{(float)mean_x, 1080};
    cv::Point2f delta =
        cv::Point2d{std::cos(mean_angle), std::sin(mean_angle)} * 1000;
    cv::Point2f p1 = middle + delta;
    cv::Point2f p2 = middle - delta;

    static cv::Point2f point1 = p1;
    static cv::Point2f point2 = p2;

    p1 = p1 * 0.2 + point1 * 0.8;
    p2 = p2 * 0.2 + point2 * 0.8;

    Line magic{p1, p2};
    magic.color = 4;
    // filtered.push_back(magic);
  }

  double output_length = length(filtered);
  double fraction = output_length / total_distance;

  // fmt::println("Line fraction: {}", fraction);
  // if (fraction < 0.8) {
  //   filtered.clear();
  //   for (const auto& side: {left, right}){
  //     auto filtered_side = filter_by_angle(side, 30);
  //     filtered_side = filter_by_angle(filtered_side, 10);
  //     filtered.insert(filtered.end(), filtered_side.begin(),
  //     filtered_side.end());
  //   }
  // }

  return filtered;
}

float App::get_heading(const std::vector<Line> &lines) const {
  float angle = calc_angle(lines).second;
  return angle;
}

std::pair<std::vector<Detection>, std::vector<Detection>>
App::match_detections(const std::vector<Detection> &detections) {
  std::vector<Detection> matched_new;
  std::vector<Detection> matched_old;

  // Track which detections in the current frame have been matched.
  std::vector<bool> usedCurrent(previous_detections.size(), false);

  constexpr float maxDistanceThreshold = 100.0f;

  for (const Detection& det : detections) {
    if (det.center().y < 50){
      continue;
    }
    size_t best_ix{0};
    float best_score = INFINITY;
    for (size_t i = 0; i < previous_detections.size(); i++){
      if(usedCurrent.at(i)){
        continue;
      }
      const Detection& previous_detection = previous_detections.at(i);

      const auto diff = previous_detection.center() - det.center();
      const float distance = cv::norm(diff);

      if (distance < best_score){
        best_score = distance;
        best_ix = i;
      }
    }

    // Must be at least something

    if (best_score < maxDistanceThreshold){
      // We have a match
      Detection copy = det;
      copy.id = previous_detections.at(best_ix).id;
      matched_old.push_back(copy);
      usedCurrent.at(best_ix) = true;
    } else {
      Detection copy = det;
      copy.id = detection_ix++;
      matched_new.push_back((copy));
    }

  }
  return {matched_old, matched_new};
}
