#include "yolo_detection.h"
#include <filesystem>
#include <fmt/base.h>
#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Based on: https://docs.opencv.org/4.x/da/d9d/tutorial_dnn_yolo.html

YoloDetection::YoloDetection() {
  std::filesystem::path possible_paths[] = {"last.onnx", "../last.onnx"};
  std::filesystem::path model_path{};
  for (const auto &path : possible_paths) {
    if (std::filesystem::exists(path)) {
      model_path = path;
    }
  }
  if (model_path.empty()) {
    std::cerr << "Could not onnx model path\n";
    std::exit(2);
  }
  net = cv::dnn::readNet(model_path);
  // net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
  // net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);

  // const auto supported_backends = cv::dnn::getAvailableBackends();
  // std::cout << "Backend count: " << supported_backends.size() << '\n';
}

std::array<std::string_view, 1> CLASSES = {"slup"};

static void drawBoundingBox(cv::Mat &image, int classId, float confidence,
                            cv::Rect rect) {
  // Draw rectangle
  cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);

  // Create label text "<class>: <confidence>"
  std::string label = std::string{CLASSES[classId]} + ": " + std::to_string(confidence);

  // Display the label at the top of the bounding box
  int baseLine = 0;
  cv::Size labelSize =
      cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
  // top = std::max(top, labelSize.height);
  cv::putText(image, label, rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(0, 255, 0), 1);
}

std::vector<Detection> YoloDetection::predict(const cv::Mat &frame) {
  cv::Mat image = frame.clone();
  int height = image.rows;
  int width = image.cols;

  constexpr int input_size{640}; // Needs to match onnx

  // Prepare a square image for inference
  // FIXME: this way of cropping image is stuid, can be optimized
  int length = std::max(height, width);
  cv::Mat square = cv::Mat::zeros(length, length, CV_8UC3);
  image.copyTo(square(cv::Rect(0, 0, width, height)));

  float scaleFactor = static_cast<float>(length) / input_size;

  cv::Mat blob =
      cv::dnn::blobFromImage(square,                           // input image
                             1.0f / 255.0f,                    // scalefactor
                             cv::Size(input_size, input_size), // size
                             cv::Scalar(),                     // mean
                             true,                             // swapRB
                             false                             // crop
      );

  // Set the input to the network
  net.setInput(blob);

  // Perform forward pass
  cv::Mat outputs = net.forward();

  int batch = outputs.size[0];
  int c = outputs.size[1];
  int n = outputs.size[2];

  // the onnx file output is transposed to what you normally expect, we need
  // to fix it
  cv::Mat mat0(c, n, CV_32F, outputs.ptr<float>(0, 0));
  cv::Mat transposed;
  cv::transpose(mat0, transposed);

  int rows = transposed.rows;
  int cols = transposed.cols;
  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> classIds;

  for (int i = 0; i < rows; ++i) {
    float xCenter = transposed.at<float>(i, 0);
    float yCenter = transposed.at<float>(i, 1);
    float boxWidth = transposed.at<float>(i, 2);
    float boxHeight = transposed.at<float>(i, 3);

    float maxScore = -1.f;
    int maxClassId = -1;
    for (int cIdx = 4; cIdx < cols; ++cIdx) {
      float classScore = transposed.at<float>(i, cIdx);
      if (classScore > maxScore) {
        maxScore = classScore;
        maxClassId = cIdx - 4;
      }
    }

    if (maxScore >= 0.25f) {
      float left = xCenter - 0.5f * boxWidth;
      float top = yCenter - 0.5f * boxHeight;

      boxes.push_back(cv::Rect2f(left, top, boxWidth, boxHeight));
      scores.push_back(maxScore);
      classIds.push_back(maxClassId);
    }
  }

  std::vector<int> nmsIndices;
  float scoreThreshold = 0.25f;
  float nmsThreshold = 0.45f;
  float boxThreshold = 0.5f;
  cv::dnn::NMSBoxes(boxes, scores, scoreThreshold, nmsThreshold, nmsIndices,
                    1.f, boxThreshold);

  std::vector<Detection> ret;

  // cv::Mat annotated = frame.clone();
  for (int idx : nmsIndices) {
    const cv::Rect2f &not_scaled_box = boxes[idx];
    float confidence = scores[idx];
    int classId = classIds[idx];

    cv::Rect2f box = not_scaled_box;
    box.x *= scaleFactor;
    box.y *= scaleFactor;
    box.width *= scaleFactor;
    box.height *= scaleFactor;

    ret.push_back({box, confidence});

    // drawBoundingBox(annotated, classId, confidence, box);
  }
  // cv::imshow("Annotated", annotated);

  return ret;
}
