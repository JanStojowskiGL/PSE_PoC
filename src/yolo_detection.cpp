#include "yolo_detection.h"
#include <filesystem>
#include <fmt/base.h>
#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

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
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
  // const auto supported_backends = cv::dnn::getAvailableBackends();
  // std::cout << "Backend count: " << supported_backends.size() << '\n';
}

std::array<std::string_view, 1> CLASSES = {"slup"};

static void drawBoundingBox(cv::Mat &image, int classId, float confidence,
                            cv::Rect rect) {
  // Draw rectangle
  cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);

  // Create label text "<class>: <confidence>"
  std::string label =
      std::string{CLASSES[classId]} + ": " + std::to_string(confidence);

  // Display the label at the top of the bounding box
  int baseLine = 0;
  cv::Size labelSize =
      cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
  // top = std::max(top, labelSize.height);
  cv::putText(image, label, rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(0, 255, 0), 1);
}

cv::Mat create_blob(const cv::Mat &frame) {
  float paddingValue = 0.0;
  bool swapRB = false;
  int inpWidth = 640;
  int inpHeight = 640;
  cv::Scalar scale = 1.0f / 255.0f;
  cv::Scalar mean = 0.0;
  cv::dnn::ImagePaddingMode paddingMode =
      cv::dnn::ImagePaddingMode::DNN_PMODE_LETTERBOX;

  cv::Size size(inpWidth, inpHeight);
  cv::dnn::Image2BlobParams imgParams(scale, size, mean, swapRB, CV_32F,
                                      cv::dnn::DNN_LAYOUT_NCHW, paddingMode,
                                      paddingValue);

  // rescale boxes back to original image
  cv::dnn::Image2BlobParams paramNet;
  paramNet.scalefactor = scale;
  paramNet.size = size;
  paramNet.mean = mean;
  paramNet.swapRB = swapRB;
  paramNet.paddingmode = paddingMode;

  cv::Mat inp = cv::dnn::blobFromImageWithParams(frame, imgParams);
  std::cout << "Inp shape: " << inp.size << '\n';
  return inp;
}

std::vector<Detection> YoloDetection::predict(const cv::Mat &frame) {
  int height = frame.rows;
  int width = frame.cols;

  constexpr int input_size{640}; // Needs to match onnx

  // Prepare a square image for inference
  int length = std::max(height, width);

  cv::Mat square = cv::Mat::zeros(input_size, input_size, CV_8UC3);
  int small_height = height * input_size / width;
  auto small_roi = square(cv::Rect(0, 0, input_size, small_height));
  cv::resize(frame, small_roi, small_roi.size());

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

cv::Point2f Detection::center() const { return (box.tl() + box.br()) / 2; }
