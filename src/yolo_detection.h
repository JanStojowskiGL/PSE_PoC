#pragma once

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/dnn.hpp>

struct Detection {
    cv::Rect box;
    float confidence;
};

struct YoloDetection {
    cv::dnn::Net net;
    YoloDetection();

    std::vector<Detection> predict(const cv::Mat &frame);
};