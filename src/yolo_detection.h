#pragma once

#include <cstdint>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/dnn.hpp>

struct Detection {
    cv::Rect box;
    float confidence;
    uint32_t id{0};
    cv::Point2f center() const;
};

struct YoloDetection {
    cv::dnn::Net net;
    YoloDetection();

    std::vector<Detection> predict(const cv::Mat &frame);
};