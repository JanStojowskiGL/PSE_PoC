#include "line_detection.h"
#include <fmt/ranges.h>
#include "fmt/base.h"
#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

void draw_lines(cv::Mat &frame, const std::vector<Line> &lines){
    for (const auto line: lines){
        cv::line(frame, line.a, line.b, cv::Scalar{255, 0, 255}, 2);
    }
}

std::vector<Line> LineDetector::predict(const cv::Mat &frame) {
    const auto lines = detect_lines(frame);
    cv::Mat annotated = frame.clone();
    // draw_lines(annotated, lines);

    // const auto lines_filtered = filter_lines(lines);
    // draw_lines(annotated, lines_filtered);

    // cv::Mat resized;
    // cv::resize(annotated, resized, annotated.size() / 2);
    // cv::imshow("Lines", resized);

    return lines;
}

std::vector<Line> LineDetector::filter_lines(const std::vector<Line> &input) {
    std::vector<float> angles;
    // std::vector<float> x_intercept;
    for(const Line& line: input){
        cv::Point2f diff = line.a - line.b;
        float angle = std::atan2(diff.y, diff.x);
        if (angle < 0.0f){
            angle += M_PI;
        }
        angles.push_back(angle);
    }
    fmt::println("Angles: {}", angles);

    return {};
}

std::vector<Line> LineDetector::detect_lines(const cv::Mat &frame) {
    cv::Mat downscaled = frame.clone();
    constexpr int downscaled_size = 500;
    while(downscaled.size().width > downscaled_size *2){
        cv::pyrDown(downscaled, downscaled);
    }

    const float scale = static_cast<float>(frame.rows) / downscaled.rows;

    cv::Mat hsv;
    cv::cvtColor(downscaled, hsv, cv::COLOR_BGR2HSV);

    cv::Mat saturation;
    cv::extractChannel(hsv, saturation, 1);

    const int kernel_size{5};
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {kernel_size, kernel_size});

    cv::Mat blackhat;
    cv::morphologyEx(saturation, blackhat, cv::MORPH_BLACKHAT, kernel);

    // cv::imshow("Blackhat", blackhat);

    const cv::Ptr<cv::LineSegmentDetector> lsd =
        cv::createLineSegmentDetector(cv::LSD_REFINE_ADV);

    std::vector<cv::Vec4f> detected_lines;
    lsd->detect(blackhat, detected_lines);


    // cv::Mat annotated = downscaled.clone();
    // lsd->drawSegments(annotated, detected_lines);

    // cv::imshow("Annotated", annotated);
    std::vector<Line> ret;
    for(const auto e : detected_lines) {
        cv::Point2f a = cv::Point2f{e[0], e[1]} * scale;
        cv::Point2f b = cv::Point2f{e[2], e[3]} * scale;
        ret.emplace_back(Line{a,b});
    }

    return ret;
}

std::vector<Line> LineDetector::detect_lines_hugh(const cv::Mat &frame) {
    // Does not work, do not use
    cv::Mat downscaled = frame.clone();
    constexpr int downscaled_size = 500;
    while(downscaled.size().width > downscaled_size *2){
        cv::pyrDown(downscaled, downscaled);
    }

    const float scale = static_cast<float>(frame.rows) / downscaled.rows;

    cv::Mat hsv;
    cv::cvtColor(downscaled, hsv, cv::COLOR_BGR2HSV);

    cv::Mat saturation;
    cv::extractChannel(hsv, saturation, 1);

    const int kernel_size{5};
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {kernel_size, kernel_size});

    cv::Mat blackhat;
    cv::morphologyEx(saturation, blackhat, cv::MORPH_BLACKHAT, kernel);
    cv::imshow("Blackhat", blackhat);

    const auto clahe_ptr = cv::createCLAHE();
    cv::Mat clahe;
    clahe_ptr->apply(blackhat, clahe);
    cv::imshow("clahe", clahe);


    std::vector<cv::Vec4f> detected_lines;
    cv::HoughLinesP(clahe, detected_lines, 1, CV_PI/180, 19, 0, 10);

    std::vector<Line> ret;
    for(const auto e : detected_lines) {
        cv::Point2f a = cv::Point2f{e[0], e[1]} * scale;
        cv::Point2f b = cv::Point2f{e[2], e[3]} * scale;
        ret.emplace_back(Line{a,b});
    }

    return ret;

}
