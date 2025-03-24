#include "line_detection.h"

#include "fmt/base.h"
#include <algorithm>
#include <cmath>
#include <fmt/ranges.h>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <vector>

void draw_lines(cv::Mat &frame, const std::vector<Line> &lines) {
  for (const auto line : lines) {
    cv::line(frame, line.a, line.b, cv::Scalar{255, 0, 255}, 2);
  }
}

std::vector<Line> LineDetector::predict(const cv::Mat &frame) {
  const auto lines = detect_lines(frame);
  //   cv::Mat annotated = frame.clone();
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
  for (const Line &line : input) {
    cv::Point2f diff = line.a - line.b;
    float angle = std::atan2(diff.y, diff.x);
    if (angle < 0.0f) {
      angle += M_PI;
    }
    angles.push_back(angle);
  }
  fmt::println("Angles: {}", angles);

  return {};
}

std::vector<Line> LineDetector::detect_lines(const cv::Mat &frame) {
  cv::Mat downscaled;
  constexpr int downscaled_size = 500;

  if (frame.size().width > downscaled_size * 2) {
    cv::pyrDown(frame, downscaled);
  } else {
    downscaled = frame;
  }
  while (downscaled.empty() || downscaled.size().width > downscaled_size * 2) {
    cv::pyrDown(downscaled, downscaled);
  }

  const float scale = static_cast<float>(frame.rows) / downscaled.rows;

  cv::Mat hsv;
  cv::cvtColor(downscaled, hsv, cv::COLOR_BGR2HSV);

  cv::Mat saturation;
  cv::extractChannel(hsv, saturation, 1);

  // cv::imshow("Saturation", saturation);
  cv::Mat saturation_filtered;
  cv::medianBlur(saturation, saturation_filtered, 3);

  const int kernel_size{5};
  const cv::Mat kernel =
      cv::getStructuringElement(cv::MORPH_RECT, {kernel_size, kernel_size});

  cv::Mat blackhat;
  cv::morphologyEx(saturation_filtered, blackhat, cv::MORPH_BLACKHAT, kernel);

  cv::circle(blackhat, cv::Point{blackhat.cols, 0}, blackhat.cols / 8,
             cv::Scalar{0}, -1);

  // cv::imshow("Blackhat", blackhat);
  // cv::waitKey(1);

  const cv::Ptr<cv::LineSegmentDetector> lsd =
      // cv::createLineSegmentDetector(cv::LSD_REFINE_ADV);
      cv::createLineSegmentDetector(cv::LSD_REFINE_NONE);

  // cv::UMat blackhat_u;
  // blackhat.copyTo(blackhat_u);

  std::vector<cv::Vec4f> detected_lines;
  // lsd->detect(blackhat, detected_lines);
  lsd->detect(blackhat, detected_lines);

  // cv::Mat annotated = downscaled.clone();
  // lsd->drawSegments(annotated, detected_lines);

  // cv::imshow("Annotated", annotated);
  std::vector<Line> ret;
  for (const auto e : detected_lines) {
    cv::Point2f a = cv::Point2f{e[0], e[1]} * scale;
    cv::Point2f b = cv::Point2f{e[2], e[3]} * scale;
    ret.emplace_back(Line{a, b});
  }

  return ret;
}

std::vector<Line> LineDetector::detect_lines_hugh(const cv::Mat &frame) {
  // Does not work, do not use
  cv::Mat downscaled = frame.clone();
  constexpr int downscaled_size = 500;
  while (downscaled.size().width > downscaled_size * 2) {
    cv::pyrDown(downscaled, downscaled);
  }

  const float scale = static_cast<float>(frame.rows) / downscaled.rows;

  cv::Mat hsv;
  cv::cvtColor(downscaled, hsv, cv::COLOR_BGR2HSV);

  cv::Mat saturation;
  cv::extractChannel(hsv, saturation, 1);

  const int kernel_size{5};
  const cv::Mat kernel =
      cv::getStructuringElement(cv::MORPH_RECT, {kernel_size, kernel_size});

  cv::Mat blackhat;
  cv::morphologyEx(saturation, blackhat, cv::MORPH_BLACKHAT, kernel);
  cv::imshow("Blackhat", blackhat);

  const auto clahe_ptr = cv::createCLAHE();
  cv::Mat clahe;
  clahe_ptr->apply(blackhat, clahe);
  cv::imshow("clahe", clahe);

  std::vector<cv::Vec4f> detected_lines;
  cv::HoughLinesP(clahe, detected_lines, 1, CV_PI / 180, 19, 0, 10);

  std::vector<Line> ret;
  for (const auto e : detected_lines) {
    cv::Point2f a = cv::Point2f{e[0], e[1]} * scale;
    cv::Point2f b = cv::Point2f{e[2], e[3]} * scale;
    ret.emplace_back(Line{a, b});
  }

  return ret;
}

std::vector<Line> LineDetector::detect_lines2(const cv::Mat &frame) {
  constexpr int downscaled_width = 800;
  const float scale =
      static_cast<float>(downscaled_width) / static_cast<float>(frame.cols);

  cv::Mat downscaled;
  cv::resize(frame, downscaled,
             cv::Size{downscaled_width, static_cast<int>(frame.rows * scale)},
             0, 0, cv::INTER_AREA);

  cv::imshow("Downscaled", downscaled);

  //   cv::Mat filtered;
  //   cv::pyrMeanShiftFiltering(downscaled, filtered, 11, 11);

  //   cv::imshow("Filtered", filtered);

  //   return {};

  cv::Mat smoothed;
  cv::bilateralFilter(downscaled, smoothed, 5, 10, 2);

  //   cv::imshow("Smoothed", smoothed);

  cv::Mat hsv;
  cv::cvtColor(downscaled, hsv, cv::COLOR_BGR2HSV);

  cv::Mat saturation;
  cv::extractChannel(hsv, saturation, 1);

  cv::imshow("Saturation", saturation);

  cv::Mat laplacian;
  cv::Laplacian(saturation, laplacian, CV_64F, 11);
  cv::Mat laplace_scaled;
  cv::convertScaleAbs(laplacian, laplace_scaled, 1 / 1e5);

  cv::Mat binarized;
  cv::threshold(laplace_scaled, binarized, 10, 255, cv::THRESH_BINARY);

  cv::imshow("Laplace scaled", laplace_scaled);
  cv::imshow("binarized", binarized);

  laplacian /= 7000000.0;
  laplacian += 0.5;
  cv::imshow("Laplacian", laplacian);

  cv::Mat sharr;
  cv::Scharr(saturation, sharr, CV_32F, 0, 1);
  sharr /= 4000.0;
  sharr = cv::abs(sharr);
  cv::imshow("sharr", sharr);

  cv::waitKey(1);
  return {};

  cv::Mat background;
  cv::GaussianBlur(saturation, background, cv::Size{51, 51}, 0);
  cv::imshow("Background", background);

  cv::Mat diff = background - saturation;

  const int kernel_size{3};
  const cv::Mat kernel =
      cv::getStructuringElement(cv::MORPH_ELLIPSE, {kernel_size, kernel_size});
  cv::dilate(diff, diff, kernel);
  cv::equalizeHist(diff, diff);
  cv::imshow("Diff", diff);

  cv::Mat thresholded;
  //   cv::adaptiveThreshold(saturation, thresholded, 255,
  //   cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 51, 0);
  cv::threshold(saturation, thresholded, 50, 255, cv::THRESH_BINARY);
  cv::imshow("Thresholded", thresholded);

  std::vector<Line> ret;

  return ret;
}
cv::Point2f Line::midpoint() const { return (a + b) / 2.0F; }
float Line::length() const {
  cv::Point2f diff = a - b;
  return cv::norm(diff);
}
float Line::angle() const {
  cv::Point2f diff = a - b;
  float angle = std::atan2(diff.y, diff.x);
  if (angle < 0) {
    angle += M_PI;
  }
  return angle;
}

void calcGST(const cv::Mat &inputImg, cv::Mat &imgCoherencyOut,
             cv::Mat &imgOrientationOut, int w) {
  cv::Mat img;
  inputImg.convertTo(img, CV_64F);
  // GST components calculation (start)
  // J =  (J11 J12; J12 J22) - GST
  cv::Mat imgDiffX, imgDiffY, imgDiffXY;
  Sobel(img, imgDiffX, CV_64F, 1, 0, 3);
  Sobel(img, imgDiffY, CV_64F, 0, 1, 3);
  multiply(imgDiffX, imgDiffY, imgDiffXY);
  cv::Mat imgDiffXX, imgDiffYY;
  multiply(imgDiffX, imgDiffX, imgDiffXX);
  multiply(imgDiffY, imgDiffY, imgDiffYY);
  cv::Mat J11, J22, J12; // J11, J22 and J12 are GST components
  boxFilter(imgDiffXX, J11, CV_64F, cv::Size(w, w));
  boxFilter(imgDiffYY, J22, CV_64F, cv::Size(w, w));
  boxFilter(imgDiffXY, J12, CV_64F, cv::Size(w, w));
  // GST components calculation (stop)
  // eigenvalue calculation (start)
  // lambda1 = 0.5*(J11 + J22 + sqrt((J11-J22)^2 + 4*J12^2))
  // lambda2 = 0.5*(J11 + J22 - sqrt((J11-J22)^2 + 4*J12^2))
  cv::Mat tmp1, tmp2, tmp3, tmp4;
  tmp1 = J11 + J22;
  tmp2 = J11 - J22;
  multiply(tmp2, tmp2, tmp2);
  multiply(J12, J12, tmp3);
  sqrt(tmp2 + 4.0 * tmp3, tmp4);
  cv::Mat lambda1, lambda2;
  lambda1 = tmp1 + tmp4;
  lambda1 = 0.5 * lambda1; // biggest eigenvalue
  lambda2 = tmp1 - tmp4;
  lambda2 = 0.5 * lambda2; // smallest eigenvalue
  // eigenvalue calculation (stop)
  // Coherency calculation (start)
  // Coherency = (lambda1 - lambda2)/(lambda1 + lambda2)) - measure of
  // anisotropism Coherency is anisotropy degree (consistency of local
  // orientation)
  divide(lambda1 - lambda2, lambda1 + lambda2, imgCoherencyOut);
  // Coherency calculation (stop)
  // orientation angle calculation (start)
  // tan(2*Alpha) = 2*J12/(J22 - J11)
  // Alpha = 0.5 atan2(2*J12/(J22 - J11))
  phase(J22 - J11, 2.0 * J12, imgOrientationOut, true);
  imgOrientationOut = 0.5 * imgOrientationOut;
  // orientation angle calculation (stop)
}

void calcGST_CUDA(const cv::Mat &inputImg, cv::Mat &imgCoherencyOut,
                  cv::Mat &imgOrientationOut, int w) {
  // Upload input image and convert to CV_32F (GPU-friendly precision)
  cv::cuda::GpuMat d_input(inputImg);
  cv::cuda::GpuMat d_img;
  d_input.convertTo(d_img, CV_32F);

  // Compute Sobel derivatives on GPU
  cv::Ptr<cv::cuda::Filter> sobelX =
      cv::cuda::createSobelFilter(d_img.type(), d_img.type(), 1, 0, 3);
  cv::Ptr<cv::cuda::Filter> sobelY =
      cv::cuda::createSobelFilter(d_img.type(), d_img.type(), 0, 1, 3);
  cv::cuda::GpuMat d_imgDiffX, d_imgDiffY;
  sobelX->apply(d_img, d_imgDiffX);
  sobelY->apply(d_img, d_imgDiffY);

  // Compute cross derivative and squared derivatives
  cv::cuda::GpuMat d_imgDiffXY;
  cv::cuda::multiply(d_imgDiffX, d_imgDiffY, d_imgDiffXY);

  cv::cuda::GpuMat d_imgDiffXX, d_imgDiffYY;
  cv::cuda::multiply(d_imgDiffX, d_imgDiffX, d_imgDiffXX);
  cv::cuda::multiply(d_imgDiffY, d_imgDiffY, d_imgDiffYY);

  // Apply box filter to compute GST components
  cv::Ptr<cv::cuda::Filter> boxFilter =
      cv::cuda::createBoxFilter(CV_32F, CV_32F, cv::Size(w, w));
  cv::cuda::GpuMat J11, J22, J12;
  boxFilter->apply(d_imgDiffXX, J11);
  boxFilter->apply(d_imgDiffYY, J22);
  boxFilter->apply(d_imgDiffXY, J12);

  // Compute eigenvalues of the structure tensor
  cv::cuda::GpuMat tmp1, tmp2, tmp3, tmp4;
  cv::cuda::add(J11, J22, tmp1);        // tmp1 = J11 + J22
  cv::cuda::subtract(J11, J22, tmp2);   // tmp2 = J11 - J22
  cv::cuda::multiply(tmp2, tmp2, tmp2); // tmp2 = (J11 - J22)^2

  cv::cuda::multiply(J12, J12, tmp3); // tmp3 = J12^2
  cv::cuda::GpuMat tmp3_4;
  cv::cuda::multiply(tmp3, cv::Scalar(4.0), tmp3_4); // 4*J12^2

  cv::cuda::GpuMat sum_tmp;
  cv::cuda::add(tmp2, tmp3_4, sum_tmp);
  cv::cuda::sqrt(sum_tmp, tmp4); // tmp4 = sqrt((J11 - J22)^2 + 4*J12^2)

  cv::cuda::GpuMat lambda1, lambda2;
  cv::cuda::GpuMat tmp_plus, tmp_minus;
  cv::cuda::add(tmp1, tmp4, tmp_plus);       // J11 + J22 + sqrt(...)
  cv::cuda::subtract(tmp1, tmp4, tmp_minus); // J11 + J22 - sqrt(...)
  cv::cuda::multiply(tmp_plus, cv::Scalar(0.5),
                     lambda1); // lambda1 (largest eigenvalue)
  cv::cuda::multiply(tmp_minus, cv::Scalar(0.5),
                     lambda2); // lambda2 (smallest eigenvalue)

  // Compute coherency: (lambda1 - lambda2)/(lambda1 + lambda2)
  cv::cuda::GpuMat diff, sum_eig, coherency;
  cv::cuda::subtract(lambda1, lambda2, diff);
  cv::cuda::add(lambda1, lambda2, sum_eig);
  cv::cuda::divide(diff, sum_eig, coherency);

  // Compute local orientation using cartToPolar.
  // Note: We compute atan2(2*J12, J22 - J11) and then half it.
  cv::cuda::GpuMat diffOrient, twoJ12, orientation;
  cv::cuda::subtract(J22, J11, diffOrient);         // diffOrient = J22 - J11
  cv::cuda::multiply(J12, cv::Scalar(2.0), twoJ12); // twoJ12 = 2*J12
  cv::cuda::GpuMat dummy; // dummy magnitude output (unused)
  cv::cuda::cartToPolar(diffOrient, twoJ12, dummy, orientation, true);
  cv::cuda::multiply(orientation, cv::Scalar(0.5), orientation);

  // Download results from GPU to host memory
  coherency.download(imgCoherencyOut);
  orientation.download(imgOrientationOut);
}

void cudaLineSegmentDetector(const cv::Mat &mask,
                             std::vector<cv::Vec4i> &segments) {
  // Upload the mask to GPU memory.
  cv::cuda::GpuMat d_mask(mask);

  // Optionally, if your mask is not already binary, you might want to threshold
  // it here.

  // Create a CUDA Hough segment detector.
  // Parameters:
  //   rho           - Distance resolution in pixels.
  //   theta         - Angle resolution in radians.
  //   threshold     - Minimum number of votes to consider a line.
  //   minLineLength - Minimum length of a line.
  //   maxLineGap    - Maximum allowed gap between line segments.
  cv::Ptr<cv::cuda::HoughSegmentDetector> houghSegmentDetector =
      cv::cuda::createHoughSegmentDetector(1.0, CV_PI / 180, 50, 30, 100);

  // Detect line segments on the GPU.
  // The detected lines are stored in a GpuMat, where each row is [x1, y1, x2,
  // y2].
  cv::cuda::GpuMat d_lines;
  houghSegmentDetector->detect(d_mask, d_lines);

  // Download the detected lines from GPU to CPU.
  cv::Mat lines;
  d_lines.download(lines);

  // Convert the resulting lines to a vector of cv::Vec4i.
  segments.clear();
  for (int i = 0; i < lines.rows; i++) {
    // Each row is expected to be a 4-element vector [x1, y1, x2, y2].
    cv::Vec4i line = lines.at<cv::Vec4i>(i, 0);
    segments.push_back(line);
  }
}

std::vector<Line> LineDetector::detect_lines3(const cv::Mat &frame) {
  constexpr int downscaled_width = 800;
  const float scale =
      static_cast<float>(downscaled_width) / static_cast<float>(frame.cols);

  cv::Mat downscaled;
  cv::resize(frame, downscaled,
             cv::Size{downscaled_width, static_cast<int>(frame.rows * scale)},
             0, 0, cv::INTER_AREA);

  cv::imshow("Downscaled", downscaled);

  cv::Mat gray;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

  cv::Mat imgCoherencyOut, imgOrientationOut;
  calcGST_CUDA(gray, imgCoherencyOut, imgOrientationOut, 20);

  cv::Mat imgCoherency, imgOrientation;
  cv::normalize(imgCoherencyOut, imgCoherency, 0, 1, cv::NORM_MINMAX, CV_32F);
  cv::normalize(imgOrientationOut, imgOrientation, 0, 1, cv::NORM_MINMAX,
                CV_32F);

  // cv::imshow("Coherency", imgCoherency);
  // cv::imshow("Orientation", imgOrientation);

  double C_Thr = 0.60;   // threshold for coherency
  int LowThr = 90 - 20;  // threshold1 for orientation, it ranges from 0 to 180
  int HighThr = 90 + 20; // threshold2 for orientation, it ranges from 0 to 180

  cv::Mat imgCoherencyBin = imgCoherencyOut > C_Thr;

  // cv::imshow("imgCoherencyBin", imgCoherencyBin);

  cv::Mat imgOrientationBin;
  cv::inRange(imgOrientationOut, cv::Scalar(LowThr), cv::Scalar(HighThr),
              imgOrientationBin);

  // cv::imshow("imgOrientationBin", imgOrientationBin);
  cv::Mat imgBin = imgCoherencyBin & imgOrientationBin;

  cv::medianBlur(imgBin, imgBin, 11);

  // cv::imshow("imgBin", imgBin);

  cv::Mat mask;

  cv::convertScaleAbs(imgBin, mask, 255.0);

  cv::imshow("mask", mask);

  cv::cuda::GpuMat cuda_mask{mask};

  cv::Canny(mask, mask, 50, 50);
  cv::imshow("Mask with_sobel", mask);

  std::vector<cv::Vec4i> lines;
  cudaLineSegmentDetector(mask, lines);

  cv::Mat color_dst = frame.clone();
  for (size_t i = 0; i < lines.size(); i++) {
    cv::line(color_dst, cv::Point(lines[i][0], lines[i][1]),
             cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 255), 3, 8);
  }

  cv::imshow("Color dst", color_dst);

  cv::waitKey(1);

  return {};
}

void calcGST_CUDA2(const cv::cuda::GpuMat &inputImg,
                   cv::cuda::GpuMat &imgCoherencyOut,
                   cv::cuda::GpuMat &imgOrientationOut, int w) {
  // Upload input image and convert to CV_32F (GPU-friendly precision)
  cv::cuda::GpuMat d_input(inputImg);
  cv::cuda::GpuMat d_img;
  d_input.convertTo(d_img, CV_32F);

  // Compute Sobel derivatives on GPU
  cv::Ptr<cv::cuda::Filter> sobelX =
      cv::cuda::createSobelFilter(d_img.type(), d_img.type(), 1, 0, 3);
  cv::Ptr<cv::cuda::Filter> sobelY =
      cv::cuda::createSobelFilter(d_img.type(), d_img.type(), 0, 1, 3);
  cv::cuda::GpuMat d_imgDiffX, d_imgDiffY;
  sobelX->apply(d_img, d_imgDiffX);
  sobelY->apply(d_img, d_imgDiffY);

  // Compute cross derivative and squared derivatives
  cv::cuda::GpuMat d_imgDiffXY;
  cv::cuda::multiply(d_imgDiffX, d_imgDiffY, d_imgDiffXY);

  cv::cuda::GpuMat d_imgDiffXX, d_imgDiffYY;
  cv::cuda::multiply(d_imgDiffX, d_imgDiffX, d_imgDiffXX);
  cv::cuda::multiply(d_imgDiffY, d_imgDiffY, d_imgDiffYY);

  // Apply box filter to compute GST components
  cv::Ptr<cv::cuda::Filter> boxFilter =
      cv::cuda::createBoxFilter(CV_32F, CV_32F, cv::Size(w, w));
  cv::cuda::GpuMat J11, J22, J12;
  boxFilter->apply(d_imgDiffXX, J11);
  boxFilter->apply(d_imgDiffYY, J22);
  boxFilter->apply(d_imgDiffXY, J12);

  // Compute eigenvalues of the structure tensor
  cv::cuda::GpuMat tmp1, tmp2, tmp3, tmp4;
  cv::cuda::add(J11, J22, tmp1);        // tmp1 = J11 + J22
  cv::cuda::subtract(J11, J22, tmp2);   // tmp2 = J11 - J22
  cv::cuda::multiply(tmp2, tmp2, tmp2); // tmp2 = (J11 - J22)^2

  cv::cuda::multiply(J12, J12, tmp3); // tmp3 = J12^2
  cv::cuda::GpuMat tmp3_4;
  cv::cuda::multiply(tmp3, cv::Scalar(4.0), tmp3_4); // 4*J12^2

  cv::cuda::GpuMat sum_tmp;
  cv::cuda::add(tmp2, tmp3_4, sum_tmp);
  cv::cuda::sqrt(sum_tmp, tmp4); // tmp4 = sqrt((J11 - J22)^2 + 4*J12^2)

  cv::cuda::GpuMat lambda1, lambda2;
  cv::cuda::GpuMat tmp_plus, tmp_minus;
  cv::cuda::add(tmp1, tmp4, tmp_plus);       // J11 + J22 + sqrt(...)
  cv::cuda::subtract(tmp1, tmp4, tmp_minus); // J11 + J22 - sqrt(...)
  cv::cuda::multiply(tmp_plus, cv::Scalar(0.5),
                     lambda1); // lambda1 (largest eigenvalue)
  cv::cuda::multiply(tmp_minus, cv::Scalar(0.5),
                     lambda2); // lambda2 (smallest eigenvalue)

  // Compute coherency: (lambda1 - lambda2)/(lambda1 + lambda2)
  cv::cuda::GpuMat diff, sum_eig, coherency;
  cv::cuda::subtract(lambda1, lambda2, diff);
  cv::cuda::add(lambda1, lambda2, sum_eig);
  cv::cuda::divide(diff, sum_eig, coherency);

  // Compute local orientation using cartToPolar.
  // Note: We compute atan2(2*J12, J22 - J11) and then half it.
  cv::cuda::GpuMat diffOrient, twoJ12, orientation;
  cv::cuda::subtract(J22, J11, diffOrient);         // diffOrient = J22 - J11
  cv::cuda::multiply(J12, cv::Scalar(2.0), twoJ12); // twoJ12 = 2*J12
  cv::cuda::GpuMat dummy; // dummy magnitude output (unused)
  cv::cuda::cartToPolar(diffOrient, twoJ12, dummy, orientation, true);
  cv::cuda::multiply(orientation, cv::Scalar(0.5), orientation);

  // Download results from GPU to host memory
  // coherency.download(imgCoherencyOut);
  // orientation.download(imgOrientationOut);
  imgCoherencyOut = coherency;
  imgOrientationOut = orientation;
}

std::vector<Line> LineDetector::detect_lines4(const cv::Mat &frame) {
  cv::cuda::GpuMat frame_color{frame};
  cv::cuda::GpuMat gray;
  cv::cuda::cvtColor(frame_color, gray, cv::COLOR_BGR2GRAY);

  cv::cuda::GpuMat imgCoherencyOut, imgOrientationOut;
  calcGST_CUDA2(gray, imgCoherencyOut, imgOrientationOut, 20);

  // cv::Mat imgCoherency, imgOrientation;
  // cv::normalize(imgCoherencyOut, imgCoherency, 0, 1, cv::NORM_MINMAX,
  // CV_32F); cv::normalize(imgOrientationOut, imgOrientation, 0, 1,
  // cv::NORM_MINMAX,
  //               CV_32F);

  // cv::imshow("Coherency", imgCoherency);
  // cv::imshow("Orientation", imgOrientation);

  double C_Thr = 0.60;   // threshold for coherency
  int LowThr = 90 - 20;  // threshold1 for orientation, it ranges from 0 to 180
  int HighThr = 90 + 20; // threshold2 for orientation, it ranges from 0 to 180

  cv::cuda::GpuMat imgCoherencyBin;

  cv::cuda::threshold(imgCoherencyOut, imgCoherencyBin, C_Thr, 255, CV_8UC1);

  // cv::imshow("imgCoherencyBin", imgCoherencyBin);

  cv::cuda::GpuMat imgOrientationBin;
  cv::cuda::inRange(imgOrientationOut, cv::Scalar(LowThr), cv::Scalar(HighThr),
                    imgOrientationBin);

  cv::cuda::GpuMat imgBin;
  cv::cuda::bitwise_and(imgCoherencyBin, imgOrientationBin, imgBin);

  cv::Mat binarized;
  imgBin.download(binarized);
  cv::imshow("Binarized", binarized);

  cv::waitKey(1);

  return {};
}

static cv::Mat sigmoid(const cv::Mat &inputMat) {
  cv::Mat outputMat;
  cv::exp(-inputMat, outputMat);
  cv::divide(1.0, 1.0 + outputMat, outputMat);
  return outputMat;
}

std::vector<Line> LineDetector::detect_lines_yolo(const cv::Mat &frame) {
  constexpr int num_classes = 1;

  constexpr float modelScoreThreshold = 0.6f;
  constexpr float modelNMSThreshold = 0.9f;


  int height = frame.rows;
  int width = frame.cols;

  int length = std::max(height, width);

  cv::Mat square = cv::Mat::zeros(input_size, CV_8UC3);
  int small_height = height * input_size.width / width;
  auto small_roi = square(cv::Rect(0, 0, input_size.height, small_height));
  cv::resize(frame, small_roi, small_roi.size());

  float scaleFactor = static_cast<float>(length) / input_size.height;

  const cv::Mat blob = cv::dnn::blobFromImage(frame,         // input images
                                              1.0f / 255.0f, // scale factor
                                              input_size,    // spatial size
                                              cv::Scalar(),  // mean
                                              true,          // swap RB
                                              true          // crop
  );

  // Set the input for the network
  net.setInput(blob);

  std::vector<cv::Mat> netOutputs;
  std::vector<cv::String> output_names = {"output0", "output1"};
  net.forward(netOutputs, output_names);
  // Usually netOutputs.size() == 2 for YOLOv8-seg:
  //   netOutputs[0] = [1, (4+num_classes+someExtra?), N]
  //   netOutputs[1] = [1, 32, 160, 160] (the 'proto' mask)

  //--------------------------------------------------------------------------
  // 5) Postprocessing for bounding boxes
  //--------------------------------------------------------------------------

  // The first output is typically shaped [1, 85, N] for YOLOv8n-seg,
  // but shape can vary. We'll interpret it similarly to your original code.
  cv::Mat out0 = netOutputs[0]; // dimension: [1, C, N]
  // out0.size[0] = 1, out0.size[1] = # of channels, out0.size[2] = # of
  // candidates
  int numProposals = out0.size[2];
  int dimensions = out0.size[1];

  // We want a 2D shape: [N, C]
  // out0 is float, so we do:
  out0 = out0.reshape(1, {dimensions, numProposals}); // [C, N]
  // Transpose it so we get [N, C]
  cv::transpose(out0, out0); // now it's [N, C]

  float *data = (float *)out0.data;

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<int> num_index; // keep track of the row index that survived

  // For each row, retrieve bounding box + class scores
  for (int i = 0; i < numProposals; ++i) {
    float x = data[i * dimensions + 0];
    float y = data[i * dimensions + 1];
    float w = data[i * dimensions + 2];
    float h = data[i * dimensions + 3];

    // classes are stored after these 4
    float *classes_scores = &data[i * dimensions + 4];

    // Get the max score and corresponding class

    cv::Mat scores(1, num_classes, CV_32FC1, classes_scores);
    cv::Point class_id_point;
    double maxClassScore;
    cv::minMaxLoc(scores, nullptr, &maxClassScore, nullptr, &class_id_point);

    if (maxClassScore > modelScoreThreshold) {
      confidences.push_back((float)maxClassScore);
      class_ids.push_back(class_id_point.x);

      // Convert from [x, y, w, h] to [left, top, width, height]
      int left = static_cast<int>(x - 0.5f * w);
      int top = static_cast<int>(y - 0.5f * h);
      int width = static_cast<int>(w);
      int height = static_cast<int>(h);

      boxes.push_back(cv::Rect(left, top, width, height));
      num_index.push_back(i);
    }
  }

  //--------------------------------------------------------------------------
  // 6) NMS
  //--------------------------------------------------------------------------
  std::vector<int> nms_result;
  // cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold,
  //                   nms_result);
  for(size_t i = 0; i < boxes.size(); i++){
    nms_result.push_back(i);
  }

  // Prepare for final detection objects
  std::vector<SegDetection> detections;
  detections.reserve(nms_result.size());

  for (auto idx : nms_result) {
    SegDetection d;
    d.class_id = class_ids[idx];
    d.confidence = confidences[idx];
    d.color = cv::Scalar(111, 222, 111); // or random color, if you prefer
    d.className = "Class_" + std::to_string(d.class_id);
    d.box = boxes[idx];

    // Copy the mask coefficients from the tail of the row (4 + num_classes +
    // 32) For YOLOv8-seg, the mask coeff starts at index = 4 + num_classes
    float *rowPtr = &data[num_index[idx] * dimensions];
    std::copy(rowPtr + 4 + num_classes, rowPtr + 4 + num_classes + 32, d.mask);

    detections.push_back(d);
  }

  //--------------------------------------------------------------------------
  // 7) Mask postprocessing (second output)
  //--------------------------------------------------------------------------
  cv::Mat resultimage(input_size.height, input_size.width, CV_8UC1,
                      cv::Scalar(0));

  // netOutputs[1] should be [1, 32, 160, 160]
  cv::Mat out1 = netOutputs[1];
  // out1.size[0] = 1, out1.size[1] = 32, out1.size[2] = 160, out1.size[3] = 160
  int protoChannels = out1.size[1]; // 32
  int protoHeight = out1.size[2];   // 160
  int protoWidth = out1.size[3];    // 160

  // We want a 2D shape [32, 160 * 160]
  cv::Mat maskMat(protoChannels, protoHeight * protoWidth, CV_32F,
                  out1.ptr<float>());

  // Loop over detections to generate segmentation masks
  for (auto &detection : detections) {
    // 1x32 row for the detection
    cv::Mat tmpMat(1, 32, CV_32F, detection.mask);

    // Multiply by the proto mask: result -> 1x(160*160)
    cv::Mat tmpResult;
    cv::gemm(tmpMat, maskMat, 1.0, cv::Mat(), 0.0, tmpResult); // 1 x (160*160)

    // Reshape to 160x160
    cv::Mat mask2D = tmpResult.reshape(1, protoHeight); // 160 rows

    // Apply sigmoid
    // cv::Mat sigmoidMat = sigmoid(mask2D);
    cv::Mat sigmoidMat = mask2D;

    // Crop from the smaller 160x160 scale
    // Downscaling by factor of 4 from 640 -> 160
    cv::Rect rfor4(detection.box.x / 4, detection.box.y / 4,
                   detection.box.width / 4, detection.box.height / 4);

    // Safety checks
    rfor4 &= cv::Rect(0, 0, protoWidth, protoHeight); // to avoid out-of-bounds
    if (rfor4.empty()) {
      continue;
    }

    // Extract region of interest from proto mask
    cv::Mat roi = sigmoidMat(rfor4);

    // Resize ROI back to detection box size
    cv::Mat crop_mask;
    cv::resize(roi, crop_mask,
               cv::Size(detection.box.width, detection.box.height), 0, 0,
               cv::INTER_CUBIC);

    // Optional: blur and threshold
    cv::Mat blurredImage;
    cv::blur(crop_mask, blurredImage, cv::Size(3, 3));

    cv::Mat thresholdMat;
    // cv::threshold(blurredImage, thresholdMat, 0.5, 255, cv::THRESH_BINARY);
    cv::threshold(blurredImage, thresholdMat, 0.1, 255, cv::THRESH_BINARY);
    thresholdMat.convertTo(thresholdMat, CV_8UC1);

    // Place the final mask into the result image
    cv::Rect boundingRect(detection.box.x, detection.box.y, detection.box.width,
                          detection.box.height);
    boundingRect &= cv::Rect(0, 0, resultimage.cols, resultimage.rows);

    if (boundingRect.width > 0 && boundingRect.height > 0) {
      cv::Mat roi2 = resultimage(boundingRect);
      thresholdMat.copyTo(roi2);
    }
  }

  cv::imshow("Result image", resultimage);
  cv::waitKey(1);

  return {};
}
LineDetector::LineDetector() {
  net = cv::dnn::readNetFromONNX("line_detector.onnx");
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
}
