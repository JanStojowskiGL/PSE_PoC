#include "app.h"
#include "line_detection.h"
#include "yolo_detection.h"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char**argv){
    if (argc < 2){
        std::cerr << "Usage: " << argv[0] << " <video_file_path>\n";
        std::exit(1);
    }
    
    const std::filesystem::path video_path = argv[1];
    cv::VideoCapture cap{video_path};
    cv::Mat frame;
    App app{};
    while (true) {
        const auto tick = std::chrono::high_resolution_clock::now();
        cap.read(frame);
        if (frame.empty()){
            break;
        }
        app.predict(frame);

        const auto tock = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tock - tick);
        std::cout << "Took: " << duration.count() << "ms\n";

        const auto key = cv::waitKey(1);
        if (key == 'q' || key == 27) { // 27 - ESC
            break;
        }
    }
    return 0;
}
