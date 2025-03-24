#include "app.h"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "async_queue.h"
#include "gui_app.h"
#include "imgui.h"
#include "prediction.h"

int main(int argc, char**argv){
    if (argc < 2){
        std::cerr << "Usage: " << argv[0] << " <video_file_path>\n";
        std::exit(1);
    }
    
    const std::filesystem::path video_path = argv[1];

    AsyncQueue<Prediction> prediction_queue(2);

    App app{video_path};
    app.prediction_queue = &prediction_queue;
    app.start_worker();

    GuiApp gui_app{};
    gui_app.prediction_queue = &prediction_queue;

    while(!gui_app.should_close())
    {
        gui_app.draw_frame();
    }
    return 0;
}
