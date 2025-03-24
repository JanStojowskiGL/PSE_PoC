cmake_minimum_required(VERSION 3.10.0)
project(pse_poc VERSION 0.1.0 LANGUAGES C CXX)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 20 REQUIRED)

add_subdirectory(dependencies)

include_directories(${OpenCV_INCLUDE_DIRS})

include(FetchContent)
FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG master
)
FetchContent_MakeAvailable(fmt)

# add_compile_options(-fsanitize=address)
# add_link_options(-fsanitize=address)

add_executable(video_demo 
src/main.cpp
src/yolo_detection.cpp
src/line_detection.cpp
src/app.cpp
src/gui_app.cpp
src/gui_app_base.cpp
)

target_link_libraries(video_demo ${OpenCV_LIBS} fmt::fmt imgui_glfw)
