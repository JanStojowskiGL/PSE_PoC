#pragma once

#include "async_queue.h"
#include "gui_app_base.h"
#include "prediction.h"
#include <chrono>
#include <cstdint>
#include <thread>

struct GuiApp : public GuiAppBase {
  GuiApp();
  ~GuiApp();
  void imgui_ui();
  void RenderSplitScreenWindow();

  void draw_left_pane();
  void draw_right_pane();

  bool shouldTakePicture{false};
  std::chrono::time_point<std::chrono::steady_clock> take_picture_start;

  std::vector<uint32_t> captured_ids;
  bool was_captured(uint32_t id) const;

  GLuint imageTexture = 0;

  AsyncQueue<Prediction> *prediction_queue;

  Prediction current_prediction{};
  void overlay_lines(const Prediction &pred, ImVec2 drawSize, ImVec2 cursorPos);
};