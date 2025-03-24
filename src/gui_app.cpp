#include "fmt/format.h"
#include <cfloat>
#define IMGUI_DEFINE_MATH_OPERATORS
#include "gui_app.h"
#include "fmt/base.h"
#include "gui_app_base.h"
#include "imgui.h"
#include "prediction.h"
#include "utils.h"
#include <GL/gl.h>
#include <chrono>
#include <cmath>

void GuiApp::RenderSplitScreenWindow() {
  static float dividerFraction = 0.8f;     // 50% left pane by default
  static const float dividerWidth = 4.0f;  // Width of the draggable divider
  static const float minPaneWidth = 50.0f; // Minimum width for each pane

  // Obtain the display size from ImGui IO
  ImGuiIO &io = ImGui::GetIO();
  ImVec2 displaySize = io.DisplaySize;

  // Set up a full-screen window
  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(displaySize);
  ImGuiWindowFlags window_flags =
      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;
  ImGui::Begin("FullScreenSplitWindow", nullptr, window_flags);

  // Get the available content region inside the window
  ImVec2 availSize = ImGui::GetContentRegionAvail();

  // Calculate the left pane width (ensuring it stays within a min/max range)
  float leftPaneWidth = availSize.x * dividerFraction;
  if (leftPaneWidth < minPaneWidth)
    leftPaneWidth = minPaneWidth;
  if (leftPaneWidth > availSize.x - dividerWidth - minPaneWidth)
    leftPaneWidth = availSize.x - dividerWidth - minPaneWidth;
  float rightPaneWidth = availSize.x - leftPaneWidth - dividerWidth;

  // Create the left pane as a child window
  ImGui::BeginChild("LeftPane", ImVec2(leftPaneWidth, availSize.y), true);
  draw_left_pane();
  // Additional UI elements for the left pane can be added here.
  ImGui::EndChild();

  // Place the divider on the same line as the left pane
  ImGui::SameLine();

  // Render the divider as a button with custom colors so that it appears as a
  // draggable bar.
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
  if (ImGui::Button("##Divider", ImVec2(dividerWidth, availSize.y))) {
    // The button itself does nothing; we handle dragging below.
  }
  ImGui::PopStyleColor(3);

  // If the divider is active and the mouse is dragging, update the divider
  // fraction.
  if (ImGui::IsItemActive() && ImGui::IsMouseDragging(0)) {
    float mouseDelta = ImGui::GetIO().MouseDelta.x;
    leftPaneWidth += mouseDelta;
    dividerFraction = leftPaneWidth / availSize.x;
    // Clamp the divider fraction so that both panes maintain at least
    // minPaneWidth.
    if (dividerFraction < minPaneWidth / availSize.x)
      dividerFraction = minPaneWidth / availSize.x;
    if (dividerFraction >
        (availSize.x - dividerWidth - minPaneWidth) / availSize.x)
      dividerFraction =
          (availSize.x - dividerWidth - minPaneWidth) / availSize.x;
  }

  ImGui::SameLine();

  // Create the right pane as a child window
  ImGui::BeginChild("RightPane", ImVec2(rightPaneWidth, availSize.y), true);
  draw_right_pane();
  // Additional UI elements for the right pane can be added here.
  ImGui::EndChild();

  ImGui::End();
}

static GLuint matToTexture(const cv::Mat &mat, GLuint &textureID) {
  if (mat.empty()) {
    std::fprintf(stderr, "matToTexture: Input image is empty!\n");
    return 0;
  }

  // Generate and bind texture
  // GLuint textureID;
  glGenTextures(1, &textureID);
  glBindTexture(GL_TEXTURE_2D, textureID);

  // Set texture filtering parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  // Decide GL texture format based on cv::Mat channels
  GLenum inputColorFormat =
      GL_BGR; // OpenCV usually stores in BGR order for CV_8UC3
  // If you have 4 channels (e.g., CV_8UC4), use GL_BGRA
  // If you have 1 channel (CV_8UC1), use GL_RED or GL_ALPHA, etc.

  // Upload the image data to the GPU
  glTexImage2D(GL_TEXTURE_2D, 0,
               GL_RGB, // Internal format
               mat.cols, mat.rows, 0, inputColorFormat, GL_UNSIGNED_BYTE,
               mat.data);

  // Unbind texture
  glBindTexture(GL_TEXTURE_2D, 0);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mat.cols, mat.rows, 0,
               inputColorFormat, GL_UNSIGNED_BYTE, mat.data);

  return textureID;
}

static void matToTexture2(const cv::Mat &mat, GLuint &textureID) {
  if (mat.empty()) {
    std::fprintf(stderr, "matToTexture: Input Mat is empty.\n");
    return;
  }

  // If this is the first time, generate a new texture ID.
  if (textureID == 0) {
    glGenTextures(1, &textureID);
  }

  // Bind the texture so we can upload/update data.
  glBindTexture(GL_TEXTURE_2D, textureID);

  // Set basic texture parameters (here: linear filtering).
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  // Optionally set wrapping modes:
  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  // Decide the proper format: here we assume CV_8UC3 -> BGR
  GLenum inputColorFormat = GL_BGR;
  // If you have 4 channels (CV_8UC4), use GL_BGRA
  // If 1 channel (CV_8UC1), you might use GL_RED or GL_ALPHA, etc.

  // Upload the image data to the texture. (Re)allocate if size changes.
  glTexImage2D(GL_TEXTURE_2D, 0,
               GL_RGB, // internal format
               mat.cols, mat.rows, 0, inputColorFormat, GL_UNSIGNED_BYTE,
               mat.data);

  // Unbind the texture
  glBindTexture(GL_TEXTURE_2D, 0);
}

void matToTexture3(const cv::Mat &frame, GLuint &textureID) {
  const unsigned char *data = frame.ptr();
  const int width = frame.cols;
  const int height = frame.rows;
  if (textureID == 0)
    glGenTextures(1, &textureID);
  glBindTexture(GL_TEXTURE_2D, textureID);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR,
               GL_UNSIGNED_BYTE, data);
}

void GuiApp::imgui_ui() {
  RenderSplitScreenWindow();
  auto pred_opt = prediction_queue->pop();
  // return;
  if (pred_opt.has_value()) {
    current_prediction = std::move(pred_opt.value());
    const Prediction &pred = current_prediction;
    // fmt::println("Yaay, pred queue has value!");
    if (!pred.frame.empty())
      // imageTexture = matToTexture(pred.frame);
      matToTexture3(pred.frame, imageTexture);
  }
}

GuiApp::GuiApp() : GuiAppBase() {}

GuiApp::~GuiApp() {
  if (imageTexture != 0) {
    glDeleteTextures(1, &imageTexture);
    imageTexture = 0;
  }
}

void GuiApp::overlay_lines(const Prediction &pred, ImVec2 drawSize, ImVec2 cursorPos) {
  const auto &lines = pred.lines;
  const auto &image = pred.frame;

  for (const auto &line : lines) {
    const auto [x1, y1] = line.a;
    const auto [x2, y2] = line.b;
    // // Suppose we want to fill the available content area while preserving
    // // aspect ratio:

    // // Compute scaled width/height (drawSize), then:
    // ImVec2 cursorPos = ImGui::GetCursorScreenPos();
    // ImGui::Image((ImTextureID)(intptr_t)imageTexture, drawSize);

    // Calculate angle

    // For a line from (x1, y1) to (x2, y2) in original image coordinates:
    float scaleX = drawSize.x / (float)image.cols;
    float scaleY = drawSize.y / (float)image.rows;

    ImVec2 startPt =
        ImVec2(cursorPos.x + x1 * scaleX, cursorPos.y + y1 * scaleY);
    ImVec2 endPt = ImVec2(cursorPos.x + x2 * scaleX, cursorPos.y + y2 * scaleY);

    static const std::array<ImU32, 6> colors{
        IM_COL32(0, 0, 255, 255),   IM_COL32(0, 255, 0, 255),
        IM_COL32(255, 0, 0, 255),

        IM_COL32(0, 255, 255, 255), IM_COL32(255, 255, 0, 255),
        IM_COL32(255, 0, 255, 255),
    };

    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    // draw_list->AddLine(startPt, endPt, colors.at(line.color % colors.size()),
    //                    2.0f);
    draw_list->AddLine(startPt, endPt, IM_COL32(0, 255, 0, 255),
                       2.0f);
  }

  for (const auto &box : pred.boxes) {
    if (box.confidence < 0.5f) {
      continue;
    }
    const auto [x1, y1] = box.box.tl();
    const auto [x2, y2] = box.box.br();
    // fmt::println("x1: {}, y1: {}, x2: {}, y2: {}", x1, y1, x2, y2);
    // // Suppose we want to fill the available content area while preserving
    // // aspect ratio:

    // // Compute scaled width/height (drawSize), then:
    // ImVec2 cursorPos = ImGui::GetCursorScreenPos();
    // ImGui::Image((ImTextureID)(intptr_t)imageTexture, drawSize);

    // For a line from (x1, y1) to (x2, y2) in original image coordinates:
    float scaleX = drawSize.x / (float)image.cols;
    float scaleY = drawSize.y / (float)image.rows;

    ImVec2 startPt =
        ImVec2(cursorPos.x + x1 * scaleX, cursorPos.y + y1 * scaleY);
    ImVec2 endPt = ImVec2(cursorPos.x + x2 * scaleX, cursorPos.y + y2 * scaleY);

    const ImU32 box_color = was_captured(box.id) ? IM_COL32(255, 255, 255, 255) : IM_COL32(0, 0, 255, 255);
    const ImU32 box_color_translucent = (box_color & 0x00FFFFFF) | (100 << 24);

    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRect(startPt, endPt, box_color, 2.0f);
    if (ImGui::IsMouseHoveringRect(startPt, endPt)) {
      draw_list->AddRectFilled(startPt, endPt, box_color_translucent, 2.0f);
    }

    auto center_in_original_image = box.center();
    cv::Size image_size{1920 * 2, 1080 * 2};
    if (center_in_original_image.y > image_size.height * 0.5 && center_in_original_image.y < image_size.height * 0.65 && !was_captured(box.id)){
      captured_ids.push_back(box.id);
      std::sort(captured_ids.begin(), captured_ids.end(), [](int a, int b){return a > b;});
      shouldTakePicture = true;
    }

    const std::string box_text = fmt::format("{}", box.id);
    ImFont* font = ImGui::GetFont();
    float font_size = std::min(std::abs((startPt - endPt).y) * 0.8, 24.0);
    const ImVec2 text_size = font->CalcTextSizeA(font_size, FLT_MAX, 0.0F, box_text.c_str());
    // const ImVec2 text_size = ImGui::CalcTextSize(box_text.c_str());

    ImVec2 box_center = (startPt + endPt) * 0.5F;
    ImVec2 text_pos = box_center - text_size * 0.5F;
    draw_list->AddText(font, font_size, text_pos, box_color, box_text.c_str());

  }
}

void overlay_direction_queue(float angle, ImVec2 image_size, ImVec2 cursorPos) {
  ImDrawList *draw_list = ImGui::GetWindowDrawList();

  float fixing_angle = angle - M_PI_2;
  if (std::abs(fixing_angle) < 2 * M_PI / 180.0F){
    return;
  }
  const float start_angle = -M_PI_2 - fixing_angle;
  const float end_angle = -M_PI_2 + fixing_angle;

  constexpr ImU32 color = IM_COL32(255, 0, 0, 255);
  const ImVec2 center = cursorPos + image_size / 2.0F;
  const float radius = image_size.y * 0.4;

  draw_list->PathArcTo(center, radius, start_angle, end_angle);
  draw_list->PathStroke(color, false, 2.0f);

  // Calculate coords for line
  ImVec2 tangent = ImVec2(-std::sin(end_angle), std::cos(end_angle));
  if(fixing_angle > 0){
    tangent *= -1;
  }

  const ImVec2 perpendicular = ImVec2(std::cos(end_angle), std::sin(end_angle));

  const ImVec2 arc_end =
      center + ImVec2{std::cos(end_angle), std::sin(end_angle)} * radius;

  const float arrow_length = 10.0F;
  const float arrow_width_2 = 6.0F;

  draw_list->AddTriangleFilled(
      arc_end, arc_end + tangent * arrow_length + perpendicular * arrow_width_2,
      arc_end + tangent * arrow_length - perpendicular * arrow_width_2, color);
}

void GuiApp::draw_left_pane() {
  if (imageTexture) {
    const cv::Mat &image = current_prediction.frame;
    // Get the available size in the current window
    ImVec2 availSize = ImGui::GetContentRegionAvail();

    // Original image dimensions
    float imgWidth = static_cast<float>(image.cols);
    float imgHeight = static_cast<float>(image.rows);

    // Calculate aspect ratios
    float imageAspect = imgWidth / imgHeight;
    float windowAspect = availSize.x / availSize.y;

    // Target size to draw
    ImVec2 drawSize;

    // If the image is “wider” than the available region, limit by width
    if (imageAspect > windowAspect) {
      drawSize.x = availSize.x;
      drawSize.y = availSize.x / imageAspect;
    } else {
      // Otherwise, limit by height
      drawSize.y = availSize.y;
      drawSize.x = availSize.y * imageAspect;
    }

    // Finally, draw the image with the chosen size
    ImVec2 cursorPos = ImGui::GetCursorScreenPos();
    ImGui::Image((ImTextureID)(intptr_t)imageTexture, drawSize);
    overlay_lines(current_prediction, drawSize, cursorPos);
    overlay_direction_queue(current_prediction.heading, drawSize, cursorPos);

    float since_take_picture = since(take_picture_start);
    constexpr float take_picture_ms = 500.0F;
    if (since_take_picture < take_picture_ms) {
      float factor = 1 - since_take_picture / take_picture_ms;

    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRectFilled(cursorPos, cursorPos + drawSize, IM_COL32(255, 255, 255, 255 * factor * factor), 2.0f);
    shouldTakePicture = false;
    }
    if (shouldTakePicture){
      shouldTakePicture = false;
      take_picture_start = std::chrono::steady_clock::now();
    }
  } else {
    ImGui::Text("Waiting for image ...");
    static auto start = std::chrono::high_resolution_clock::now();
    float time_since = since(start);
    ImGui::ProgressBar(time_since / 5000.0);
  }
}
void GuiApp::draw_right_pane() {
  ImGui::Text("Mission stats:");

  ImGui::Text("Heading angle: %f", current_prediction.heading * 180.0 / M_PI);

  ImGui::Text("Boxes ms: %.1f", current_prediction.boxes_ms);
  ImGui::Text("Lines ms: %.1f", current_prediction.lines_ms);

// Set desired table height for the scrollable region (e.g., 300 pixels).
    const float tableHeight = 300.0f;
    
    // Begin the table with 1 column, scrollable in the Y direction.
    if (ImGui::BeginTable("ScrollableTable", 1, ImGuiTableFlags_ScrollY | ImGuiTableFlags_Borders, ImVec2(0, tableHeight)))
    {
        // Setup the column and create header
        ImGui::TableSetupColumn("Captured towers");
        ImGui::TableHeadersRow();

        // Iterate over the vector and add each value as a row
        for (size_t i = 0; i < captured_ids.size(); ++i)
        {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%d", captured_ids[i]);
        }

        // End the table
        ImGui::EndTable();
    }
}

bool GuiApp::was_captured(uint32_t id) const {
  for (auto i : captured_ids) {
    if (i == id) {
      return true;
    }
  }
  return false;
}
