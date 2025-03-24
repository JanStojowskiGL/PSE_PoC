#include "gui_app_base.h"
#include "app.h"
#include "fmt/base.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

static void glfw_error_callback(int error, const char *description) {
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

GuiAppBase::GuiAppBase() {
  // Setup GLFW
  initialize_glfw();
  setup_imgui();
}

int GuiAppBase::initialize_glfw() {
  if (!glfwInit()) {
    std::fprintf(stderr, "Failed to initialize GLFW\n");
    return 1;
  }
  glfwSetErrorCallback(glfw_error_callback);

  // (Optional) Set some GLFW hints for an OpenGL 3.x context
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // required on macOS
#endif

  // Create a windowed mode window and its OpenGL context
  window = glfwCreateWindow(1280, 720, "PSE_PoC", nullptr, nullptr);

  if (!window) {
    std::fprintf(stderr, "Failed to create GLFW window\n");
    glfwTerminate();
    return 1;
  }

  // Make the window's context current
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1); // Enable vsync
  return 0;
}

int GuiAppBase::setup_imgui() {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  // Optionally enable/disable various flags in io.ConfigFlags here.

  // Setup ImGui style
  ImGui::StyleColorsDark();

  // Initialize ImGui backends
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  // Use GLSL version that matches your context. "#version 130" works for
  // OpenGL 3.0+.
  ImGui_ImplOpenGL3_Init("#version 130");
  return 0;
}

bool GuiAppBase::should_close() { return glfwWindowShouldClose(window); }

GuiAppBase::~GuiAppBase() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();
}
void GuiAppBase::draw_frame() {
  // Poll and handle input events (keyboard, mouse, etc.)
  glfwPollEvents();

  // Start a new ImGui frame
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  imgui_ui();

  // Render ImGui's commands
  ImGui::Render();

  // Set your viewport and clear the screen
  int display_w, display_h;
  glfwGetFramebufferSize(window, &display_w, &display_h);
  glViewport(0, 0, display_w, display_h);
  glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
  glClear(GL_COLOR_BUFFER_BIT);

  // Draw ImGui data
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  // Swap the front and back buffers
  glfwSwapBuffers(window);
}

