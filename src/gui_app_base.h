#pragma once
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <stdio.h>

struct GuiAppBase {
  GuiAppBase();
  virtual ~GuiAppBase();

  virtual void imgui_ui() = 0;

  int initialize_glfw();
  int setup_imgui();
  void draw_frame();
  bool should_close();

  GLFWwindow *window;
};
