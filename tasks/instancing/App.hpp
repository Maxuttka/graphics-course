#pragma once

#include "wsi/OsWindowingManager.hpp"
#include "scene/Camera.hpp"

#include "Renderer.hpp"


class App
{
public:
  App();

  void run();

private:
  void processInput(float dt);
  void drawFrame();
  void moveCamera(float dt);
  void rotateCamera();

  OsWindowingManager windowing;
  std::unique_ptr<OsWindow> mainWindow;
  std::unique_ptr<Renderer> renderer;
  Camera camera;
  float moveSpeed = 12.0f;
  float rotateSpeed = 0.1f;
};
