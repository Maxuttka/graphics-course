#include "App.hpp"

#include <imgui.h>
#include <tracy/Tracy.hpp>

#include "gui/ImGuiRenderer.hpp"


App::App()
{
  glm::uvec2 initialResolution{1280, 720};
  mainWindow = windowing.createWindow(OsWindow::CreateInfo{
    .resolution = initialResolution,
    .resizeable = true,
    .refreshCb =
      [this]() {
        drawFrame();
        FrameMark;
      },
    .resizeCb =
      [this](glm::uvec2 resolution) {
        if (resolution.x != 0 && resolution.y != 0)
          renderer->recreateSwapchain(resolution);
      },
  });

  renderer = std::make_unique<Renderer>(initialResolution);
  renderer->initVulkan(windowing.getRequiredVulkanInstanceExtensions());

  auto surface = mainWindow->createVkSurface(etna::get_context().getInstance());
  renderer->initFrameDelivery(
    std::move(surface), [window = mainWindow.get()]() { return window->getResolution(); });

  ImGuiRenderer::enableImGuiForWindow(mainWindow->native());
  camera.lookAt({0.0f, 28.0f, 42.0f}, {0.0f, 4.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
}

void App::run()
{
  double previousTime = windowing.getTime();

  while (!mainWindow->isBeingClosed())
  {
    double currentTime = windowing.getTime();
    float dt = static_cast<float>(currentTime - previousTime);
    previousTime = currentTime;

    windowing.poll();
    processInput(dt);
    drawFrame();
    FrameMark;
  }
}

void App::processInput(float dt)
{
  if (mainWindow->keyboard[KeyboardKey::kEscape] == ButtonState::Falling)
    mainWindow->askToClose();

  ImGuiIO& io = ImGui::GetIO();

  if (!io.WantCaptureMouse && mainWindow->mouse[MouseButton::mbRight] == ButtonState::Rising)
    mainWindow->captureMouse = !mainWindow->captureMouse;

  if (!io.WantCaptureKeyboard)
    moveCamera(dt);

  if (mainWindow->captureMouse && !io.WantCaptureMouse)
    rotateCamera();
}

void App::moveCamera(float dt)
{
  glm::vec3 direction{0.0f};

  if (is_held_down(mainWindow->keyboard[KeyboardKey::kW]))
    direction += camera.forward();
  if (is_held_down(mainWindow->keyboard[KeyboardKey::kS]))
    direction -= camera.forward();
  if (is_held_down(mainWindow->keyboard[KeyboardKey::kD]))
    direction += camera.right();
  if (is_held_down(mainWindow->keyboard[KeyboardKey::kA]))
    direction -= camera.right();
  if (is_held_down(mainWindow->keyboard[KeyboardKey::kE]))
    direction += glm::vec3{0.0f, 1.0f, 0.0f};
  if (is_held_down(mainWindow->keyboard[KeyboardKey::kQ]))
    direction -= glm::vec3{0.0f, 1.0f, 0.0f};

  float speed = is_held_down(mainWindow->keyboard[KeyboardKey::kLeftShift])
    ? moveSpeed * 3.0f
    : moveSpeed;

  if (glm::length(direction) > 0.0001f)
    camera.move(glm::normalize(direction) * speed * dt);
}

void App::rotateCamera()
{
  camera.rotate(
    -rotateSpeed * mainWindow->mouse.capturedPosDelta.y,
    -rotateSpeed * mainWindow->mouse.capturedPosDelta.x);
}

void App::drawFrame()
{
  renderer->update(FramePacket{.camera = camera});
  renderer->drawFrame();
}
