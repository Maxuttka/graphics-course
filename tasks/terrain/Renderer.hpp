#pragma once

#include <etna/GlobalContext.hpp>
#include <etna/PerFrameCmdMgr.hpp>
#include <function2/function2.hpp>

#include "FramePacket.hpp"
#include "WorldRenderer.hpp"


class ImGuiRenderer;

using ResolutionProvider = fu2::unique_function<glm::uvec2() const>;

class Renderer
{
public:
  explicit Renderer(glm::uvec2 resolution);
  ~Renderer();

  void initVulkan(std::span<const char*> instanceExtensions);
  void initFrameDelivery(vk::UniqueSurfaceKHR surface, ResolutionProvider resolutionProvider);
  void recreateSwapchain(glm::uvec2 resolution);
  void update(const FramePacket& packet);
  void drawFrame();

private:
  ResolutionProvider resolutionProvider;
  std::unique_ptr<etna::Window> window;
  std::unique_ptr<etna::PerFrameCmdMgr> commandManager;
  std::unique_ptr<ImGuiRenderer> guiRenderer;
  std::unique_ptr<WorldRenderer> worldRenderer;
  glm::uvec2 resolution;
};
