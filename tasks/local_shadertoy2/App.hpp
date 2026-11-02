#pragma once

#include <etna/GraphicsPipeline.hpp>
#include <etna/Image.hpp>
#include <etna/PerFrameCmdMgr.hpp>
#include <etna/Sampler.hpp>
#include <etna/Window.hpp>

#include "wsi/OsWindowingManager.hpp"


class App
{
public:
  App();
  ~App();

  void run();

private:
  void drawFrame();

private:
  struct alignas(16) PushConstants
  {
    glm::vec2 resolution;
    float time;
    float padding = 0.0f;
  };

  OsWindowingManager windowing;
  std::unique_ptr<OsWindow> osWindow;

  glm::uvec2 resolution;
  bool useVsync;

  std::unique_ptr<etna::Window> vkWindow;
  std::unique_ptr<etna::PerFrameCmdMgr> commandManager;

  etna::GraphicsPipeline bufferAPipeline;
  etna::GraphicsPipeline bufferBPipeline;
  etna::GraphicsPipeline imagePipeline;

  etna::Image bufferAImage;
  etna::Image bufferBImage;
  etna::Sampler proceduralSampler;
};
