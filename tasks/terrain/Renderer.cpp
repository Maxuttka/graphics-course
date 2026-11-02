#include "Renderer.hpp"

#include <imgui.h>

#include <etna/Etna.hpp>
#include <etna/Profiling.hpp>
#include <gui/ImGuiRenderer.hpp>


Renderer::Renderer(glm::uvec2 initialResolution)
  : resolution{initialResolution}
{
}

void Renderer::initVulkan(std::span<const char*> instanceExtensions)
{
  std::vector<const char*> extensions{instanceExtensions.begin(), instanceExtensions.end()};
  std::vector<const char*> deviceExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  etna::initialize(etna::InitParams{
    .applicationName = "Terrain",
    .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
    .instanceExtensions = extensions,
    .deviceExtensions = deviceExtensions,
    .features = vk::PhysicalDeviceFeatures2{.features = {}},
    .physicalDeviceIndexOverride = {},
    .numFramesInFlight = 2,
  });
}

void Renderer::initFrameDelivery(
  vk::UniqueSurfaceKHR surface, ResolutionProvider newResolutionProvider)
{
  resolutionProvider = std::move(newResolutionProvider);
  auto& context = etna::get_context();
  commandManager = context.createPerFrameCmdMgr();
  window = context.createWindow(etna::Window::CreateInfo{.surface = std::move(surface)});

  auto [width, height] = window->recreateSwapchain(etna::Window::DesiredProperties{
    .resolution = {resolution.x, resolution.y},
    .vsync = true,
    .numFramesInFlight = static_cast<uint32_t>(commandManager->getCmdBufferCount()),
  });
  resolution = {width, height};

  worldRenderer = std::make_unique<WorldRenderer>();
  worldRenderer->allocateResources(resolution);
  worldRenderer->loadShaders();
  worldRenderer->setupPipelines(window->getCurrentFormat());
  guiRenderer = std::make_unique<ImGuiRenderer>(window->getCurrentFormat());
}

void Renderer::recreateSwapchain(glm::uvec2 newResolution)
{
  ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());

  auto [width, height] = window->recreateSwapchain(etna::Window::DesiredProperties{
    .resolution = {newResolution.x, newResolution.y},
    .vsync = true,
    .numFramesInFlight = static_cast<uint32_t>(commandManager->getCmdBufferCount()),
  });
  resolution = {width, height};
  worldRenderer->allocateResources(resolution);
  worldRenderer->setupPipelines(window->getCurrentFormat());
}

void Renderer::update(const FramePacket& packet)
{
  worldRenderer->update(packet);
}

void Renderer::drawFrame()
{
  guiRenderer->nextFrame();
  ImGui::NewFrame();
  worldRenderer->drawGui();
  ImGui::Render();

  auto commandBuffer = commandManager->acquireNext();
  etna::begin_frame();
  auto nextImage = window->acquireNext();

  if (nextImage)
  {
    auto [image, view, availableSemaphore, presentSemaphore] = *nextImage;
    ETNA_CHECK_VK_RESULT(commandBuffer.begin(vk::CommandBufferBeginInfo{}));

    worldRenderer->render(commandBuffer, image, view);
    guiRenderer->render(
      commandBuffer,
      {{0, 0}, {resolution.x, resolution.y}},
      image,
      view,
      ImGui::GetDrawData());

    etna::set_state(
      commandBuffer,
      image,
      vk::PipelineStageFlagBits2::eColorAttachmentOutput,
      {},
      vk::ImageLayout::ePresentSrcKHR,
      vk::ImageAspectFlagBits::eColor);
    etna::flush_barriers(commandBuffer);

    ETNA_CHECK_VK_RESULT(commandBuffer.end());
    auto renderingDone = commandManager->submit(
      std::move(commandBuffer), std::move(availableSemaphore), std::move(presentSemaphore));

    if (!window->present(std::move(renderingDone), view))
      nextImage = std::nullopt;
  }

  etna::end_frame();

  if (!nextImage)
  {
    glm::uvec2 newResolution = resolutionProvider();
    if (newResolution.x != 0 && newResolution.y != 0)
      recreateSwapchain(newResolution);
  }
}

Renderer::~Renderer()
{
  ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
}
