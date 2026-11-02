#include "App.hpp"

#include <chrono>

#include <etna/DescriptorSet.hpp>
#include <etna/Etna.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/RenderTargetStates.hpp>


namespace
{
constexpr vk::Format PROCEDURAL_TEXTURE_FORMAT = vk::Format::eR8G8B8A8Unorm;
}


App::App()
  : resolution{1280, 720}
  , useVsync{true}
{
  // First, we need to initialize Vulkan, which is not trivial because
  // extensions are required for just about anything.
  {
    // GLFW tells us which extensions it needs to present frames to the OS window.
    // Actually rendering anything to a screen is optional in Vulkan, you can
    // alternatively save rendered frames into files, send them over network, etc.
    // Instance extensions do not depend on the actual GPU, only on the OS.
    auto glfwInstExts = windowing.getRequiredVulkanInstanceExtensions();
    std::vector<const char*> instanceExtensions{glfwInstExts.begin(), glfwInstExts.end()};

    // We also need the swapchain device extension to get access to the OS
    // window from inside of Vulkan on the GPU.
    // Device extensions require HW support from the GPU.
    // Generally, in Vulkan, we call the GPU a "device" and the CPU/OS combination a "host."
    std::vector<const char*> deviceExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    // Etna does all of the Vulkan initialization heavy lifting.
    // You can skip figuring out how it works for now.
    etna::initialize(etna::InitParams{
      .applicationName = "Local Shadertoy 2",
      .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
      .instanceExtensions = instanceExtensions,
      .deviceExtensions = deviceExtensions,
      .physicalDeviceIndexOverride = {},
      .numFramesInFlight = 1,
    });
  }

  // Next, we need a magical Etna helper to send commands to the GPU.
  // How it is actually performed is not trivial, but we can skip this for now.
  commandManager = etna::get_context().createPerFrameCmdMgr();

  // Now we can create an OS window
  osWindow = windowing.createWindow(OsWindow::CreateInfo{
    .resolution = resolution,
  });

  // But we also need to hook the OS window up to Vulkan manually!
  {
    // First, we ask GLFW to provide a "surface" for the window,
    // which is an opaque description of the area where we can actually render.
    auto surface = osWindow->createVkSurface(etna::get_context().getInstance());

    // Then we pass it to Etna to do the complicated work for us
    vkWindow = etna::get_context().createWindow(etna::Window::CreateInfo{
      .surface = std::move(surface),
    });

    // And finally ask Etna to create the actual swapchain so that we can
    // get (different) images each frame to render stuff into.
    // Here, we do not support window resizing, so we only need to call this once.
    auto [w, h] = vkWindow->recreateSwapchain(etna::Window::DesiredProperties{
      .resolution = {resolution.x, resolution.y},
      .vsync = useVsync,
      .numFramesInFlight = static_cast<uint32_t>(commandManager->getCmdBufferCount()),
    });

    // Technically, Vulkan might fail to initialize a swapchain with the requested
    // resolution and pick a different one. This, however, does not occur on platforms
    // we support. Still, it's better to follow the "intended" path.
    resolution = {w, h};
  }

  auto& context = etna::get_context();

  bufferAImage = context.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "shadertoy_buffer_a",
    .format = PROCEDURAL_TEXTURE_FORMAT,
    .imageUsage =
      vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
  });

  bufferBImage = context.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "shadertoy_buffer_b",
    .format = PROCEDURAL_TEXTURE_FORMAT,
    .imageUsage =
      vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
  });

  proceduralSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eLinear,
    .addressMode = vk::SamplerAddressMode::eRepeat,
    .name = "procedural_texture_sampler",
  });

  etna::create_program(
    "shadertoy_buffer_a",
    {LOCAL_SHADERTOY2_SHADERS_ROOT "fullscreen.vert.spv",
     LOCAL_SHADERTOY2_SHADERS_ROOT "buffer_a.frag.spv"});
  etna::create_program(
    "shadertoy_buffer_b",
    {LOCAL_SHADERTOY2_SHADERS_ROOT "fullscreen.vert.spv",
     LOCAL_SHADERTOY2_SHADERS_ROOT "buffer_b.frag.spv"});
  etna::create_program(
    "shadertoy_image",
    {LOCAL_SHADERTOY2_SHADERS_ROOT "fullscreen.vert.spv",
     LOCAL_SHADERTOY2_SHADERS_ROOT "image.frag.spv"});

  const auto makeFullscreenPipeline = [&](const char* programName, vk::Format outputFormat) {
    return context.getPipelineManager().createGraphicsPipeline(
      programName,
      etna::GraphicsPipeline::CreateInfo{
        .depthConfig =
          vk::PipelineDepthStencilStateCreateInfo{
            .depthTestEnable = vk::False,
            .depthWriteEnable = vk::False,
          },
        .fragmentShaderOutput =
          {
            .colorAttachmentFormats = {outputFormat},
          },
      });
  };

  bufferAPipeline = makeFullscreenPipeline("shadertoy_buffer_a", PROCEDURAL_TEXTURE_FORMAT);
  bufferBPipeline = makeFullscreenPipeline("shadertoy_buffer_b", PROCEDURAL_TEXTURE_FORMAT);
  imagePipeline = makeFullscreenPipeline("shadertoy_image", vkWindow->getCurrentFormat());
}

App::~App()
{
  ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
}

void App::run()
{
  while (!osWindow->isBeingClosed())
  {
    windowing.poll();
    drawFrame();
  }

  // We need to wait for the GPU to execute the last frame before destroying
  // all resources and closing the application.
  ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
}

void App::drawFrame()
{
  // First, get a command buffer to write GPU commands into.
  auto currentCmdBuf = commandManager->acquireNext();

  // Next, tell Etna that we are going to start processing the next frame.
  etna::begin_frame();

  // And now get the image we should be rendering the picture into.
  auto nextSwapchainImage = vkWindow->acquireNext();

  // When window is minimized, we can't render anything in Windows
  // because it kills the swapchain, so we skip frames in this case.
  if (nextSwapchainImage)
  {
    auto [backbuffer, backbufferView, backbufferAvailableSem, backbufferReadyForPresentSem] =
      *nextSwapchainImage;

    static const auto startTime = std::chrono::steady_clock::now();
    const float timeSeconds =
      std::chrono::duration<float>(std::chrono::steady_clock::now() - startTime).count();
    const PushConstants params{
      .resolution = glm::vec2{resolution},
      .time = timeSeconds,
    };

    const vk::Rect2D renderArea{{0, 0}, {resolution.x, resolution.y}};

    ETNA_CHECK_VK_RESULT(currentCmdBuf.begin(vk::CommandBufferBeginInfo{}));
    {
      {
        etna::RenderTargetState renderTargets(
          currentCmdBuf,
          renderArea,
          {{.image = bufferAImage.get(), .view = bufferAImage.getView({})}},
          {});

        currentCmdBuf.bindPipeline(
          vk::PipelineBindPoint::eGraphics, bufferAPipeline.getVkPipeline());
        currentCmdBuf.pushConstants<PushConstants>(
          bufferAPipeline.getVkPipelineLayout(),
          vk::ShaderStageFlagBits::eFragment,
          0,
          {params});
        currentCmdBuf.draw(3, 1, 0, 0);
      }

      {
        etna::RenderTargetState renderTargets(
          currentCmdBuf,
          renderArea,
          {{.image = bufferBImage.get(), .view = bufferBImage.getView({})}},
          {});

        currentCmdBuf.bindPipeline(
          vk::PipelineBindPoint::eGraphics, bufferBPipeline.getVkPipeline());
        currentCmdBuf.pushConstants<PushConstants>(
          bufferBPipeline.getVkPipelineLayout(),
          vk::ShaderStageFlagBits::eFragment,
          0,
          {params});
        currentCmdBuf.draw(3, 1, 0, 0);
      }

      const auto imageProgram = etna::get_shader_program("shadertoy_image");
      auto descriptorSet = etna::create_descriptor_set(
        imageProgram.getDescriptorLayoutId(0),
        currentCmdBuf,
        {
          etna::Binding{
            0,
            bufferAImage.genBinding(
              proceduralSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
          etna::Binding{
            1,
            bufferBImage.genBinding(
              proceduralSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
        });

      {
        etna::RenderTargetState renderTargets(
          currentCmdBuf,
          renderArea,
          {{.image = backbuffer, .view = backbufferView}},
          {});

        currentCmdBuf.bindPipeline(
          vk::PipelineBindPoint::eGraphics, imagePipeline.getVkPipeline());
        currentCmdBuf.bindDescriptorSets(
          vk::PipelineBindPoint::eGraphics,
          imagePipeline.getVkPipelineLayout(),
          0,
          {descriptorSet.getVkSet()},
          {});
        currentCmdBuf.pushConstants<PushConstants>(
          imagePipeline.getVkPipelineLayout(),
          vk::ShaderStageFlagBits::eFragment,
          0,
          {params});
        currentCmdBuf.draw(3, 1, 0, 0);
      }

      // At the end of "rendering", we are required to change how the pixels of the
      // swpchain image are laid out in memory to something that is appropriate
      // for presenting to the window (while preserving the content of the pixels!).
      etna::set_state(
        currentCmdBuf,
        backbuffer,
        // This looks weird, but is correct. Ask about it later.
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        {},
        vk::ImageLayout::ePresentSrcKHR,
        vk::ImageAspectFlagBits::eColor);
      // And of course flush the layout transition.
      etna::flush_barriers(currentCmdBuf);
    }
    ETNA_CHECK_VK_RESULT(currentCmdBuf.end());

    // We are done recording GPU commands now and we can send them to be executed by the GPU.
    // Note that the GPU won't start executing our commands before the backbufferAvailableSem
    // semaphore is signalled, which will happen when the OS says that the next swapchain image
    // is ready, and the result image will be ready for present after backbufferReadyForPresent
    // is signalled by GPU
    auto renderingDone = commandManager->submit(
      std::move(currentCmdBuf),
      std::move(backbufferAvailableSem),
      std::move(backbufferReadyForPresentSem));

    // Finally, present the backbuffer the screen, but only after the GPU tells the OS
    // that it is done executing the command buffer via the renderingDone semaphore.
    const bool presented = vkWindow->present(std::move(renderingDone), backbufferView);
    if (!presented)
      nextSwapchainImage = std::nullopt;
  }

  etna::end_frame();

  // After a window us un-minimized, we need to restore the swapchain to continue rendering.
  if (!nextSwapchainImage && osWindow->getResolution() != glm::uvec2{0, 0})
  {
    auto [w, h] = vkWindow->recreateSwapchain(etna::Window::DesiredProperties{
      .resolution = {resolution.x, resolution.y},
      .vsync = useVsync,
      .numFramesInFlight = static_cast<uint32_t>(commandManager->getCmdBufferCount()),
    });
    ETNA_VERIFY((resolution == glm::uvec2{w, h}));
  }
}
