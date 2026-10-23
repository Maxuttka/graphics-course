#include "WorldRenderer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <span>
#include <vector>

#include <imgui.h>

#include <etna/BlockingTransferHelper.hpp>
#include <etna/Etna.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/RenderTargetStates.hpp>


namespace
{
constexpr uint32_t GridCells = 256;
constexpr uint32_t HeightMapSize = 1024;
constexpr uint32_t MaterialSize = 64;
constexpr uint32_t MaterialCount = 4;

float randomValue(uint32_t x, uint32_t y, uint32_t layer)
{
  uint32_t value = x * 374761393u + y * 668265263u + layer * 2246822519u;
  value = (value ^ (value >> 13u)) * 1274126177u;
  value ^= value >> 16u;
  return static_cast<float>(value & 255u) / 255.0f;
}

std::span<const std::byte> asBytes(const std::vector<uint8_t>& data)
{
  return {
    reinterpret_cast<const std::byte*>(data.data()),
    data.size(),
  };
}
}

void WorldRenderer::allocateResources(glm::uvec2 newResolution)
{
  resolution = newResolution;
  depthImage = etna::get_context().createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "terrain_depth",
    .format = vk::Format::eD32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
  });

  if (!initialized)
    initializeTerrain();
}

void WorldRenderer::initializeTerrain()
{
  auto& context = etna::get_context();
  heightMap = context.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{HeightMapSize, HeightMapSize, 1},
    .name = "terrain_height_map",
    .format = vk::Format::eR32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
  });
  heightSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eLinear,
    .addressMode = vk::SamplerAddressMode::eClampToEdge,
    .name = "terrain_height_sampler",
  });
  materialSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eLinear,
    .addressMode = vk::SamplerAddressMode::eRepeat,
    .name = "terrain_material_sampler",
  });
  uniformBuffer = context.createBuffer(etna::Buffer::CreateInfo{
    .size = sizeof(Uniforms),
    .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_AUTO,
    .allocationCreate =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
      VMA_ALLOCATION_CREATE_MAPPED_BIT,
    .name = "terrain_uniforms",
  });

  createGrid();
  createMaterialTextures();
  initialized = true;
}

void WorldRenderer::createGrid()
{
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  vertices.reserve((GridCells + 1) * (GridCells + 1));
  indices.reserve(GridCells * GridCells * 6);

  for (uint32_t z = 0; z <= GridCells; ++z)
  {
    for (uint32_t x = 0; x <= GridCells; ++x)
    {
      glm::vec2 uv{
        static_cast<float>(x) / static_cast<float>(GridCells),
        static_cast<float>(z) / static_cast<float>(GridCells),
      };
      vertices.push_back(Vertex{.position = uv - 0.5f, .uv = uv});
    }
  }

  for (uint32_t z = 0; z < GridCells; ++z)
  {
    for (uint32_t x = 0; x < GridCells; ++x)
    {
      uint32_t a = z * (GridCells + 1) + x;
      uint32_t b = a + 1;
      uint32_t c = a + GridCells + 1;
      uint32_t d = c + 1;
      indices.insert(indices.end(), {a, c, b, b, c, d});
    }
  }

  auto& context = etna::get_context();
  vertexBuffer = context.createBuffer(etna::Buffer::CreateInfo{
    .size = vertices.size() * sizeof(Vertex),
    .bufferUsage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    .name = "terrain_vertices",
  });
  indexBuffer = context.createBuffer(etna::Buffer::CreateInfo{
    .size = indices.size() * sizeof(uint32_t),
    .bufferUsage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    .name = "terrain_indices",
  });

  auto commands = context.createOneShotCmdMgr();
  etna::BlockingTransferHelper transfer({.stagingSize = 4 * 1024 * 1024});
  transfer.uploadBuffer<Vertex>(*commands, vertexBuffer, 0, std::span<const Vertex>{vertices});
  transfer.uploadBuffer<uint32_t>(
    *commands, indexBuffer, 0, std::span<const uint32_t>{indices});
  indexCount = static_cast<uint32_t>(indices.size());
}

void WorldRenderer::createMaterialTextures()
{
  auto& context = etna::get_context();
  albedoMaps = context.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{MaterialSize, MaterialSize, 1},
    .name = "terrain_albedo_maps",
    .format = vk::Format::eR8G8B8A8Srgb,
    .imageUsage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
    .layers = MaterialCount,
  });
  roughnessMaps = context.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{MaterialSize, MaterialSize, 1},
    .name = "terrain_roughness_maps",
    .format = vk::Format::eR8Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
    .layers = MaterialCount,
  });
  detailHeightMaps = context.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{MaterialSize, MaterialSize, 1},
    .name = "terrain_detail_height_maps",
    .format = vk::Format::eR8Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
    .layers = MaterialCount,
  });

  constexpr std::array<glm::vec3, MaterialCount> colors{
    glm::vec3{0.67f, 0.52f, 0.30f},
    glm::vec3{0.16f, 0.36f, 0.10f},
    glm::vec3{0.32f, 0.30f, 0.28f},
    glm::vec3{0.88f, 0.91f, 0.94f},
  };
  constexpr std::array<float, MaterialCount> roughness{0.82f, 0.78f, 0.9f, 0.48f};

  auto commands = context.createOneShotCmdMgr();
  etna::BlockingTransferHelper transfer({.stagingSize = 256 * 1024});

  for (uint32_t layer = 0; layer < MaterialCount; ++layer)
  {
    std::vector<uint8_t> albedo(MaterialSize * MaterialSize * 4);
    std::vector<uint8_t> rough(MaterialSize * MaterialSize);
    std::vector<uint8_t> height(MaterialSize * MaterialSize);

    for (uint32_t y = 0; y < MaterialSize; ++y)
    {
      for (uint32_t x = 0; x < MaterialSize; ++x)
      {
        uint32_t pixel = y * MaterialSize + x;
        float noiseValue = randomValue(x, y, layer);
        float wave = 0.5f + 0.5f * std::sin(
          static_cast<float>(x) * (0.28f + 0.05f * static_cast<float>(layer)) +
          static_cast<float>(y) * 0.19f);
        float colorScale = 0.82f + 0.28f * noiseValue;

        for (uint32_t channel = 0; channel < 3; ++channel)
        {
          float value = std::clamp(colors[layer][channel] * colorScale, 0.0f, 1.0f);
          albedo[pixel * 4 + channel] = static_cast<uint8_t>(value * 255.0f);
        }
        albedo[pixel * 4 + 3] = 255;
        rough[pixel] = static_cast<uint8_t>(
          std::clamp(roughness[layer] + (noiseValue - 0.5f) * 0.12f, 0.0f, 1.0f) * 255.0f);
        height[pixel] = static_cast<uint8_t>(
          std::clamp(0.65f * noiseValue + 0.35f * wave, 0.0f, 1.0f) * 255.0f);
      }
    }

    transfer.uploadImage(*commands, albedoMaps, 0, layer, asBytes(albedo));
    transfer.uploadImage(*commands, roughnessMaps, 0, layer, asBytes(rough));
    transfer.uploadImage(*commands, detailHeightMaps, 0, layer, asBytes(height));
  }
}

void WorldRenderer::loadShaders()
{
  etna::create_program("terrain_height", {TERRAIN_SHADERS_ROOT "heightmap.comp.spv"});
  etna::create_program(
    "terrain_surface",
    {TERRAIN_SHADERS_ROOT "terrain.vert.spv", TERRAIN_SHADERS_ROOT "terrain.frag.spv"});
}

void WorldRenderer::setupPipelines(vk::Format swapchainFormat)
{
  auto& pipelineManager = etna::get_context().getPipelineManager();
  heightPipeline = pipelineManager.createComputePipeline("terrain_height", {});

  etna::VertexShaderInputDescription vertexInput{
    .bindings = {etna::VertexShaderInputDescription::Binding{
      .byteStreamDescription = etna::VertexByteStreamFormatDescription{
        .stride = sizeof(Vertex),
        .attributes = {
          {.format = vk::Format::eR32G32Sfloat, .offset = 0},
          {.format = vk::Format::eR32G32Sfloat, .offset = sizeof(glm::vec2)},
        },
      },
    }},
  };

  terrainPipeline = pipelineManager.createGraphicsPipeline(
    "terrain_surface",
    etna::GraphicsPipeline::CreateInfo{
      .vertexShaderInput = vertexInput,
      .rasterizationConfig =
        vk::PipelineRasterizationStateCreateInfo{
          .polygonMode = vk::PolygonMode::eFill,
          .cullMode = vk::CullModeFlagBits::eNone,
          .frontFace = vk::FrontFace::eCounterClockwise,
          .lineWidth = 1.0f,
        },
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {swapchainFormat},
          .depthAttachmentFormat = vk::Format::eD32Sfloat,
        },
    });
}

void WorldRenderer::update(const FramePacket& packet)
{
  float aspect = static_cast<float>(resolution.x) / static_cast<float>(resolution.y);
  uniforms.projView = packet.camera.projTm(aspect) * packet.camera.viewTm();
  uniforms.cameraPosition = glm::vec4(packet.camera.position, 1.0f);
}

void WorldRenderer::drawGui()
{
  ImGui::Begin("Terrain");
  ImGui::SliderFloat("Frequency", &noise.frequency, 0.5f, 8.0f);
  ImGui::SliderInt("Octaves", &noise.octaves, 1, 10);
  ImGui::SliderFloat("Lacunarity", &noise.lacunarity, 1.2f, 3.5f);
  ImGui::SliderFloat("Gain", &noise.gain, 0.2f, 0.8f);
  ImGui::SliderInt("Seed", &noise.seed, 0, 1000);
  ImGui::SliderFloat("Island radius", &noise.radius, 0.2f, 0.9f);
  ImGui::SliderFloat("Island falloff", &noise.falloff, 0.05f, 0.8f);
  ImGui::SliderFloat("Sea level", &noise.seaLevel, 0.0f, 0.6f);

  if (ImGui::Button("Regenerate"))
    regenerate = true;

  ImGui::Separator();
  ImGui::SliderFloat("Height", &uniforms.terrainParams.y, 2.0f, 30.0f);
  ImGui::SliderFloat("Texture scale", &uniforms.terrainParams.z, 0.03f, 0.5f);
  ImGui::SliderFloat("Detail strength", &uniforms.terrainParams.w, 0.0f, 4.0f);
  ImGui::SliderFloat("Sand level", &uniforms.materialLevels.x, 0.05f, 0.4f);
  ImGui::SliderFloat("Rock level", &uniforms.materialLevels.y, 0.25f, 0.7f);
  ImGui::SliderFloat("Snow level", &uniforms.materialLevels.z, 0.5f, 0.95f);
  ImGui::SliderFloat("Blend", &uniforms.materialLevels.w, 0.01f, 0.15f);

  uniforms.materialLevels.y = std::max(
    uniforms.materialLevels.y, uniforms.materialLevels.x + 0.03f);
  uniforms.materialLevels.z = std::max(
    uniforms.materialLevels.z, uniforms.materialLevels.y + 0.03f);

  ImGui::Separator();
  ImGui::SliderFloat3("Light direction", &uniforms.lightDirectionIntensity.x, -1.0f, 1.0f);
  ImGui::SliderFloat("Light intensity", &uniforms.lightDirectionIntensity.w, 0.1f, 8.0f);
  ImGui::ColorEdit3("Light color", &uniforms.lightColorAmbient.x);
  ImGui::SliderFloat("Ambient", &uniforms.lightColorAmbient.w, 0.0f, 0.4f);
  ImGui::Text("WASD, Q/E, Shift, right mouse");
  ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
  ImGui::End();
}

void WorldRenderer::render(
  vk::CommandBuffer commandBuffer, vk::Image target, vk::ImageView targetView)
{
  if (regenerate)
  {
    const auto program = etna::get_shader_program("terrain_height");
    auto set = etna::create_descriptor_set(
      program.getDescriptorLayoutId(0),
      commandBuffer,
      {etna::Binding{
        0,
        heightMap.genBinding(vk::Sampler{}, vk::ImageLayout::eGeneral)}});

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, heightPipeline.getVkPipeline());
    commandBuffer.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,
      heightPipeline.getVkPipelineLayout(),
      0,
      {set.getVkSet()},
      {});
    commandBuffer.pushConstants(
      heightPipeline.getVkPipelineLayout(),
      vk::ShaderStageFlagBits::eCompute,
      0,
      sizeof(NoiseParams),
      &noise);
    etna::flush_barriers(commandBuffer);
    commandBuffer.dispatch(HeightMapSize / 16, HeightMapSize / 16, 1);
    regenerate = false;
  }

  std::memcpy(uniformBuffer.data(), &uniforms, sizeof(Uniforms));

  const auto program = etna::get_shader_program("terrain_surface");
  etna::Image::ViewParams arrayView{.type = vk::ImageViewType::e2DArray};
  auto set = etna::create_descriptor_set(
    program.getDescriptorLayoutId(0),
    commandBuffer,
    {
      etna::Binding{0, uniformBuffer.genBinding()},
      etna::Binding{
        1,
        heightMap.genBinding(heightSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
      etna::Binding{
        2,
        albedoMaps.genBinding(
          materialSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, arrayView)},
      etna::Binding{
        3,
        roughnessMaps.genBinding(
          materialSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, arrayView)},
      etna::Binding{
        4,
        detailHeightMaps.genBinding(
          materialSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal, arrayView)},
    });

  etna::RenderTargetState renderTargets(
    commandBuffer,
    {{0, 0}, {resolution.x, resolution.y}},
    {{
      .image = target,
      .view = targetView,
      .clearColorValue = std::array<float, 4>{0.42f, 0.61f, 0.78f, 1.0f},
    }},
    {.image = depthImage.get(), .view = depthImage.getView({})});

  commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, terrainPipeline.getVkPipeline());
  commandBuffer.bindDescriptorSets(
    vk::PipelineBindPoint::eGraphics,
    terrainPipeline.getVkPipelineLayout(),
    0,
    {set.getVkSet()},
    {});
  commandBuffer.bindVertexBuffers(0, {vertexBuffer.get()}, {0});
  commandBuffer.bindIndexBuffer(indexBuffer.get(), 0, vk::IndexType::eUint32);
  commandBuffer.drawIndexed(indexCount, 1, 0, 0, 0);
}
