#include "WorldRenderer.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <imgui.h>
#include <tiny_gltf.h>

#include <etna/BlockingTransferHelper.hpp>
#include <etna/Etna.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/RenderTargetStates.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <spdlog/spdlog.h>


namespace
{
constexpr uint32_t GridCells = 256;
constexpr uint32_t HeightMapSize = 1024;
constexpr uint32_t MaterialSize = 64;
constexpr uint32_t MaterialCount = 4;
constexpr uint32_t MaxRockInstances = 4096;
constexpr float RockCellSize = 2.2f;

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

uint32_t hashBits(int x, int y, uint32_t salt)
{
  uint32_t value = static_cast<uint32_t>(x) * 374761393u;
  value += static_cast<uint32_t>(y) * 668265263u + salt * 2246822519u;
  value = (value ^ (value >> 13u)) * 1274126177u;
  return value ^ (value >> 16u);
}

float hash01(int x, int y, uint32_t salt)
{
  return static_cast<float>(hashBits(x, y, salt) & 0x00ffffffu) /
    static_cast<float>(0x01000000u);
}

float fract(float value)
{
  return value - std::floor(value);
}

float smoothStep(float edge0, float edge1, float value)
{
  float t = std::clamp((value - edge0) / (edge1 - edge0), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

std::vector<uint8_t> rgbaPixels(const tinygltf::Image& image)
{
  if (image.bits != 8 || image.pixel_type != TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
    throw std::runtime_error("Only 8-bit rock textures are supported");

  const auto pixelCount = static_cast<size_t>(image.width) * static_cast<size_t>(image.height);
  std::vector<uint8_t> result(pixelCount * 4);

  for (size_t pixel = 0; pixel < pixelCount; ++pixel)
  {
    const size_t source = pixel * static_cast<size_t>(image.component);
    const size_t target = pixel * 4;
    result[target + 0] = image.image[source + 0];
    result[target + 1] = image.component > 1 ? image.image[source + 1] : image.image[source + 0];
    result[target + 2] = image.component > 2 ? image.image[source + 2] : image.image[source + 0];
    result[target + 3] = image.component > 3 ? image.image[source + 3] : 255;
  }

  return result;
}

const std::byte* accessorData(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
  const auto& view = model.bufferViews.at(static_cast<size_t>(accessor.bufferView));
  const auto& buffer = model.buffers.at(static_cast<size_t>(view.buffer));
  return reinterpret_cast<const std::byte*>(buffer.data.data()) + view.byteOffset +
    accessor.byteOffset;
}

size_t accessorStride(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
  const auto& view = model.bufferViews.at(static_cast<size_t>(accessor.bufferView));
  const int stride = accessor.ByteStride(view);
  if (stride <= 0)
    throw std::runtime_error("Invalid glTF accessor stride");
  return static_cast<size_t>(stride);
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
  uniformBuffers.emplace(context.getMainWorkCount(), [&context](std::size_t) {
    return context.createBuffer(etna::Buffer::CreateInfo{
      .size = sizeof(Uniforms),
      .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer,
      .memoryUsage = VMA_MEMORY_USAGE_AUTO,
      .allocationCreate =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT,
      .name = "terrain_uniforms",
    });
  });

  createGrid();
  createMaterialTextures();
  createRockResources();
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

void WorldRenderer::createRockResources()
{
  auto& context = etna::get_context();
  rockSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eLinear,
    .addressMode = vk::SamplerAddressMode::eRepeat,
    .name = "rock_sampler",
  });

  for (uint32_t modelIndex = 0; modelIndex < rockInstanceBuffers.size(); ++modelIndex)
  {
    rockInstanceBuffers[modelIndex].emplace(
      context.getMainWorkCount(),
      [&context, modelIndex](std::size_t) {
        return context.createBuffer(etna::Buffer::CreateInfo{
          .size = MaxRockInstances * sizeof(RockInstance),
          .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
          .memoryUsage = VMA_MEMORY_USAGE_AUTO,
          .allocationCreate =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT,
          .name = modelIndex == 0 ? "boulder_instances" : "rock_instances",
        });
      });
  }

  loadRockModel(
    0,
    GRAPHICS_COURSE_RESOURCES_ROOT "/scenes/rocks/boulder_01/boulder_01_1k.gltf");
  loadRockModel(1, GRAPHICS_COURSE_RESOURCES_ROOT "/scenes/rocks/rock_09/rock_09_1k.gltf");
}

void WorldRenderer::loadRockModel(uint32_t modelIndex, const std::filesystem::path& path)
{
  tinygltf::TinyGLTF loader;
  tinygltf::Model gltf;
  std::string error;
  std::string warning;

  if (!loader.LoadASCIIFromFile(&gltf, &error, &warning, path.string()))
    throw std::runtime_error("Failed to load " + path.string() + ": " + error);
  if (!warning.empty())
    spdlog::warn("glTF {}: {}", path.string(), warning);
  if (gltf.meshes.empty() || gltf.meshes.front().primitives.empty())
    throw std::runtime_error("Rock glTF has no mesh: " + path.string());

  const auto& primitive = gltf.meshes.front().primitives.front();
  const auto positionIt = primitive.attributes.find("POSITION");
  const auto normalIt = primitive.attributes.find("NORMAL");
  const auto uvIt = primitive.attributes.find("TEXCOORD_0");
  if (
    positionIt == primitive.attributes.end() || normalIt == primitive.attributes.end() ||
    uvIt == primitive.attributes.end() || primitive.indices < 0)
    throw std::runtime_error("Rock glTF is missing required vertex data: " + path.string());

  const auto& positionAccessor = gltf.accessors.at(static_cast<size_t>(positionIt->second));
  const auto& normalAccessor = gltf.accessors.at(static_cast<size_t>(normalIt->second));
  const auto& uvAccessor = gltf.accessors.at(static_cast<size_t>(uvIt->second));
  const auto& indexAccessor = gltf.accessors.at(static_cast<size_t>(primitive.indices));
  if (
    positionAccessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT ||
    normalAccessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT ||
    uvAccessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT)
    throw std::runtime_error("Rock glTF uses an unsupported vertex format: " + path.string());

  const std::byte* positions = accessorData(gltf, positionAccessor);
  const std::byte* normals = accessorData(gltf, normalAccessor);
  const std::byte* uvs = accessorData(gltf, uvAccessor);
  const size_t positionStride = accessorStride(gltf, positionAccessor);
  const size_t normalStride = accessorStride(gltf, normalAccessor);
  const size_t uvStride = accessorStride(gltf, uvAccessor);

  std::vector<RockVertex> vertices(positionAccessor.count);
  float minY = std::numeric_limits<float>::max();
  float maxY = std::numeric_limits<float>::lowest();
  for (size_t vertexIndex = 0; vertexIndex < vertices.size(); ++vertexIndex)
  {
    std::memcpy(&vertices[vertexIndex].position, positions, sizeof(glm::vec3));
    std::memcpy(&vertices[vertexIndex].normal, normals, sizeof(glm::vec3));
    std::memcpy(&vertices[vertexIndex].uv, uvs, sizeof(glm::vec2));
    minY = std::min(minY, vertices[vertexIndex].position.y);
    maxY = std::max(maxY, vertices[vertexIndex].position.y);
    positions += positionStride;
    normals += normalStride;
    uvs += uvStride;
  }

  const std::byte* sourceIndices = accessorData(gltf, indexAccessor);
  const size_t indexStride = accessorStride(gltf, indexAccessor);
  std::vector<uint32_t> indices(indexAccessor.count);
  for (size_t index = 0; index < indices.size(); ++index)
  {
    if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
    {
      uint16_t value = 0;
      std::memcpy(&value, sourceIndices, sizeof(value));
      indices[index] = value;
    }
    else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
    {
      std::memcpy(&indices[index], sourceIndices, sizeof(uint32_t));
    }
    else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
    {
      uint8_t value = 0;
      std::memcpy(&value, sourceIndices, sizeof(value));
      indices[index] = value;
    }
    else
    {
      throw std::runtime_error("Rock glTF uses an unsupported index format: " + path.string());
    }
    sourceIndices += indexStride;
  }

  RockModel& model = rockModels.at(modelIndex);
  auto& context = etna::get_context();
  model.vertexBuffer = context.createBuffer(etna::Buffer::CreateInfo{
    .size = vertices.size() * sizeof(RockVertex),
    .bufferUsage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    .name = modelIndex == 0 ? "boulder_vertices" : "rock_vertices",
  });
  model.indexBuffer = context.createBuffer(etna::Buffer::CreateInfo{
    .size = indices.size() * sizeof(uint32_t),
    .bufferUsage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    .name = modelIndex == 0 ? "boulder_indices" : "rock_indices",
  });

  auto commands = context.createOneShotCmdMgr();
  etna::BlockingTransferHelper transfer({.stagingSize = 4 * 1024 * 1024});
  transfer.uploadBuffer<RockVertex>(*commands, model.vertexBuffer, 0, vertices);
  transfer.uploadBuffer<uint32_t>(*commands, model.indexBuffer, 0, indices);

  if (gltf.materials.empty())
    throw std::runtime_error("Rock glTF has no material: " + path.string());
  const auto& material = gltf.materials.front();
  const int albedoTexture = material.pbrMetallicRoughness.baseColorTexture.index;
  const int normalTexture = material.normalTexture.index;
  const int armTexture = material.pbrMetallicRoughness.metallicRoughnessTexture.index;
  if (albedoTexture < 0 || normalTexture < 0 || armTexture < 0)
    throw std::runtime_error("Rock glTF has incomplete PBR textures: " + path.string());

  auto imageForTexture = [&gltf](int textureIndex) -> const tinygltf::Image& {
    const auto& texture = gltf.textures.at(static_cast<size_t>(textureIndex));
    return gltf.images.at(static_cast<size_t>(texture.source));
  };
  auto uploadTexture = [&context, &commands, &transfer](
                         const tinygltf::Image& image,
                         vk::Format format,
                         const char* name) {
    std::vector<uint8_t> pixels = rgbaPixels(image);
    etna::Image texture = context.createImage(etna::Image::CreateInfo{
      .extent = vk::Extent3D{
        static_cast<uint32_t>(image.width), static_cast<uint32_t>(image.height), 1},
      .name = name,
      .format = format,
      .imageUsage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
    });
    transfer.uploadImage(*commands, texture, 0, 0, asBytes(pixels));
    return texture;
  };

  model.albedo = uploadTexture(
    imageForTexture(albedoTexture),
    vk::Format::eR8G8B8A8Srgb,
    modelIndex == 0 ? "boulder_albedo" : "rock_albedo");
  model.normal = uploadTexture(
    imageForTexture(normalTexture),
    vk::Format::eR8G8B8A8Unorm,
    modelIndex == 0 ? "boulder_normal" : "rock_normal");
  model.arm = uploadTexture(
    imageForTexture(armTexture),
    vk::Format::eR8G8B8A8Unorm,
    modelIndex == 0 ? "boulder_arm" : "rock_arm");
  model.indexCount = static_cast<uint32_t>(indices.size());
  model.minY = minY;
  model.normalizationScale = 1.0f / std::max(maxY - minY, 0.001f);
}

void WorldRenderer::loadShaders()
{
  etna::create_program("terrain_height", {INSTANCING_SHADERS_ROOT "heightmap.comp.spv"});
  etna::create_program(
    "terrain_surface",
    {INSTANCING_SHADERS_ROOT "terrain.vert.spv", INSTANCING_SHADERS_ROOT "terrain.frag.spv"});
  etna::create_program(
    "rock_surface",
    {INSTANCING_SHADERS_ROOT "rock.vert.spv", INSTANCING_SHADERS_ROOT "rock.frag.spv"});
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

  etna::VertexShaderInputDescription rockVertexInput{
    .bindings = {etna::VertexShaderInputDescription::Binding{
      .byteStreamDescription = etna::VertexByteStreamFormatDescription{
        .stride = sizeof(RockVertex),
        .attributes = {
          {.format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(RockVertex, position)},
          {.format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(RockVertex, normal)},
          {.format = vk::Format::eR32G32Sfloat, .offset = offsetof(RockVertex, uv)},
        },
      },
    }},
  };

  rockPipeline = pipelineManager.createGraphicsPipeline(
    "rock_surface",
    etna::GraphicsPipeline::CreateInfo{
      .vertexShaderInput = rockVertexInput,
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
  updateVisibleRocks();
}

float WorldRenderer::sampleTerrainHeight(float worldX, float worldZ) const
{
  const glm::vec2 point{
    2.0f * worldX / uniforms.terrainParams.x,
    2.0f * worldZ / uniforms.terrainParams.x,
  };

  auto hash = [this](glm::vec2 position) {
    position.x = fract(position.x * 123.34f);
    position.y = fract(position.y * 456.21f);
    const float offset = glm::dot(
      position,
      position + glm::vec2(45.32f + static_cast<float>(noise.seed)));
    position += glm::vec2(offset);
    return fract(position.x * position.y);
  };
  auto valueNoise = [&hash](glm::vec2 position) {
    const glm::vec2 cell = glm::floor(position);
    const glm::vec2 local = position - cell;
    const glm::vec2 blend = local * local * (3.0f - 2.0f * local);
    const float a = hash(cell);
    const float b = hash(cell + glm::vec2(1.0f, 0.0f));
    const float c = hash(cell + glm::vec2(0.0f, 1.0f));
    const float d = hash(cell + glm::vec2(1.0f, 1.0f));
    return glm::mix(glm::mix(a, b, blend.x), glm::mix(c, d, blend.x), blend.y);
  };

  float value = 0.0f;
  float weight = 1.0f;
  float weightSum = 0.0f;
  float frequency = noise.frequency;
  for (int octave = 0; octave < noise.octaves; ++octave)
  {
    value += valueNoise(point * frequency) * weight;
    weightSum += weight;
    frequency *= noise.lacunarity;
    weight *= noise.gain;
  }
  value /= std::max(weightSum, 0.0001f);

  const float radial = smoothStep(noise.radius, noise.radius + noise.falloff, glm::length(point));
  value = value * noise.amplitude - radial;
  return std::clamp(
    (value - noise.seaLevel) / std::max(1.0f - noise.seaLevel, 0.001f),
    0.0f,
    1.0f);
}

bool WorldRenderer::isInsideFrustum(const glm::vec3& position) const
{
  const glm::vec4 clip = uniforms.projView * glm::vec4(position, 1.0f);
  if (clip.w <= 0.0f)
    return false;

  const glm::vec3 ndc = glm::vec3(clip) / clip.w;
  constexpr float margin = 0.08f;
  return ndc.x >= -1.0f - margin && ndc.x <= 1.0f + margin &&
    ndc.y >= -1.0f - margin && ndc.y <= 1.0f + margin &&
    ndc.z >= -margin && ndc.z <= 1.0f + margin;
}

void WorldRenderer::updateVisibleRocks()
{
  for (auto& instances : visibleRocks)
  {
    instances.clear();
    instances.reserve(512);
  }

  const glm::vec3 camera = glm::vec3(uniforms.cameraPosition);
  const float terrainHalfSize = uniforms.terrainParams.x * 0.5f;
  const int terrainMinCell = static_cast<int>(std::floor(-terrainHalfSize / RockCellSize));
  const int terrainMaxCell = static_cast<int>(std::ceil(terrainHalfSize / RockCellSize));
  const int minCellX = std::max(
    terrainMinCell,
    static_cast<int>(std::floor((camera.x - rockSettings.drawDistance) / RockCellSize)));
  const int maxCellX = std::min(
    terrainMaxCell,
    static_cast<int>(std::ceil((camera.x + rockSettings.drawDistance) / RockCellSize)));
  const int minCellZ = std::max(
    terrainMinCell,
    static_cast<int>(std::floor((camera.z - rockSettings.drawDistance) / RockCellSize)));
  const int maxCellZ = std::min(
    terrainMaxCell,
    static_cast<int>(std::ceil((camera.z + rockSettings.drawDistance) / RockCellSize)));

  for (int cellZ = minCellZ; cellZ <= maxCellZ; ++cellZ)
  {
    for (int cellX = minCellX; cellX <= maxCellX; ++cellX)
    {
      const float worldX = (static_cast<float>(cellX) + hash01(cellX, cellZ, 0)) * RockCellSize;
      const float worldZ = (static_cast<float>(cellZ) + hash01(cellX, cellZ, 1)) * RockCellSize;
      if (
        worldX < -terrainHalfSize || worldX > terrainHalfSize || worldZ < -terrainHalfSize ||
        worldZ > terrainHalfSize)
        continue;

      const float height = sampleTerrainHeight(worldX, worldZ);
      const float aboveBeach = smoothStep(0.08f, 0.28f, height);
      const float belowSnow = 1.0f - smoothStep(0.72f, 0.92f, height);
      const float placementChance = rockSettings.density * aboveBeach * belowSnow;
      if (hash01(cellX, cellZ, 2) > placementChance)
        continue;

      const glm::vec2 cameraOffset{worldX - camera.x, worldZ - camera.z};
      if (glm::length(cameraOffset) > rockSettings.drawDistance)
        continue;

      const uint32_t modelIndex = hash01(cellX, cellZ, 3) < 0.5f ? 0u : 1u;
      auto& instances = visibleRocks[modelIndex];
      if (instances.size() >= MaxRockInstances)
        continue;

      const float randomScale = glm::mix(
        rockSettings.minScale,
        rockSettings.maxScale,
        hash01(cellX, cellZ, 4));
      const RockModel& model = rockModels[modelIndex];
      const float modelScale = randomScale * model.normalizationScale;
      const float groundY = height * uniforms.terrainParams.y;
      const glm::vec3 center{worldX, groundY + randomScale * 0.5f, worldZ};
      if (!isInsideFrustum(center))
        continue;

      glm::mat4 transform = glm::translate(
        glm::mat4(1.0f),
        glm::vec3(worldX, groundY - model.minY * modelScale, worldZ));
      transform = glm::rotate(
        transform,
        hash01(cellX, cellZ, 5) * 6.28318530718f,
        glm::vec3(0.0f, 1.0f, 0.0f));
      transform = glm::scale(transform, glm::vec3(modelScale));

      const float tint = 0.86f + 0.25f * hash01(cellX, cellZ, 6);
      instances.push_back(RockInstance{
        .transform = transform,
        .color = glm::vec4(tint, tint * 0.98f, tint * 0.94f, 1.0f),
      });
    }
  }
}

void WorldRenderer::drawGui()
{
  ImGui::Begin("Instanced rocks");
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
  ImGui::Text("Instanced rocks");
  ImGui::SliderFloat("Rock density", &rockSettings.density, 0.0f, 1.0f);
  ImGui::SliderFloat("Rock draw distance", &rockSettings.drawDistance, 5.0f, 80.0f);
  ImGui::SliderFloat("Minimum rock size", &rockSettings.minScale, 0.2f, 2.5f);
  ImGui::SliderFloat("Maximum rock size", &rockSettings.maxScale, 0.3f, 4.0f);
  rockSettings.maxScale = std::max(rockSettings.maxScale, rockSettings.minScale);
  ImGui::Text(
    "Visible: %zu boulders, %zu rocks",
    visibleRocks[0].size(),
    visibleRocks[1].size());

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

  etna::Buffer& currentUniformBuffer = uniformBuffers->get();
  std::memcpy(currentUniformBuffer.data(), &uniforms, sizeof(Uniforms));

  const auto program = etna::get_shader_program("terrain_surface");
  etna::Image::ViewParams arrayView{.type = vk::ImageViewType::e2DArray};
  auto set = etna::create_descriptor_set(
    program.getDescriptorLayoutId(0),
    commandBuffer,
    {
      etna::Binding{0, currentUniformBuffer.genBinding()},
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
  renderRocks(commandBuffer, currentUniformBuffer);
}

void WorldRenderer::renderRocks(
  vk::CommandBuffer commandBuffer, const etna::Buffer& currentUniformBuffer)
{
  const auto program = etna::get_shader_program("rock_surface");
  commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, rockPipeline.getVkPipeline());

  for (uint32_t modelIndex = 0; modelIndex < rockModels.size(); ++modelIndex)
  {
    const auto& instances = visibleRocks[modelIndex];
    if (instances.empty())
      continue;

    etna::Buffer& instanceBuffer = rockInstanceBuffers[modelIndex]->get();
    std::memcpy(
      instanceBuffer.data(),
      instances.data(),
      instances.size() * sizeof(RockInstance));

    RockModel& model = rockModels[modelIndex];
    auto set = etna::create_descriptor_set(
      program.getDescriptorLayoutId(0),
      commandBuffer,
      {
        etna::Binding{0, currentUniformBuffer.genBinding()},
        etna::Binding{
          1,
          model.albedo.genBinding(rockSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
        etna::Binding{
          2,
          model.normal.genBinding(rockSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
        etna::Binding{
          3,
          model.arm.genBinding(rockSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
        etna::Binding{4, instanceBuffer.genBinding()},
      });

    commandBuffer.bindDescriptorSets(
      vk::PipelineBindPoint::eGraphics,
      rockPipeline.getVkPipelineLayout(),
      0,
      {set.getVkSet()},
      {});
    etna::flush_barriers(commandBuffer);
    commandBuffer.bindVertexBuffers(0, {model.vertexBuffer.get()}, {0});
    commandBuffer.bindIndexBuffer(model.indexBuffer.get(), 0, vk::IndexType::eUint32);
    commandBuffer.drawIndexed(
      model.indexCount,
      static_cast<uint32_t>(instances.size()),
      0,
      0,
      0);
  }
}
