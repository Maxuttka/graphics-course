#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <vector>

#include <etna/Buffer.hpp>
#include <etna/ComputePipeline.hpp>
#include <etna/GpuSharedResource.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <glm/glm.hpp>

#include "FramePacket.hpp"


class WorldRenderer
{
public:
  void allocateResources(glm::uvec2 resolution);
  void loadShaders();
  void setupPipelines(vk::Format swapchainFormat);
  void update(const FramePacket& packet);
  void drawGui();
  void render(vk::CommandBuffer commandBuffer, vk::Image target, vk::ImageView targetView);

private:
  struct Vertex
  {
    glm::vec2 position;
    glm::vec2 uv;
  };

  struct RockVertex
  {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
  };

  struct RockInstance
  {
    glm::mat4 transform{1.0f};
    glm::vec4 color{1.0f};
  };

  struct RockModel
  {
    etna::Buffer vertexBuffer;
    etna::Buffer indexBuffer;
    etna::Image albedo;
    etna::Image normal;
    etna::Image arm;
    uint32_t indexCount = 0;
    float minY = 0.0f;
    float normalizationScale = 1.0f;
  };

  struct RockSettings
  {
    float density = 0.55f;
    float drawDistance = 45.0f;
    float minScale = 0.65f;
    float maxScale = 1.8f;
  };

  struct NoiseParams
  {
    float frequency = 2.8f;
    float lacunarity = 2.0f;
    float gain = 0.5f;
    float amplitude = 1.0f;
    float radius = 0.58f;
    float falloff = 0.38f;
    float seaLevel = 0.18f;
    int seed = 3;
    int octaves = 6;
  };

  struct Uniforms
  {
    glm::mat4 projView{1.0f};
    glm::vec4 cameraPosition{0.0f};
    glm::vec4 lightDirectionIntensity{-0.4f, -1.0f, -0.3f, 3.0f};
    glm::vec4 lightColorAmbient{1.0f, 0.95f, 0.82f, 0.08f};
    glm::vec4 terrainParams{70.0f, 15.0f, 0.18f, 1.4f};
    glm::vec4 materialLevels{0.18f, 0.43f, 0.72f, 0.055f};
  };

  void initializeTerrain();
  void createGrid();
  void createMaterialTextures();
  void createRockResources();
  void loadRockModel(uint32_t modelIndex, const std::filesystem::path& path);
  void updateVisibleRocks();
  float sampleTerrainHeight(float worldX, float worldZ) const;
  bool isInsideFrustum(const glm::vec3& position) const;
  void renderRocks(vk::CommandBuffer commandBuffer, const etna::Buffer& currentUniformBuffer);

  bool initialized = false;
  bool regenerate = true;
  uint32_t indexCount = 0;
  glm::uvec2 resolution{};
  NoiseParams noise;
  Uniforms uniforms;
  RockSettings rockSettings;
  std::array<RockModel, 2> rockModels;
  std::array<std::vector<RockInstance>, 2> visibleRocks;
  std::array<std::optional<etna::GpuSharedResource<etna::Buffer>>, 2> rockInstanceBuffers;
  std::optional<etna::GpuSharedResource<etna::Buffer>> uniformBuffers;
  etna::Image depthImage;
  etna::Image heightMap;
  etna::Image albedoMaps;
  etna::Image roughnessMaps;
  etna::Image detailHeightMaps;
  etna::Sampler heightSampler;
  etna::Sampler materialSampler;
  etna::Sampler rockSampler;
  etna::Buffer vertexBuffer;
  etna::Buffer indexBuffer;
  etna::ComputePipeline heightPipeline;
  etna::GraphicsPipeline terrainPipeline;
  etna::GraphicsPipeline rockPipeline;
};
