#version 450

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inUv;

layout(set = 0, binding = 0) uniform Scene
{
  mat4 projView;
  vec4 cameraPosition;
  vec4 lightDirectionIntensity;
  vec4 lightColorAmbient;
  vec4 terrain;
  vec4 materialLevels;
} scene;

layout(set = 0, binding = 1) uniform sampler2D heightMap;

layout(location = 0) out vec2 outUv;
layout(location = 1) out vec3 outWorldPosition;
layout(location = 2) out float outHeight;

void main()
{
  float height = textureLod(heightMap, inUv, 0.0).r;
  vec3 worldPosition = vec3(
    inPosition.x * scene.terrain.x,
    height * scene.terrain.y,
    inPosition.y * scene.terrain.x);

  outUv = inUv;
  outWorldPosition = worldPosition;
  outHeight = height;
  gl_Position = scene.projView * vec4(worldPosition, 1.0);
}
