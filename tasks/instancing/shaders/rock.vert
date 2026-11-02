#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv;

layout(set = 0, binding = 0) uniform Scene
{
  mat4 projView;
  vec4 cameraPosition;
  vec4 lightDirectionIntensity;
  vec4 lightColorAmbient;
  vec4 terrain;
  vec4 materialLevels;
} scene;

struct InstanceData
{
  mat4 transform;
  vec4 color;
};

layout(std430, set = 0, binding = 4) readonly buffer Instances
{
  InstanceData data[];
} instances;

layout(location = 0) out vec3 outWorldPosition;
layout(location = 1) out vec3 outWorldNormal;
layout(location = 2) out vec2 outUv;
layout(location = 3) out vec3 outColorVariation;

void main()
{
  InstanceData instance = instances.data[gl_InstanceIndex];
  vec4 worldPosition = instance.transform * vec4(inPosition, 1.0);

  outWorldPosition = worldPosition.xyz;
  outWorldNormal = normalize(mat3(instance.transform) * inNormal);
  outUv = inUv;
  outColorVariation = instance.color.rgb;
  gl_Position = scene.projView * worldPosition;
}
