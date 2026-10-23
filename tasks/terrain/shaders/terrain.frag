#version 450

layout(location = 0) in vec2 inUv;
layout(location = 1) in vec3 inWorldPosition;
layout(location = 2) in float inHeight;

layout(location = 0) out vec4 outColor;

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
layout(set = 0, binding = 2) uniform sampler2DArray albedoMaps;
layout(set = 0, binding = 3) uniform sampler2DArray roughnessMaps;
layout(set = 0, binding = 4) uniform sampler2DArray detailHeightMaps;

const float Pi = 3.14159265359;

vec4 materialWeights(float height)
{
  float blend = scene.materialLevels.w;
  float sandGrass = smoothstep(
    scene.materialLevels.x - blend,
    scene.materialLevels.x + blend,
    height);
  float grassRock = smoothstep(
    scene.materialLevels.y - blend,
    scene.materialLevels.y + blend,
    height);
  float rockSnow = smoothstep(
    scene.materialLevels.z - blend,
    scene.materialLevels.z + blend,
    height);

  vec4 weights = vec4(
    1.0 - sandGrass,
    sandGrass * (1.0 - grassRock),
    grassRock * (1.0 - rockSnow),
    rockSnow);
  return weights / max(dot(weights, vec4(1.0)), 0.0001);
}

vec3 terrainNormal()
{
  ivec2 size = textureSize(heightMap, 0);
  vec2 texel = 1.0 / vec2(size);
  float left = texture(heightMap, inUv - vec2(texel.x, 0.0)).r;
  float right = texture(heightMap, inUv + vec2(texel.x, 0.0)).r;
  float down = texture(heightMap, inUv - vec2(0.0, texel.y)).r;
  float up = texture(heightMap, inUv + vec2(0.0, texel.y)).r;
  float worldTexel = scene.terrain.x / float(size.x - 1);
  float dx = (right - left) * scene.terrain.y;
  float dz = (up - down) * scene.terrain.y;
  return normalize(vec3(-dx / (2.0 * worldTexel), 1.0, -dz / (2.0 * worldTexel)));
}

vec2 detailGradient(vec2 uv, vec4 weights)
{
  vec2 texel = 1.0 / vec2(textureSize(detailHeightMaps, 0).xy);
  vec2 result = vec2(0.0);

  for (int layer = 0; layer < 4; ++layer)
  {
    float left = texture(detailHeightMaps, vec3(uv - vec2(texel.x, 0.0), layer)).r;
    float right = texture(detailHeightMaps, vec3(uv + vec2(texel.x, 0.0), layer)).r;
    float down = texture(detailHeightMaps, vec3(uv - vec2(0.0, texel.y), layer)).r;
    float up = texture(detailHeightMaps, vec3(uv + vec2(0.0, texel.y), layer)).r;
    result += weights[layer] * vec2(right - left, up - down);
  }

  return result;
}

float distributionGgx(vec3 normal, vec3 halfway, float roughness)
{
  float a = roughness * roughness;
  float a2 = a * a;
  float nDotH = max(dot(normal, halfway), 0.0);
  float denominator = nDotH * nDotH * (a2 - 1.0) + 1.0;
  return a2 / max(Pi * denominator * denominator, 0.0001);
}

float geometrySchlick(float nDotV, float roughness)
{
  float value = roughness + 1.0;
  float k = value * value / 8.0;
  return nDotV / max(nDotV * (1.0 - k) + k, 0.0001);
}

vec3 fresnelSchlick(float cosTheta, vec3 baseReflectance)
{
  return baseReflectance + (1.0 - baseReflectance) * pow(1.0 - cosTheta, 5.0);
}

void main()
{
  vec4 weights = materialWeights(inHeight);
  vec2 materialUv = inWorldPosition.xz * scene.terrain.z;
  vec3 albedo = vec3(0.0);
  float roughness = 0.0;

  for (int layer = 0; layer < 4; ++layer)
  {
    albedo += weights[layer] * texture(albedoMaps, vec3(materialUv, layer)).rgb;
    roughness += weights[layer] * texture(roughnessMaps, vec3(materialUv, layer)).r;
  }

  vec3 macroNormal = terrainNormal();
  vec3 tangent = normalize(
    vec3(1.0, 0.0, 0.0) - macroNormal * dot(macroNormal, vec3(1.0, 0.0, 0.0)));
  vec3 bitangent = normalize(cross(tangent, macroNormal));
  vec2 gradient = detailGradient(materialUv, weights) * scene.terrain.w;
  vec3 normal = normalize(macroNormal - tangent * gradient.x - bitangent * gradient.y);

  vec3 viewDirection = normalize(scene.cameraPosition.xyz - inWorldPosition);
  vec3 lightDirection = normalize(-scene.lightDirectionIntensity.xyz + vec3(0.00001));
  vec3 halfway = normalize(viewDirection + lightDirection);
  float nDotL = max(dot(normal, lightDirection), 0.0);
  float nDotV = max(dot(normal, viewDirection), 0.0);
  vec3 reflectance = fresnelSchlick(max(dot(halfway, viewDirection), 0.0), vec3(0.04));
  float distribution = distributionGgx(normal, halfway, max(roughness, 0.05));
  float geometry = geometrySchlick(nDotL, roughness) * geometrySchlick(nDotV, roughness);
  vec3 specular = distribution * geometry * reflectance /
    max(4.0 * nDotV * nDotL, 0.0001);
  vec3 diffuse = (vec3(1.0) - reflectance) * albedo / Pi;
  vec3 radiance = scene.lightColorAmbient.rgb * scene.lightDirectionIntensity.w;
  vec3 color = (diffuse + specular) * radiance * nDotL;
  color += albedo * scene.lightColorAmbient.w;
  color = color / (color + vec3(1.0));
  outColor = vec4(color, 1.0);
}
