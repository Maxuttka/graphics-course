#version 450

layout(location = 0) in vec3 inWorldPosition;
layout(location = 1) in vec3 inWorldNormal;
layout(location = 2) in vec2 inUv;
layout(location = 3) in vec3 inColorVariation;

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

layout(set = 0, binding = 1) uniform sampler2D albedoMap;
layout(set = 0, binding = 2) uniform sampler2D normalMap;
layout(set = 0, binding = 3) uniform sampler2D armMap;

const float Pi = 3.14159265359;

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

vec3 mappedNormal()
{
  vec3 normal = normalize(inWorldNormal);
  vec3 positionDx = dFdx(inWorldPosition);
  vec3 positionDy = dFdy(inWorldPosition);
  vec2 uvDx = dFdx(inUv);
  vec2 uvDy = dFdy(inUv);
  vec3 tangent = positionDx * uvDy.y - positionDy * uvDx.y;

  if (dot(tangent, tangent) < 0.000001)
    return normal;

  tangent = normalize(tangent - normal * dot(normal, tangent));
  vec3 bitangent = normalize(cross(normal, tangent));
  vec3 textureNormal = texture(normalMap, inUv).xyz * 2.0 - 1.0;
  return normalize(mat3(tangent, bitangent, normal) * textureNormal);
}

void main()
{
  vec3 albedo = texture(albedoMap, inUv).rgb * inColorVariation;
  vec3 arm = texture(armMap, inUv).rgb;
  float ambientOcclusion = arm.r;
  float roughness = clamp(arm.g, 0.08, 1.0);
  float metallic = clamp(arm.b, 0.0, 1.0);
  vec3 normal = mappedNormal();

  vec3 viewDirection = normalize(scene.cameraPosition.xyz - inWorldPosition);
  vec3 lightDirection = normalize(-scene.lightDirectionIntensity.xyz + vec3(0.00001));
  vec3 halfway = normalize(viewDirection + lightDirection);
  float nDotL = max(dot(normal, lightDirection), 0.0);
  float nDotV = max(dot(normal, viewDirection), 0.0);
  vec3 baseReflectance = mix(vec3(0.04), albedo, metallic);
  vec3 reflectance = fresnelSchlick(max(dot(halfway, viewDirection), 0.0), baseReflectance);
  float distribution = distributionGgx(normal, halfway, roughness);
  float geometry = geometrySchlick(nDotL, roughness) * geometrySchlick(nDotV, roughness);
  vec3 specular = distribution * geometry * reflectance /
    max(4.0 * nDotV * nDotL, 0.0001);
  vec3 diffuse = (vec3(1.0) - reflectance) * (1.0 - metallic) * albedo / Pi;
  vec3 radiance = scene.lightColorAmbient.rgb * scene.lightDirectionIntensity.w;
  vec3 color = (diffuse + specular) * radiance * nDotL;
  color += albedo * scene.lightColorAmbient.w * ambientOcclusion;
  color = color / (color + vec3(1.0));
  outColor = vec4(color, 1.0);
}
