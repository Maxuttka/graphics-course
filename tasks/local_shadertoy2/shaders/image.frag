#version 450

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D iChannel0;
layout(set = 0, binding = 1) uniform sampler2D iChannel1;

layout(push_constant) uniform Params
{
  vec2 resolution;
  float time;
  float padding;
} params;

const float PI = 3.14159265;


vec3 getSpherePos()
{
  return vec3(0.0, 1.0 + 0.7 * sin(params.time), 0.0);
}

float sdSphere(vec3 p, float r)
{
  return length(p) - r;
}

float sdPlane(vec3 p)
{
  return p.y;
}

float mapScene(vec3 p)
{
  float sphereDistance = sdSphere(p - getSpherePos(), 1.0);
  float planeDistance = sdPlane(p);
  return min(sphereDistance, planeDistance);
}

vec3 getNormal(vec3 p)
{
  const float e = 0.001;
  vec3 n = vec3(
    mapScene(p + vec3(e, 0.0, 0.0)) - mapScene(p - vec3(e, 0.0, 0.0)),
    mapScene(p + vec3(0.0, e, 0.0)) - mapScene(p - vec3(0.0, e, 0.0)),
    mapScene(p + vec3(0.0, 0.0, e)) - mapScene(p - vec3(0.0, 0.0, e)));
  return normalize(n);
}

float rayMarch(vec3 rayOrigin, vec3 rayDirection)
{
  float totalDistance = 0.0;

  for (int i = 0; i < 100; ++i)
  {
    vec3 p = rayOrigin + rayDirection * totalDistance;
    float sceneDistance = mapScene(p);

    if (abs(sceneDistance) < 0.001)
      return totalDistance;

    totalDistance += sceneDistance;
    if (totalDistance > 100.0)
      break;
  }

  return 100.0;
}

float getHeight(vec3 p)
{
  return
    sin(p.x * 8.0 + params.time) *
    sin(p.z * 8.0 + params.time * 0.7);
}

vec3 bumpNormal(vec3 p, vec3 n)
{
  const float e = 0.01;
  const float strength = 0.12;
  float height = getHeight(p);
  vec3 gradient = vec3(
    getHeight(p + vec3(e, 0.0, 0.0)) - height,
    0.0,
    getHeight(p + vec3(0.0, 0.0, e)) - height) / e;
  gradient -= n * dot(gradient, n);

  return normalize(n - strength * gradient);
}

vec2 getUV(vec3 p)
{
  vec3 localPosition = p - getSpherePos();
  float sphereDistance = sdSphere(localPosition, 1.0);
  float planeDistance = sdPlane(p);

  if (sphereDistance < planeDistance)
  {
    vec3 q = normalize(localPosition);
    return vec2(
      atan(q.z, q.x) / (2.0 * PI) + 0.5,
      asin(clamp(q.y, -1.0, 1.0)) / PI + 0.5);
  }

  return fract(p.xz * 0.2);
}

float distributionGGX(float NdotH, float roughness)
{
  float a = roughness * roughness;
  float a2 = a * a;
  float denominator = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / max(PI * denominator * denominator, 0.000001);
}

float geometrySchlick(float NdotX, float roughness)
{
  float r = roughness + 1.0;
  float k = r * r / 8.0;
  return NdotX / (NdotX * (1.0 - k) + k);
}

vec3 fresnelSchlick(float VdotH, vec3 F0)
{
  float factor = pow(1.0 - clamp(VdotH, 0.0, 1.0), 5.0);
  return F0 + (vec3(1.0) - F0) * factor;
}

vec3 lighting(vec3 p, vec3 rayDirection)
{
  vec3 n = bumpNormal(p, getNormal(p));
  vec3 v = normalize(-rayDirection);
  vec3 l = normalize(vec3(1.0, 1.0, 2.0));
  vec3 h = normalize(v + l);

  vec2 uv = getUV(p);
  vec3 baseColor = texture(iChannel0, uv).rgb;
  baseColor = pow(baseColor, vec3(2.2));

  vec2 material = texture(iChannel1, uv).rg;
  float roughness = clamp(material.r, 0.2, 1.0);
  float metalness = clamp(material.g, 0.0, 1.0);

  float NdotL = max(dot(n, l), 0.0);
  float NdotV = max(dot(n, v), 0.0);
  float NdotH = max(dot(n, h), 0.0);
  float VdotH = max(dot(v, h), 0.0);
  vec3 F0 = mix(vec3(0.04), baseColor, metalness);

  vec3 F = fresnelSchlick(VdotH, F0);
  float D = distributionGGX(NdotH, roughness);
  float G =
    geometrySchlick(NdotV, roughness) *
    geometrySchlick(NdotL, roughness);

  vec3 specular = D * G * F / max(4.0 * NdotV * NdotL, 0.001);
  vec3 kD = (vec3(1.0) - F) * (1.0 - metalness);
  vec3 diffuse = kD * baseColor / PI;

  vec3 lightColor = vec3(4.0);
  vec3 directLight = (diffuse + specular) * lightColor * NdotL;
  vec3 ambient = 0.05 * baseColor * (1.0 - metalness);
  return ambient + directLight;
}

void main()
{
  vec2 fragCoord = gl_FragCoord.xy;
  fragCoord.y = params.resolution.y - fragCoord.y;

  vec2 uv = (fragCoord - 0.5 * params.resolution) / params.resolution.y;
  vec3 rayOrigin = vec3(0.0, 1.5, 5.0);
  vec3 rayDirection = normalize(vec3(uv, -1.0));
  float distance = rayMarch(rayOrigin, rayDirection);
  vec3 color;

  if (distance < 100.0)
  {
    vec3 p = rayOrigin + rayDirection * distance;
    color = lighting(p, rayDirection);
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));
  }
  else
  {
    color = vec3(0.5, 0.7, 1.0);
  }

  outColor = vec4(color, 1.0);
}
