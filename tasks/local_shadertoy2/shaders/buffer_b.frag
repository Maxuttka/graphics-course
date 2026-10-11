#version 450

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform Params
{
  vec2 resolution;
  float time;
  float padding;
} params;


void main()
{
  vec2 fragCoord = gl_FragCoord.xy;
  fragCoord.y = params.resolution.y - fragCoord.y;

  vec2 uv = fragCoord / params.resolution;
  const float PI = 3.14159265;

  float pattern =
    sin(uv.x * 2.0 * PI * 4.0) *
    sin(uv.y * 2.0 * PI * 4.0);
  pattern = pattern * 0.5 + 0.5;

  float roughness = mix(0.2, 0.85, pattern);
  float metalness = step(0.0, sin(uv.x * 2.0 * PI * 3.0));

  outColor = vec4(roughness, metalness, 0.0, 1.0);
}
