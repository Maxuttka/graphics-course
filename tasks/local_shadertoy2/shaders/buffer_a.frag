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
  vec2 p = uv * 6.0;
  p += vec2(params.time * 0.35, sin(params.time * 0.5) * 0.5);

  float pattern =
    sin(p.x * 3.0 + sin(p.y * 2.0 + params.time)) *
    sin(p.y * 3.0 + sin(p.x * 2.0));
  pattern = pattern * 0.5 + 0.5;

  vec3 colorA = vec3(0.15, 0.25, 0.8);
  vec3 colorB = vec3(0.9, 0.25, 0.15);
  vec3 color = mix(colorA, colorB, pattern);

  outColor = vec4(color, 1.0);
}
