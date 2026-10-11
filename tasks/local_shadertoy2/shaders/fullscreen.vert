#version 450

void main()
{
  vec2 position = gl_VertexIndex == 0
    ? vec2(-1.0, -1.0)
    : (gl_VertexIndex == 1 ? vec2(3.0, -1.0) : vec2(-1.0, 3.0));

  gl_Position = vec4(position, 0.0, 1.0);
}
