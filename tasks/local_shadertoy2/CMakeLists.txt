add_executable(local_shadertoy2
  main.cpp
  App.cpp
)

target_link_libraries(local_shadertoy2
  PRIVATE glfw etna glm::glm wsi gui)

target_add_shaders(local_shadertoy2
  shaders/fullscreen.vert
  shaders/buffer_a.frag
  shaders/buffer_b.frag
  shaders/image.frag
)
