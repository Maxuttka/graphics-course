add_executable(instancing
  main.cpp
  App.cpp
  Renderer.cpp
  WorldRenderer.cpp
)

target_link_libraries(instancing
  PRIVATE glfw etna glm::glm tinygltf wsi gui scene)

target_add_shaders(instancing
  shaders/heightmap.comp
  shaders/terrain.vert
  shaders/terrain.frag
  shaders/rock.vert
  shaders/rock.frag
)
