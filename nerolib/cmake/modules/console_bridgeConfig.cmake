
if(NOT TARGET console_bridge::console_bridge)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(console_bridge REQUIRED console_bridge)

  add_library(console_bridge::console_bridge INTERFACE IMPORTED)
  target_include_directories(console_bridge::console_bridge INTERFACE ${console_bridge_INCLUDE_DIRS})
  target_link_libraries(console_bridge::console_bridge INTERFACE ${console_bridge_LIBRARIES})
  target_link_directories(console_bridge::console_bridge INTERFACE ${console_bridge_LIBRARY_DIRS})
  
  set(console_bridge_FOUND TRUE)
  set(console_bridge_DIR "${CMAKE_CURRENT_LIST_DIR}")
endif()
