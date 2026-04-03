
if(NOT TARGET urdfdom_headers::urdfdom_headers)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(urdfdom_headers REQUIRED urdfdom_headers)
  
  add_library(urdfdom_headers::urdfdom_headers INTERFACE IMPORTED)
  target_include_directories(urdfdom_headers::urdfdom_headers INTERFACE ${urdfdom_headers_INCLUDE_DIRS})
  target_link_options(urdfdom_headers::urdfdom_headers INTERFACE ${urdfdom_headers_LDFLAGS})

  set(urdfdom_headers_FOUND TRUE)
endif()
