cmake_minimum_required(VERSION 3.15)
project(nerolib LANGUAGES CXX)

# Set build type based on environment
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)


# Ensure build-tree RPATHs use $ORIGIN so binaries/modules find colocated libs
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
if (APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

if (NOT DEFINED ENV{NEROLIB_CONDA_ENV} OR "$ENV{NEROLIB_CONDA_ENV}" STREQUAL "")

  if (DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
    set(NEROLIB_CONDA_ENV $ENV{CONDA_PREFIX})
  else()
    set(NEROLIB_CONDA_ENV "$ENV{HOME}/miniforge3/envs/nerolib")
  endif()
else()
  set(NEROLIB_CONDA_ENV $ENV{NEROLIB_CONDA_ENV})
endif()
message(STATUS "Using conda environment: ${NEROLIB_CONDA_ENV}")



set(CMAKE_PREFIX_PATH 
    ${NEROLIB_CONDA_ENV} 
    "${CMAKE_SOURCE_DIR}/cmake/modules"
    ${CMAKE_PREFIX_PATH}
)


if (APPLE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${NEROLIB_CONDA_ENV}/lib/libomp.dylib")
endif()

include_directories("${NEROLIB_CONDA_ENV}/include")
link_directories("${NEROLIB_CONDA_ENV}/lib")

# set(Boost_NO_BOOST_CMAKE ON)
# set(BOOST_ROOT "${NEROLIB_CONDA_ENV}")
# set(Boost_INCLUDE_DIR "${NEROLIB_CONDA_ENV}/include")
# set(Boost_LIBRARY_DIR "${NEROLIB_CONDA_ENV}/lib")



add_compile_definitions(SDK_ROOT="${CMAKE_BINARY_DIR}/..")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem serialization)

find_package(pinocchio REQUIRED)
find_package(spdlog REQUIRED)
find_package(Catch2 REQUIRED)

set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(ruckig)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${NEROLIB_CONDA_ENV}/include)



find_library(FMT_LIB fmt HINTS "${NEROLIB_CONDA_ENV}/lib" NO_DEFAULT_PATH)
message(STATUS "FMT_LIB found at: ${FMT_LIB}")




# Create a single static library that bundles everything
add_library(NeroController SHARED
    src/socket_can.cpp
    src/nero_interface.cpp
    src/utils.cpp
    src/solver.cpp
    src/controller_base.cpp
)
target_link_libraries(NeroController
    spdlog::spdlog_header_only
    ${FMT_LIB}
    pinocchio::pinocchio
    ruckig
)



# TESTS

# TESTS (only build if not installing via pip)
if(NOT SKBUILD)
  add_executable(test_interface test/test_interface.cpp)
  target_link_libraries(test_interface PRIVATE NeroController Catch2::Catch2)

  add_executable(test_gravity_compensation test/test_gravity_compensation.cpp)
  target_link_libraries(test_gravity_compensation PRIVATE NeroController)

  add_executable(test_motor_control test/test_motor_control.cpp)
  target_link_libraries(test_motor_control PRIVATE NeroController Catch2::Catch2WithMain)

  add_executable(verify_fk test/verify_fk.cpp)
  target_link_libraries(verify_fk NeroController pinocchio::pinocchio spdlog::spdlog)

  add_executable(test_compliant_control test/test_compliant_control.cpp)
  target_link_libraries(test_compliant_control PRIVATE NeroController)

  add_executable(test_nero_controller test/test_nero_controller.cpp)
  target_link_libraries(test_nero_controller PRIVATE NeroController)

endif()

# EXTRAS
add_executable(resume_emergency_stop extra/resume_emergency_stop.cpp)
target_link_libraries(resume_emergency_stop PRIVATE NeroController)

add_executable(emergency_stop extra/emergency_stop.cpp)
target_link_libraries(emergency_stop PRIVATE NeroController)

add_executable(diag_raw_can extra/diag_raw_can.cpp)
target_link_libraries(diag_raw_can PRIVATE NeroController)

# PYTHON
pybind11_add_module(_nerolib src/nero_pybind.cpp)
target_link_libraries(_nerolib PRIVATE NeroController)
set_target_properties(_nerolib PROPERTIES PREFIX "")
set_target_properties(_nerolib PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/nerolib")
set_target_properties(NeroController PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/nerolib")

install(TARGETS _nerolib DESTINATION nerolib)
install(TARGETS NeroController DESTINATION nerolib)
install(TARGETS ruckig DESTINATION nerolib)




