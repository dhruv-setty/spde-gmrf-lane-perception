# GPU estimation of (kappa, tau). Configured only when a CUDA toolchain and cuDSS are found;
# the Python package in python/ is the CPU reference. Not built in CI (no GPU runner).
cmake_minimum_required(VERSION 3.20)
project(spde_gpu LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA not found: skipping spde_gpu")
  return()
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(cudss CONFIG QUIET)
if(NOT cudss_FOUND)
  message(STATUS "cuDSS not found (set cudss_DIR): skipping spde_gpu")
  return()
endif()

add_executable(spde_gpu src/spde_gpu.cu)
set_target_properties(spde_gpu PROPERTIES CUDA_STANDARD 17 CXX_STANDARD 17)
target_link_libraries(spde_gpu PRIVATE CUDA::cudart CUDA::cusparse cudss)
