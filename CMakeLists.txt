cmake_minimum_required(VERSION 3.28)

# Supported CUDA architectures: sm_86 (RTX 3060/3070/3080, Ampere) and
# sm_120a (RTX 5090, Blackwell). Set and validate before CUDA compiler
# detection so CMake never injects a generic host-toolchain default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to build")
endif()
# Validate: accept 86 (Ampere) or 120a (Blackwell).
string(REPLACE ";" "," _ninfer_arch_list "${CMAKE_CUDA_ARCHITECTURES}")
foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  if(NOT _arch STREQUAL "86" AND NOT _arch STREQUAL "120a")
    message(FATAL_ERROR
      "NInfer supports CMAKE_CUDA_ARCHITECTURES of 86 or 120a; got '${_arch}'")
  endif()
endforeach()
# Track the target architecture for conditional compilation.
if(CMAKE_CUDA_ARCHITECTURES MATCHES "120a")
  set(NINFER_TARGET_SM_120 TRUE)
  set(NINFER_TARGET_SM_86 FALSE)
else()
  set(NINFER_TARGET_SM_120 FALSE)
  set(NINFER_TARGET_SM_86 TRUE)
endif()

project(ninfer LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.1)
  message(FATAL_ERROR
    "NInfer requires CUDA 13.1 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()

# Export the target-architecture selection to source files for host-side
# conditional compilation (cooperative grid sizing, feature gating).
if(NINFER_TARGET_SM_86)
  add_compile_definitions(NINFER_TARGET_SM_86=1)
endif()
if(NINFER_TARGET_SM_120)
  add_compile_definitions(NINFER_TARGET_SM_120=1)
endif()

set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
  libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
if(NINFER_BUILD_MEDIA_ACQUIRE)
  pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
endif()
find_package(Threads REQUIRED)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
