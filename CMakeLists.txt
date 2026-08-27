cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(mppi_controller_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mno-avx512f" COMPILER_SUPPORTS_AVX512)
check_cxx_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE4)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

if(COMPILER_SUPPORTS_AVX512)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mno-avx512f> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mno-avx512f>)
endif()

if(COMPILER_SUPPORTS_SSE4)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-msse4.2> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-msse4.2>)
endif()

if(COMPILER_SUPPORTS_AVX2)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mavx2> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mavx2>)
endif()

if(COMPILER_SUPPORTS_FMA)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mfma> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mfma>)
endif()

# If building on the same hardware to be deployed on, try `-march=native`!
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3> $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

add_definitions(-DEIGEN_DONT_VECTORIZE -DEIGEN_MAX_ALIGN_BYTES=0)

find_package(ament_cmake REQUIRED)
find_package(angles REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
)

add_library(mppi_controller_cuda SHARED
  src/controller.cpp
  src/controller/mppi_controller.cu
  src/core/mppi_common.cu
  src/costs/cost.cu
  src/dynamics/dynamics.cu
  src/tools/parameters_handler.cpp
  src/tools/path_handler.cpp
  src/tools/trajectory_visualizer.cpp
)

set_target_properties(mppi_controller_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(mppi_controller_cuda PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)

ament_target_dependencies(mppi_controller_cuda PUBLIC
  angles
  geometry_msgs
  nav2_core
  nav2_costmap_2d
  nav2_util
  nav_msgs
  pluginlib
  rclcpp
  rclcpp_lifecycle
  std_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  visualization_msgs
)

target_link_libraries(mppi_controller_cuda PUBLIC
  CUDA::cudart
  CUDA::curand
  CUDA::cufft
)

install(TARGETS mppi_controller_cuda
  EXPORT export_mppi_controller_cuda
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(FILES mppic.xml
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY params
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include/${PROJECT_NAME})
ament_export_libraries(mppi_controller_cuda)
ament_export_targets(export_mppi_controller_cuda)
ament_export_dependencies(
  angles
  geometry_msgs
  nav2_core
  nav2_costmap_2d
  nav2_util
  nav_msgs
  pluginlib
  rclcpp
  rclcpp_lifecycle
  std_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  visualization_msgs
  Eigen3
)

# Build the benchmark binaries
# Usage: colcon build --packages-select mppi_controller_cuda --cmake-args -DMPPI_BUILD_BENCHMARKS=ON
option(MPPI_BUILD_BENCHMARKS "Build the benchmark/{cuda,cpu}_sweep_bench_* executables" OFF)
if(MPPI_BUILD_BENCHMARKS)
  set(MPPI_BENCH_ROLLOUT_COUNTS 256 512 1024 2048 4096 8192 CACHE STRING
      "Rollout counts (NUM_ROLLOUTS) to build a benchmark/cuda_sweep_bench_* binary for")
  foreach(N ${MPPI_BENCH_ROLLOUT_COUNTS})
    add_executable(cuda_sweep_bench_${N}
      benchmark/sweep_bench.cu
      src/core/mppi_common.cu
      src/controller/mppi_controller.cu
    )
    target_include_directories(cuda_sweep_bench_${N} PRIVATE
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    )
    target_compile_definitions(cuda_sweep_bench_${N} PRIVATE MPPI_NUM_ROLLOUTS=${N})
    set_target_properties(cuda_sweep_bench_${N} PROPERTIES
      CUDA_SEPARABLE_COMPILATION OFF
      POSITION_INDEPENDENT_CODE ON
    )
    target_link_libraries(cuda_sweep_bench_${N}
      CUDA::cudart
      CUDA::curand
      CUDA::cufft
    )
  endforeach()

  # CPU benchmark binary
  find_package(nav2_mppi_controller REQUIRED)
  add_executable(cpu_sweep_bench benchmark/cpu_sweep_bench.cpp)
  ament_target_dependencies(cpu_sweep_bench
    rclcpp
    rclcpp_lifecycle
    nav2_core
    nav2_costmap_2d
    nav2_mppi_controller
    geometry_msgs
    nav_msgs
  )
endif()

pluginlib_export_plugin_description_file(nav2_core mppic.xml)

ament_package()
