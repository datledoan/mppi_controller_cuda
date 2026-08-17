cmake_minimum_required(VERSION 3.0.2)
project(mppi_controller_cuda)

add_compile_options(-O3)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mno-avx512f" COMPILER_SUPPORTS_AVX512)
check_cxx_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE4)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

if(COMPILER_SUPPORTS_AVX512)
  add_compile_options(-mno-avx512f)
  list(APPEND CUDA_NVCC_FLAGS -Xcompiler=-mno-avx512f)
endif()
if(COMPILER_SUPPORTS_SSE4)
  add_compile_options(-msse4.2)
  list(APPEND CUDA_NVCC_FLAGS -Xcompiler=-msse4.2)
endif()
if(COMPILER_SUPPORTS_AVX2)
  add_compile_options(-mavx2)
  list(APPEND CUDA_NVCC_FLAGS -Xcompiler=-mavx2)
endif()
if(COMPILER_SUPPORTS_FMA)
  add_compile_options(-mfma)
  list(APPEND CUDA_NVCC_FLAGS -Xcompiler=-mfma)
endif()
list(APPEND CUDA_NVCC_FLAGS -O3)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)

add_definitions(-DEIGEN_DONT_VECTORIZE)
list(APPEND CUDA_NVCC_FLAGS -DEIGEN_DONT_VECTORIZE)

find_package(catkin REQUIRED COMPONENTS
  angles
  base_local_planner
  costmap_2d
  mbf_costmap_core
  mbf_msgs
  nav_core
  pluginlib
  roscpp
  nav_msgs
  std_msgs
  visualization_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  dynamic_reconfigure
)

find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

generate_dynamic_reconfigure_options(
  cfg/MPPIControllerCUDA.cfg
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES mppi_controller_cuda
  CATKIN_DEPENDS
  angles
  base_local_planner
  costmap_2d
  mbf_costmap_core
  mbf_msgs
  nav_core
  pluginlib
  roscpp
  nav_msgs
  std_msgs
  visualization_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  dynamic_reconfigure
)

cuda_add_library(mppi_controller_cuda SHARED
  src/controller.cpp
  src/controller/mppi_controller.cu
  src/core/mppi_common.cu
  src/costs/cost.cu
  src/dynamics/dynamics.cu
  src/tools/parameters_handler.cpp
  src/tools/path_handler.cpp
  src/tools/trajectory_visualizer.cpp
)

target_link_libraries(mppi_controller_cuda
  ${catkin_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${CUDA_CUFFT_LIBRARIES}
)

install(TARGETS mppi_controller_cuda
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

option(MPPI_BUILD_BENCHMARKS "Build the benchmark/cuda_sweep_bench_* executables" OFF)
if(MPPI_BUILD_BENCHMARKS)
  set(MPPI_BENCH_ROLLOUT_COUNTS 256 512 1024 2048 4096 8192 CACHE STRING
      "Rollout counts (NUM_ROLLOUTS) to build a benchmark/sweep_bench binary for")
  foreach(N ${MPPI_BENCH_ROLLOUT_COUNTS})
    cuda_add_executable(cuda_sweep_bench_${N}
      benchmark/sweep_bench.cu
      src/core/mppi_common.cu
      src/controller/mppi_controller.cu
      OPTIONS -DMPPI_NUM_ROLLOUTS=${N}
    )
    target_link_libraries(cuda_sweep_bench_${N}
      ${CUDA_LIBRARIES}
      ${CUDA_curand_LIBRARY}
      ${CUDA_CUFFT_LIBRARIES}
    )
  endforeach()
endif()

install(FILES mppic.xml
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
