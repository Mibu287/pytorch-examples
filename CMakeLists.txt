cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pytorch_cuda_example LANGUAGES CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Pytorch libraries
set(TORCH_INSTALL_PREFIX "/root/pytorch/dist")
add_library(torch_external INTERFACE)
target_compile_options(torch_external INTERFACE "-fclang-abi-compat=17")
target_include_directories(
  torch_external
  INTERFACE "${TORCH_INSTALL_PREFIX}/include"
  INTERFACE "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include")

target_link_directories(torch_external INTERFACE "${TORCH_INSTALL_PREFIX}/lib")

target_link_libraries(
  torch_external
  INTERFACE "${TORCH_INSTALL_PREFIX}/lib/libtorch.so"
            "${TORCH_INSTALL_PREFIX}/lib/libtorch_cuda.so"
            "${TORCH_INSTALL_PREFIX}/lib/libtorch_cpu.so"
            "${TORCH_INSTALL_PREFIX}/lib/libc10.so"
            "${TORCH_INSTALL_PREFIX}/lib/libc10_cuda.so")

# Enable CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Set CUDA architectures (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")

# Create executable
add_executable(torch_cuda_example torch_cuda_example.cpp)
target_link_libraries(torch_cuda_example PRIVATE torch_external)

add_custom_target(
  run_torch_cuda_example
  COMMAND torch_cuda_example
  DEPENDS torch_cuda_example
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running torch_cuda_example")
