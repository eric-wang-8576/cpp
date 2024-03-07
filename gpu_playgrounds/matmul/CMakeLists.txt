cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_example LANGUAGES CXX CUDA)

add_executable(naive src/naive.cu)
add_executable(tiled src/tiled.cu)

set_target_properties(naive PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
