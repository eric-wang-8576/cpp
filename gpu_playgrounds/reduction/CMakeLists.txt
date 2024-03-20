cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_example LANGUAGES CXX CUDA)

add_executable(reduction src/reduction.cu)

set_target_properties(reduction PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
