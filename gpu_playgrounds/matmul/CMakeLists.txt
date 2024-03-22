cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_example LANGUAGES CXX CUDA)

add_executable(matmul src/matmul.cu)

set_target_properties(matmul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
