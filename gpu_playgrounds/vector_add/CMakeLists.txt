cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_example LANGUAGES CXX CUDA)

add_executable(vector_add vector_add.cu)

set_target_properties(vector_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
