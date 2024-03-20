cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_example LANGUAGES CXX CUDA)

add_executable(histogram src/histogram.cu)

set_target_properties(histogram PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
