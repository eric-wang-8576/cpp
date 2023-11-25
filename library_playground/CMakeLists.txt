cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CXX_STANDARD 17)

project(
  TestLib
  LANGUAGES CXX
  DESCRIPTION "Library Test")

add_subdirectory(library)
add_executable(main src/main.cpp)
target_link_libraries(main PUBLIC library)
target_include_directories(main PUBLIC "${PROJECT_SOURCE_DIR}/library")

