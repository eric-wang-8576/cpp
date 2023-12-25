cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CXX_STANDARD 17)

project(
  TestLib
  LANGUAGES CXX
  DESCRIPTION "CS107")

add_executable(main src/main.cpp)
