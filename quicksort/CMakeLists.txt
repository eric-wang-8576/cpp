cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 17)

project(
  TestLib
  LANGUAGES CXX
  DESCRIPTION "Quicksort")

add_executable( main main.cpp )
#target_include_directories( main PUBLIC include )
