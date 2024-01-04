cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 17)

project(
  TestLib
  LANGUAGES CXX
  DESCRIPTION "Move Semantics Book")

file( GLOB SOURCES src/*.cpp )

include_directories(include /src/)

add_executable( main ${SOURCES} )
