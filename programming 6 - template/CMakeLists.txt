cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 17)

project(
  TestLib
  LANGUAGES CXX
  DESCRIPTION "Packet Processor")

file( GLOB SOURCES src/*.cpp)

add_executable( main main.cpp ${SOURCES} )
target_include_directories( main PUBLIC include )

add_executable( tracker_test tracker_test.cpp ${SOURCES} )
target_include_directories( tracker_test PUBLIC include )
