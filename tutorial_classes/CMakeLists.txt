cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(
  TestLib
  LANGUAGES CXX
  DESCRIPTION "Move Semantics Book")

file( GLOB SOURCES src/*.cpp )

foreach( testsourcefile ${SOURCES} )
    get_filename_component( testname ${testsourcefile} NAME_WE )
    add_executable( ${testname} ${testsourcefile} )
endforeach( testsourcefile ${SOURCES} )

