cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 17)

project(
  TestLib
  LANGUAGES CXX
  DESCRIPTION "polymorphism")

file( GLOB SOURCES src/*.cpp )

include_directories(include)

add_executable(main main.cpp ${SOURCES})

#foreach( testsourcefile ${SOURCES} )
#    get_filename_component( testname ${testsourcefile} NAME_WE )
#    add_executable( ${testname} ${testsourcefile} )
#endforeach( testsourcefile ${SOURCES} )

