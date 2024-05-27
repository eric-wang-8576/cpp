cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 17)

project(
  TestLib
  LANGUAGES CXX
  DESCRIPTION "Miscellaneous")

# Find all .cpp files in the source directory
file(GLOB SOURCES "src/*.cpp")

# Loop through each .cpp file and create an executable
foreach(SOURCE_FILE ${SOURCES})
    # Get the filename without the extension
    get_filename_component(EXECUTABLE_NAME ${SOURCE_FILE} NAME_WE)

    # Create an executable for each source file
    add_executable(${EXECUTABLE_NAME} ${SOURCE_FILE})
endforeach()
