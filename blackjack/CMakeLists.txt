cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 17)

project(
  TestLib
  LANGUAGES CXX
  DESCRIPTION "Blackjack")

file( GLOB SOURCES src/engine/*.cpp )


add_executable( player src/player.cpp ${SOURCES} )
add_executable( bot src/bot.cpp ${SOURCES} )
#add_executable( tracker src/tracker.cpp ${SOURCES} )

target_include_directories( player PUBLIC /src/engine )
target_include_directories( bot PUBLIC /src/engine )
#target_include_directories( tracker PUBLIC /src/engine )
