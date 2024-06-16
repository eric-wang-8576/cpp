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
add_executable( basic_counting_bot src/basic_counting_bot.cpp ${SOURCES} )
add_executable( josh_training src/josh_training.cpp ${SOURCES} )
add_executable( josh_test src/josh_test.cpp ${SOURCES} )
#add_executable( tracker src/tracker.cpp ${SOURCES} )

target_include_directories( player PUBLIC /src/engine )
target_include_directories( bot PUBLIC /src/engine )
target_include_directories( basic_counting_bot PUBLIC /src/engine )
target_include_directories( josh_training PUBLIC /src/engine )
target_include_directories( josh_test PUBLIC /src/engine )
#target_include_directories( tracker PUBLIC /src/engine )
