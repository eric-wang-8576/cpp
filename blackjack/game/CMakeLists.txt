include_directories(${PROJECT_SOURCE_DIR}/engine/include)

# Player 
add_executable(player player.cpp)
target_link_libraries(player PRIVATE Engine)

# Bots
file(GLOB BOT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/bots/*.cpp")
foreach(SOURCE_FILE ${BOT_SOURCES})
    get_filename_component(FILENAME ${SOURCE_FILE} NAME_WE)
    add_executable(${FILENAME} ${SOURCE_FILE})
    target_link_libraries(${FILENAME} PRIVATE Engine)
endforeach()
