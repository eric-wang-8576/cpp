# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.27.8/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.27.8/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/ericwang/ALL/C++/library_playground

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/ericwang/ALL/C++/library_playground/build

# Include any dependencies generated for this target.
include CMakeFiles/add.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/add.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/add.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/add.dir/flags.make

CMakeFiles/add.dir/src/main.cpp.o: CMakeFiles/add.dir/flags.make
CMakeFiles/add.dir/src/main.cpp.o: /Users/ericwang/ALL/C++/library_playground/src/main.cpp
CMakeFiles/add.dir/src/main.cpp.o: CMakeFiles/add.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ericwang/ALL/C++/library_playground/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/add.dir/src/main.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/add.dir/src/main.cpp.o -MF CMakeFiles/add.dir/src/main.cpp.o.d -o CMakeFiles/add.dir/src/main.cpp.o -c /Users/ericwang/ALL/C++/library_playground/src/main.cpp

CMakeFiles/add.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/add.dir/src/main.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/ericwang/ALL/C++/library_playground/src/main.cpp > CMakeFiles/add.dir/src/main.cpp.i

CMakeFiles/add.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/add.dir/src/main.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/ericwang/ALL/C++/library_playground/src/main.cpp -o CMakeFiles/add.dir/src/main.cpp.s

# Object files for target add
add_OBJECTS = \
"CMakeFiles/add.dir/src/main.cpp.o"

# External object files for target add
add_EXTERNAL_OBJECTS =

add: CMakeFiles/add.dir/src/main.cpp.o
add: CMakeFiles/add.dir/build.make
add: CMakeFiles/add.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/ericwang/ALL/C++/library_playground/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable add"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/add.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/add.dir/build: add
.PHONY : CMakeFiles/add.dir/build

CMakeFiles/add.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/add.dir/cmake_clean.cmake
.PHONY : CMakeFiles/add.dir/clean

CMakeFiles/add.dir/depend:
	cd /Users/ericwang/ALL/C++/library_playground/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/ericwang/ALL/C++/library_playground /Users/ericwang/ALL/C++/library_playground /Users/ericwang/ALL/C++/library_playground/build /Users/ericwang/ALL/C++/library_playground/build /Users/ericwang/ALL/C++/library_playground/build/CMakeFiles/add.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/add.dir/depend

