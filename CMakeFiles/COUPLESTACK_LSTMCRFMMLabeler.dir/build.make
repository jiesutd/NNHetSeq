# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.3

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/Jie/Documents/Program/NNHetSeq

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/Jie/Documents/Program/NNHetSeq

# Include any dependencies generated for this target.
include CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/flags.make

CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.o: CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/flags.make
CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.o: COUPLESTACK_LSTMCRFMMLabeler.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Jie/Documents/Program/NNHetSeq/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.o -c /Users/Jie/Documents/Program/NNHetSeq/COUPLESTACK_LSTMCRFMMLabeler.cpp

CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Jie/Documents/Program/NNHetSeq/COUPLESTACK_LSTMCRFMMLabeler.cpp > CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.i

CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Jie/Documents/Program/NNHetSeq/COUPLESTACK_LSTMCRFMMLabeler.cpp -o CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.s

CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.o.requires:

.PHONY : CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.o.requires

CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.o.provides: CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.o.requires
	$(MAKE) -f CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/build.make CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.o.provides.build
.PHONY : CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.o.provides

CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.o.provides.build: CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.o


# Object files for target COUPLESTACK_LSTMCRFMMLabeler
COUPLESTACK_LSTMCRFMMLabeler_OBJECTS = \
"CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.o"

# External object files for target COUPLESTACK_LSTMCRFMMLabeler
COUPLESTACK_LSTMCRFMMLabeler_EXTERNAL_OBJECTS =

COUPLESTACK_LSTMCRFMMLabeler: CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.o
COUPLESTACK_LSTMCRFMMLabeler: CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/build.make
COUPLESTACK_LSTMCRFMMLabeler: CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/Jie/Documents/Program/NNHetSeq/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable COUPLESTACK_LSTMCRFMMLabeler"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/build: COUPLESTACK_LSTMCRFMMLabeler

.PHONY : CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/build

CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/requires: CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.o.requires

.PHONY : CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/requires

CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/cmake_clean.cmake
.PHONY : CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/clean

CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/depend:
	cd /Users/Jie/Documents/Program/NNHetSeq && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Jie/Documents/Program/NNHetSeq /Users/Jie/Documents/Program/NNHetSeq /Users/Jie/Documents/Program/NNHetSeq /Users/Jie/Documents/Program/NNHetSeq /Users/Jie/Documents/Program/NNHetSeq/CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/depend

