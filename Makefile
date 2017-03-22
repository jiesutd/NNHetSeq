# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.3

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


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

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/local/bin/ccmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/local/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/Jie/Documents/Program/NNHetSeq/CMakeFiles /Users/Jie/Documents/Program/NNHetSeq/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/Jie/Documents/Program/NNHetSeq/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named COUPLESTACK_LSTMCRFMMLabeler

# Build rule for target.
COUPLESTACK_LSTMCRFMMLabeler: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 COUPLESTACK_LSTMCRFMMLabeler
.PHONY : COUPLESTACK_LSTMCRFMMLabeler

# fast build rule for target.
COUPLESTACK_LSTMCRFMMLabeler/fast:
	$(MAKE) -f CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/build.make CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/build
.PHONY : COUPLESTACK_LSTMCRFMMLabeler/fast

#=============================================================================
# Target rules for targets named COUPLE_LSTMCRFMMLabeler

# Build rule for target.
COUPLE_LSTMCRFMMLabeler: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 COUPLE_LSTMCRFMMLabeler
.PHONY : COUPLE_LSTMCRFMMLabeler

# fast build rule for target.
COUPLE_LSTMCRFMMLabeler/fast:
	$(MAKE) -f CMakeFiles/COUPLE_LSTMCRFMMLabeler.dir/build.make CMakeFiles/COUPLE_LSTMCRFMMLabeler.dir/build
.PHONY : COUPLE_LSTMCRFMMLabeler/fast

#=============================================================================
# Target rules for targets named LSTMCRFMMLabeler

# Build rule for target.
LSTMCRFMMLabeler: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 LSTMCRFMMLabeler
.PHONY : LSTMCRFMMLabeler

# fast build rule for target.
LSTMCRFMMLabeler/fast:
	$(MAKE) -f CMakeFiles/LSTMCRFMMLabeler.dir/build.make CMakeFiles/LSTMCRFMMLabeler.dir/build
.PHONY : LSTMCRFMMLabeler/fast

#=============================================================================
# Target rules for targets named STACK_LSTMCRFMMLabeler

# Build rule for target.
STACK_LSTMCRFMMLabeler: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 STACK_LSTMCRFMMLabeler
.PHONY : STACK_LSTMCRFMMLabeler

# fast build rule for target.
STACK_LSTMCRFMMLabeler/fast:
	$(MAKE) -f CMakeFiles/STACK_LSTMCRFMMLabeler.dir/build.make CMakeFiles/STACK_LSTMCRFMMLabeler.dir/build
.PHONY : STACK_LSTMCRFMMLabeler/fast

COUPLESTACK_LSTMCRFMMLabeler.o: COUPLESTACK_LSTMCRFMMLabeler.cpp.o

.PHONY : COUPLESTACK_LSTMCRFMMLabeler.o

# target to build an object file
COUPLESTACK_LSTMCRFMMLabeler.cpp.o:
	$(MAKE) -f CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/build.make CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.o
.PHONY : COUPLESTACK_LSTMCRFMMLabeler.cpp.o

COUPLESTACK_LSTMCRFMMLabeler.i: COUPLESTACK_LSTMCRFMMLabeler.cpp.i

.PHONY : COUPLESTACK_LSTMCRFMMLabeler.i

# target to preprocess a source file
COUPLESTACK_LSTMCRFMMLabeler.cpp.i:
	$(MAKE) -f CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/build.make CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.i
.PHONY : COUPLESTACK_LSTMCRFMMLabeler.cpp.i

COUPLESTACK_LSTMCRFMMLabeler.s: COUPLESTACK_LSTMCRFMMLabeler.cpp.s

.PHONY : COUPLESTACK_LSTMCRFMMLabeler.s

# target to generate assembly for a file
COUPLESTACK_LSTMCRFMMLabeler.cpp.s:
	$(MAKE) -f CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/build.make CMakeFiles/COUPLESTACK_LSTMCRFMMLabeler.dir/COUPLESTACK_LSTMCRFMMLabeler.cpp.s
.PHONY : COUPLESTACK_LSTMCRFMMLabeler.cpp.s

COUPLE_LSTMCRFMMLabeler.o: COUPLE_LSTMCRFMMLabeler.cpp.o

.PHONY : COUPLE_LSTMCRFMMLabeler.o

# target to build an object file
COUPLE_LSTMCRFMMLabeler.cpp.o:
	$(MAKE) -f CMakeFiles/COUPLE_LSTMCRFMMLabeler.dir/build.make CMakeFiles/COUPLE_LSTMCRFMMLabeler.dir/COUPLE_LSTMCRFMMLabeler.cpp.o
.PHONY : COUPLE_LSTMCRFMMLabeler.cpp.o

COUPLE_LSTMCRFMMLabeler.i: COUPLE_LSTMCRFMMLabeler.cpp.i

.PHONY : COUPLE_LSTMCRFMMLabeler.i

# target to preprocess a source file
COUPLE_LSTMCRFMMLabeler.cpp.i:
	$(MAKE) -f CMakeFiles/COUPLE_LSTMCRFMMLabeler.dir/build.make CMakeFiles/COUPLE_LSTMCRFMMLabeler.dir/COUPLE_LSTMCRFMMLabeler.cpp.i
.PHONY : COUPLE_LSTMCRFMMLabeler.cpp.i

COUPLE_LSTMCRFMMLabeler.s: COUPLE_LSTMCRFMMLabeler.cpp.s

.PHONY : COUPLE_LSTMCRFMMLabeler.s

# target to generate assembly for a file
COUPLE_LSTMCRFMMLabeler.cpp.s:
	$(MAKE) -f CMakeFiles/COUPLE_LSTMCRFMMLabeler.dir/build.make CMakeFiles/COUPLE_LSTMCRFMMLabeler.dir/COUPLE_LSTMCRFMMLabeler.cpp.s
.PHONY : COUPLE_LSTMCRFMMLabeler.cpp.s

LSTMCRFMMLabeler.o: LSTMCRFMMLabeler.cpp.o

.PHONY : LSTMCRFMMLabeler.o

# target to build an object file
LSTMCRFMMLabeler.cpp.o:
	$(MAKE) -f CMakeFiles/LSTMCRFMMLabeler.dir/build.make CMakeFiles/LSTMCRFMMLabeler.dir/LSTMCRFMMLabeler.cpp.o
.PHONY : LSTMCRFMMLabeler.cpp.o

LSTMCRFMMLabeler.i: LSTMCRFMMLabeler.cpp.i

.PHONY : LSTMCRFMMLabeler.i

# target to preprocess a source file
LSTMCRFMMLabeler.cpp.i:
	$(MAKE) -f CMakeFiles/LSTMCRFMMLabeler.dir/build.make CMakeFiles/LSTMCRFMMLabeler.dir/LSTMCRFMMLabeler.cpp.i
.PHONY : LSTMCRFMMLabeler.cpp.i

LSTMCRFMMLabeler.s: LSTMCRFMMLabeler.cpp.s

.PHONY : LSTMCRFMMLabeler.s

# target to generate assembly for a file
LSTMCRFMMLabeler.cpp.s:
	$(MAKE) -f CMakeFiles/LSTMCRFMMLabeler.dir/build.make CMakeFiles/LSTMCRFMMLabeler.dir/LSTMCRFMMLabeler.cpp.s
.PHONY : LSTMCRFMMLabeler.cpp.s

STACK_LSTMCRFMMLabeler.o: STACK_LSTMCRFMMLabeler.cpp.o

.PHONY : STACK_LSTMCRFMMLabeler.o

# target to build an object file
STACK_LSTMCRFMMLabeler.cpp.o:
	$(MAKE) -f CMakeFiles/STACK_LSTMCRFMMLabeler.dir/build.make CMakeFiles/STACK_LSTMCRFMMLabeler.dir/STACK_LSTMCRFMMLabeler.cpp.o
.PHONY : STACK_LSTMCRFMMLabeler.cpp.o

STACK_LSTMCRFMMLabeler.i: STACK_LSTMCRFMMLabeler.cpp.i

.PHONY : STACK_LSTMCRFMMLabeler.i

# target to preprocess a source file
STACK_LSTMCRFMMLabeler.cpp.i:
	$(MAKE) -f CMakeFiles/STACK_LSTMCRFMMLabeler.dir/build.make CMakeFiles/STACK_LSTMCRFMMLabeler.dir/STACK_LSTMCRFMMLabeler.cpp.i
.PHONY : STACK_LSTMCRFMMLabeler.cpp.i

STACK_LSTMCRFMMLabeler.s: STACK_LSTMCRFMMLabeler.cpp.s

.PHONY : STACK_LSTMCRFMMLabeler.s

# target to generate assembly for a file
STACK_LSTMCRFMMLabeler.cpp.s:
	$(MAKE) -f CMakeFiles/STACK_LSTMCRFMMLabeler.dir/build.make CMakeFiles/STACK_LSTMCRFMMLabeler.dir/STACK_LSTMCRFMMLabeler.cpp.s
.PHONY : STACK_LSTMCRFMMLabeler.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... STACK_LSTMCRFMMLabeler"
	@echo "... COUPLESTACK_LSTMCRFMMLabeler"
	@echo "... COUPLE_LSTMCRFMMLabeler"
	@echo "... rebuild_cache"
	@echo "... LSTMCRFMMLabeler"
	@echo "... COUPLESTACK_LSTMCRFMMLabeler.o"
	@echo "... COUPLESTACK_LSTMCRFMMLabeler.i"
	@echo "... COUPLESTACK_LSTMCRFMMLabeler.s"
	@echo "... COUPLE_LSTMCRFMMLabeler.o"
	@echo "... COUPLE_LSTMCRFMMLabeler.i"
	@echo "... COUPLE_LSTMCRFMMLabeler.s"
	@echo "... LSTMCRFMMLabeler.o"
	@echo "... LSTMCRFMMLabeler.i"
	@echo "... LSTMCRFMMLabeler.s"
	@echo "... STACK_LSTMCRFMMLabeler.o"
	@echo "... STACK_LSTMCRFMMLabeler.i"
	@echo "... STACK_LSTMCRFMMLabeler.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system
