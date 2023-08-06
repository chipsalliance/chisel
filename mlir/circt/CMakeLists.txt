##===- CMakeLists.txt - CIRCT cmake root ----------------------*- cmake -*-===//
##
## Configure the CIRCT build.
##
##===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.13.4)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# If we are not building as a part of LLVM, build Circt as an
# standalone project, using LLVM as an external library:
if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  if (CIRCT_BINDINGS_PYTHON_ENABLED)
    message(FATAL_ERROR "CIRCT Python bindings require a unified build. \
                         See docs/PythonBindings.md.")
  endif()

# Generate a CompilationDatabase (compile_commands.json file) for our build,
# for use by clang_complete, YouCompleteMe, etc.
  set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------
  project(circt LANGUAGES CXX C)

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------

  option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
  option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHs-c- /GR-")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-rtti")
endif ()

#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------

  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  set(CIRCT_BUILT_STANDALONE 1)
  set(BACKEND_PACKAGE_STRING "LLVM ${LLVM_PACKAGE_VERSION}")

  # Handle unittests when building out-of-tree against an installed version of
  # LLVM/MLIR (not a build tree). Adapted from `llvm/flang/CMakeLists.txt`.
  set(CIRCT_GTEST_AVAILABLE 0)
  set(UNITTEST_DIR ${LLVM_THIRD_PARTY_DIR}/unittest)
  if (NOT EXISTS ${UNITTEST_DIR}/googletest/include/gtest/gtest.h)
    set(UNITTEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llvm/third-party/unittest)
  endif()
  if (EXISTS ${UNITTEST_DIR}/googletest/include/gtest/gtest.h)
    if (NOT TARGET llvm_gtest)
      find_package(Threads)
      add_llvm_library(llvm_gtest
        ${UNITTEST_DIR}/googletest/src/gtest-all.cc
        ${UNITTEST_DIR}/googlemock/src/gmock-all.cc
        LINK_COMPONENTS Support # llvm::raw_ostream
        BUILDTREE_ONLY
      )
      target_include_directories(llvm_gtest
        PUBLIC
        "${UNITTEST_DIR}/googletest/include"
        "${UNITTEST_DIR}/googlemock/include"
        PRIVATE
        "${UNITTEST_DIR}/googletest"
        "${UNITTEST_DIR}/googlemock"
      )
      target_link_libraries(llvm_gtest PUBLIC Threads::Threads)
      add_llvm_library(llvm_gtest_main
        ${UNITTEST_DIR}/UnitTestMain/TestMain.cpp
        LINK_LIBS llvm_gtest
        LINK_COMPONENTS Support # llvm::cl
        BUILDTREE_ONLY
      )
    endif()
    set(CIRCT_GTEST_AVAILABLE 1)
  else()
    message(WARNING "Skipping unittests since LLVM install does not include \
      gtest headers and libraries")
    set(CIRCT_GTEST_AVAILABLE 0)
  endif()

else()
  # CMake library generation settings.
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Default to building a static mondo-lib")
  set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON CACHE BOOL
    "Python soname linked libraries are bad")
  set(CMAKE_VISIBILITY_INLINES_HIDDEN ON CACHE BOOL "Hide inlines")

  # The -fvisibility=hidden option only works for static builds.
  if (BUILD_SHARED_LIBS AND (CMAKE_CXX_VISIBILITY_PRESET STREQUAL "hidden"))
    message(FATAL_ERROR "CMAKE_CXX_VISIBILITY_PRESET=hidden is incompatible \
                         with BUILD_SHARED_LIBS.")
  endif()

  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir ) # --src-root
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include ) # --includedir
  set(MLIR_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)
  include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
  include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})

  set(BACKEND_PACKAGE_STRING "${PACKAGE_STRING}")

  set(CIRCT_GTEST_AVAILABLE 1)
endif()

# Define the default arguments to use with 'lit', and an option for the user to
# override.
set(LIT_ARGS_DEFAULT "-sv")
if (MSVC_IDE OR XCODE)
  set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()
set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

#-------------------------------------------------------------------------------
# CIRCT configuration
#-------------------------------------------------------------------------------

# CIRCT project.
set(CIRCT_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR} ) # --src-root
set(CIRCT_MAIN_INCLUDE_DIR ${CIRCT_MAIN_SRC_DIR}/include)

set(CIRCT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CIRCT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CIRCT_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(CIRCT_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib)
set(CIRCT_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)
set(CIRCT_UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/utils)
set(CIRCT_PYTHON_PACKAGES_DIR ${CIRCT_BINARY_DIR}/python_packages)

list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(AddCIRCT)

# Installing the headers and docs needs to depend on generating any public
# tablegen'd targets.
add_custom_target(circt-headers)
set_target_properties(circt-headers PROPERTIES FOLDER "Misc")
add_custom_target(circt-doc)

# Add MLIR and LLVM headers to the include path
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Add CIRCT files to the include path
include_directories(${CIRCT_MAIN_INCLUDE_DIR})
include_directories(${CIRCT_INCLUDE_DIR})

# Set the release tag.
option(CIRCT_RELEASE_TAG_ENABLED "Emit the release tag to output." OFF)
if (NOT CIRCT_RELEASE_TAG_ENABLED)
  message(STATUS "Version generation is disabled. To enable the version "
               "generation, please set CIRCT_RELEASE_TAG_ENABLED CMake "
               "variable")
endif()

set(CIRCT_RELEASE_TAG "circtorg" CACHE STRING
    "Prefix of the release tag (e.g. circtorg, firtool, and pycde).")

#-------------------------------------------------------------------------------
# Verilator Configuration
#-------------------------------------------------------------------------------

# If Verilator hasn't been explicitly disabled, find it.
option(VERILATOR_DISABLE "Disable the Verilator tests.")
if (VERILATOR_DISABLE)
  message(STATUS "Disabling Verilator tests.")
else()
  # Detect if Verilator is present.
  if (NOT DEFINED VERILATOR_PATH)
    find_program(VERILATOR_PATH "verilator" PATHS
      "${CMAKE_CURRENT_SOURCE_DIR}/ext/bin" NO_DEFAULT_PATH)
    find_program(VERILATOR_PATH "verilator")
  endif()

  if(EXISTS ${VERILATOR_PATH})
    message(STATUS "Found Verilator at ${VERILATOR_PATH}.")

    # Find Verilator version.
    execute_process(COMMAND ${VERILATOR_PATH} --version
      OUTPUT_VARIABLE VERILATOR_VERSION)
    string(REGEX MATCH "Verilator (([0-9]+)\.([0-9]+)) \.*"
      MATCH ${VERILATOR_VERSION})
    # It's gotta be at least v4.110.
    if (${CMAKE_MATCH_1} LESS 4.110)
      message(FATAL_ERROR "CIRCT only supports Verilator version 4.110 and up. \
                           Found version: ${CMAKE_MATCH_1}. You can disable \
                           the Verilator tests with '-DVERILATOR_DISABLE=ON'.")
      set(VERILATOR_PATH "")
    endif()
  else()
    set(VERILATOR_PATH "")
    message(STATUS "Did not find Verilator.")
  endif()
endif()

#-------------------------------------------------------------------------------
# Vivado Configuration
#-------------------------------------------------------------------------------

# If vivado hasn't been explicitly disabled, find it.
option(VIVADO_DISABLE "Disable the vivado synthesis tests.")
if (VIVADO_DISABLE)
  message(STATUS "Disabling vivado tests.")
else()
  if (EXISTS ${VIVADO_PATH})
    get_filename_component(VIVADO_PATH ${VIVADO_PATH} DIRECTORY)
    message(STATUS "Setting vivado path to ${VIVADO_PATH}.")
  else()
    # Search for vivado's `vivado` command.
    find_program(VIVADO_PATH "vivado")
    if(EXISTS ${VIVADO_PATH})
      # Then strip the filename.
      get_filename_component(VIVADO_PATH ${VIVADO_PATH} DIRECTORY)
      message(STATUS "Found vivado at ${VIVADO_PATH}.")
    else()
      set(VIVADO_PATH "")
      message(STATUS "Did not find vivado.")
    endif()
  endif()
endif()

#-------------------------------------------------------------------------------
# Clang-Tidy Configuration (for integration tests to check SystemC linting)
#-------------------------------------------------------------------------------

# If clang-tidy hasn't been explicitly disabled, find it.
option(CLANG_TIDY_DISABLE "Disable the clang-tidy lint tests.")
if (CLANG_TIDY_DISABLE)
  message(STATUS "Disabling clang-tidy lint tests.")
else()
  if (EXISTS ${CLANG_TIDY_PATH})
    get_filename_component(CLANG_TIDY_PATH ${CLANG_TIDY_PATH} DIRECTORY)
    message(STATUS "Setting clang-tidy path to ${CLANG_TIDY_PATH}.")
  else()
    # Search for the `clang-tidy` command.
    find_program(CLANG_TIDY_PATH "clang-tidy")
    if(EXISTS ${CLANG_TIDY_PATH})
      # Then strip the filename.
      get_filename_component(CLANG_TIDY_PATH ${CLANG_TIDY_PATH} DIRECTORY)
      message(STATUS "Found clang-tidy at ${CLANG_TIDY_PATH}.")
    else()
      set(CLANG_TIDY_PATH "")
      message(STATUS "Did NOT find clang-tidy.")
    endif()
  endif()
endif()

#-------------------------------------------------------------------------------
# SystemC Configuration
#-------------------------------------------------------------------------------

# If SystemC hasn't been explicitly disabled, find it.
option(SYSTEMC_DISABLE "Disable the systemc tests.")
if (SYSTEMC_DISABLE)
  message(STATUS "Disabling systemc tests.")
else()
  find_file(HAVE_SYSTEMC systemc PATH /usr/include /usr/local/include ${SYSTEMC_PATH})
  if(HAVE_SYSTEMC)
    message(STATUS "Found systemc headers.")
  else()
    set(HAVE_SYSTEMC "")
    message(STATUS "Did NOT find systemc headers.")
  endif()
endif()

#-------------------------------------------------------------------------------
# Quartus Configuration
#-------------------------------------------------------------------------------

# If Quartus hasn't been explicitly disabled, find it.
option(QUARTUS_DISABLE "Disable the Quartus synthesis tests.")
if (QUARTUS_DISABLE)
  message(STATUS "Disabling Quartus tests.")
else()
  if (EXISTS ${QUARTUS_PATH})
    message(STATUS "Setting Quartus path to ${QUARTUS_PATH}.")
  else()
    # Search for Quartus's `quartus` command.
    find_program(QUARTUS_PATH "quartus")
    if(EXISTS ${QUARTUS_PATH})
      # Then strip the filename.
      get_filename_component(QUARTUS_PATH ${QUARTUS_PATH} DIRECTORY)
      message(STATUS "Found Quartus at ${QUARTUS_PATH}.")
    else()
      set(QUARTUS_PATH "")
      message(STATUS "Did not find Quartus.")
    endif()
  endif()
endif()

#-------------------------------------------------------------------------------
# Questa Configuration
#-------------------------------------------------------------------------------

# If Questa hasn't been explicitly disabled, find it.
option(QUESTA_DISABLE "Disable the Questa simulation tests.")
if (QUESTA_DISABLE)
  message(STATUS "Disabling Questa tests.")
else()
  if (EXISTS ${QUESTA_PATH})
    message(STATUS "Setting Questa path to ${QUESTA_PATH}.")
  else()
    # Search for Questa's `vsim` command.
    find_program(QUESTA_PATH "vsim")
    if(EXISTS ${QUESTA_PATH})
      # Then strip the filename.
      get_filename_component(QUESTA_PATH ${QUESTA_PATH} DIRECTORY)
      message(STATUS "Found Questa at ${QUESTA_PATH}.")
    else()
      set(QUESTA_PATH "")
      message(STATUS "Did not find Questa.")
    endif()
  endif()
endif()

#-------------------------------------------------------------------------------
# Yosys Configuration
#-------------------------------------------------------------------------------

# If Yosys hasn't been explicitly disabled, find it.
option(YOSYS_DISABLE "Disable the yosys tests.")
if (YOSYS_DISABLE)
  message(STATUS "Disabling yosys tests.")
else()
  find_program(YOSYS_PATH "yosys")
  if(EXISTS ${YOSYS_PATH})
    message(STATUS "Found yosys at ${YOSYS_PATH}.")
  else()
    set(YOSYS_PATH "")
    message(STATUS "Did not find yosys.")
  endif()
endif()

#-------------------------------------------------------------------------------
# Icarus Verilog Configuration
#-------------------------------------------------------------------------------

# If Icarus Verilog hasn't been explicitly disabled, find it.
option(IVERILOG_DISABLE "Disable the Icarus Verilog tests.")
if (IVERILOG_DISABLE)
  message(STATUS "Disabling Icarus Verilog tests.")
else()
  find_program(IVERILOG_PATH "iverilog")
  if(EXISTS ${IVERILOG_PATH})
    # Find iverilog version.
    execute_process(COMMAND ${IVERILOG_PATH} -V
      OUTPUT_VARIABLE IVERILOG_VERSION)

    string(REGEX MATCH "Icarus Verilog version (([0-9]+)\.([0-9]+)) \.*"
      MATCH ${IVERILOG_VERSION})

    if (${CMAKE_MATCH_1} LESS 11.0)
      message(FATAL_ERROR "CIRCT only supports Icarus Verilog version 11.0 and up. \
                           Found version: ${CMAKE_MATCH_1}. You can disable \
                           the Icarus Verilog tests with '-DIVERILOG_DISABLE=ON'.")
      set(IVERILOG_PATH "")
    endif()
    message(STATUS "Found iverilog at ${IVERILOG_PATH}.")
  else()
    set(IVERILOG_PATH "")
    message(STATUS "Did not find iverilog.")
  endif()
endif()

#-------------------------------------------------------------------------------
# capnp Configuration
#-------------------------------------------------------------------------------

# If capnp hasn't been explicitly disabled, find it.
option(CAPNP_DISABLE "Disable Cap'nProto (needed for cosimulation).")
if (CAPNP_DISABLE)
  message (STATUS "Disabling Cap'nProto.")
else()
  if(DEFINED CAPNP_PATH)
    set(ENV{PKG_CONFIG_PATH}
      "${CAPNP_PATH}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    find_package(CapnProto CONFIG PATHS ${CAPNP_PATH})
  else()
    set(ENV{PKG_CONFIG_PATH}
      "${CMAKE_CURRENT_SOURCE_DIR}/ext/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    find_package(CapnProto CONFIG PATHS "${CMAKE_CURRENT_SOURCE_DIR}/ext")
  endif()

  if(CapnProto_FOUND)
    message(STATUS "Found Cap'nProto at ${CapnProto_DIR}.")
    set(CMAKE_INSTALL_RPATH ${capnp_LIBDIR})
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

    # Some of our CMake code which is only used in Capnp builds uses newer CMake
    # features. Since most compilations are not capnp builds, only require the
    # new version for those builds.
    if(CMAKE_VERSION VERSION_LESS "3.23.0")
      message(SEND_ERROR
          "Building CIRCT with Cap'nProto features enabled uses CMake features "
          "that require at least version 3.23. Recommend upgrading to 3.23+ "
          "for full support. Detected current version: ${CMAKE_VERSION}. If you "
          "do not need ESI cosim features, you can use the CAPNP_DISABLE option.")
    endif()
  endif()
endif()

#-------------------------------------------------------------------------------
# OR-Tools Configuration
#-------------------------------------------------------------------------------

option(OR_TOOLS_DISABLE "Disable OR-Tools.")
if (OR_TOOLS_DISABLE)
  message(STATUS "Disabling OR-Tools.")
else()
  if(DEFINED OR_TOOLS_PATH)
    list(APPEND CMAKE_PREFIX_PATH ${OR_TOOLS_PATH})
  endif()

  list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ext)

  find_package(ortools CONFIG)

  if (ortools_FOUND)
    get_filename_component(ortools_CMAKEDIR ${ortools_DIR} DIRECTORY)
    get_filename_component(ortools_LIBDIR ${ortools_CMAKEDIR} DIRECTORY)
    list(APPEND CMAKE_INSTALL_RPATH ${ortools_LIBDIR})
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  endif()
endif()

#-------------------------------------------------------------------------------
# llhd-sim Configuration
#-------------------------------------------------------------------------------

if(NOT WIN32)
  option(CIRCT_LLHD_SIM_ENABLED "Enables LLHD sim." ON)
else()
  option(CIRCT_LLHD_SIM_ENABLED "Enables LLHD sim." OFF)
endif()

if(CIRCT_LLHD_SIM_ENABLED)
  message(STATUS "llhd-sim is enabled.")
else()
  message(STATUS "llhd-sim is disabled.")
endif()

llvm_canonicalize_cmake_booleans(CIRCT_LLHD_SIM_ENABLED)

#-------------------------------------------------------------------------------
# circt-lec Configuration
#-------------------------------------------------------------------------------

# If circt-lec hasn't been explicitly disabled, find it.
option(CIRCT_LEC_DISABLE "Disable the Logical Equivalence Checker" OFF)
if(CIRCT_LEC_DISABLE)
  message(STATUS "Disabling circt-lec")
else()
  if(Z3_DIR)
    # Search and load the package configuration file in the specified directory.
    find_package(Z3 CONFIG REQUIRED PATHS ${Z3_DIR} NO_DEFAULT_PATH)
    if(Z3_FOUND)
      # Report the found library location and version
      # similarly to LLVM's `FindZ3` CMake module.
      get_target_property(Z3_LIB_LOCATION z3::libz3 IMPORTED_LOCATION_DEBUG)
      message(STATUS "Found Z3: ${Z3_LIB_LOCATION} (found version \"${Z3_VERSION_STRING}\")")
    endif()
  else()
    # Attempt initialising Z3 according to LLVM's `FindZ3` CMake module.
    find_package(Z3)
  endif()
  
  if(Z3_FOUND)
    SET(CIRCT_LEC_Z3_VER 4.8.11)
    if(Z3_VERSION_STRING VERSION_LESS ${CIRCT_LEC_Z3_VER})
      message(WARNING "Cannot build circt-lec with outdated Z3 version ${Z3_VERSION_STRING}, requires ${CIRCT_LEC_Z3_VER}.")
    else()
      message(STATUS "Z3 identified as a logical backend.")
      # Signal to proceed building circt-lec.
      set(CIRCT_LEC_ENABLED ON)
    endif()
  endif()
endif()

#-------------------------------------------------------------------------------
# Python Configuration
#-------------------------------------------------------------------------------

option(CIRCT_BINDINGS_PYTHON_ENABLED "Enables CIRCT Python bindings." OFF)

if(CIRCT_BINDINGS_PYTHON_ENABLED)
  message(STATUS "CIRCT Python bindings are enabled.")
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
else()
  message(STATUS "CIRCT Python bindings are disabled.")
  # Lookup python either way as some integration tests use python without the
  # bindings
  find_package(Python3)
  if(Python3_FOUND)
    message(STATUS "Found python at ${Python3_EXECUTABLE}")
  endif()
endif()

#-------------------------------------------------------------------------------
# Tcl bindings
#-------------------------------------------------------------------------------
option(CIRCT_BINDINGS_TCL_ENABLED "Enables CIRCT Tcl bindings." OFF)

llvm_canonicalize_cmake_booleans(CIRCT_BINDINGS_TCL_ENABLED)
if(CIRCT_BINDINGS_TCL_ENABLED)
  message(STATUS "CIRCT Tcl bindings are enabled")
  find_package(TCL 8.6 REQUIRED)
  find_package(TclStub 8.6 REQUIRED)
  message(STATUS "Found TCL include path: ${TCL_INCLUDE_PATH}")
  message(STATUS "Found TCL library: ${TCL_LIBRARY}")
  message(STATUS "Found TCL executable: ${TCL_TCLSH}")
endif()

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

add_subdirectory(include/circt)
add_subdirectory(lib)
add_subdirectory(tools)
if (CIRCT_GTEST_AVAILABLE)
  add_subdirectory(unittests)
endif()
add_subdirectory(test)
add_subdirectory(integration_test)
add_subdirectory(frontends)

option(CIRCT_INCLUDE_DOCS "Generate build targets for the CIRCT docs.")
if (CIRCT_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()

install(DIRECTORY include/circt include/circt-c
  DESTINATION include
  COMPONENT circt-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "*.sv"
  PATTERN "LICENSE.TXT"
  )

install(DIRECTORY ${CIRCT_INCLUDE_DIR}/circt ${CIRCT_INCLUDE_DIR}/circt-c
  DESTINATION include
  COMPONENT circt-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.gen"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "config.h" EXCLUDE
  )

if (NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(install-circt-headers
                           DEPENDS circt-headers
                           COMPONENT circt-headers)
endif()

add_subdirectory(cmake/modules)

# Set RPATH to $ORIGIN on all targets.
function(set_rpath_all_targets dir)
  get_property(subdirectories DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
  foreach(subdir ${subdirectories})
    set_rpath_all_targets(${subdir})
  endforeach()

  get_directory_property(LCL_TARGETS DIRECTORY ${dir} BUILDSYSTEM_TARGETS)
  set_property(TARGET ${LCL_TARGETS} PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
endfunction()

option(STANDALONE_INSTALL "Create an 'install' for packaging which doesn't \
         require installation" off)
if (STANDALONE_INSTALL)
  message(STATUS "Setting an $ORIGIN-based RPATH on all executables")
  set_rpath_all_targets(${CMAKE_CURRENT_SOURCE_DIR})
endif()
