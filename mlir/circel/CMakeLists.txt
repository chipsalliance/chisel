cmake_minimum_required(VERSION 3.13.4)

set(CMAKE_CXX_STANDARD 17)

set(PROJECT_NAME CIRCEL)

# LLVM
add_definitions(${LLVM_DEFINITIONS})

# MLIR Setup
set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include)
set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include)
include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
set(MLIR_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/mlir/include)
include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})
set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)

# CIRCT
set(LLVM_EXTERNAL_CIRCT_SOURCE_DIR CACHE STRING "")
set(CIRCT_MAIN_SRC_DIR ${LLVM_EXTERNAL_CIRCT_SOURCE_DIR})
set(CIRCT_INCLUDE_DIR ${CIRCT_MAIN_SRC_DIR}/include)
include_directories(SYSTEM ${CIRCT_INCLUDE_DIR})
set(CIRCT_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/circt/include)
include_directories(SYSTEM ${CIRCT_TABLEGEN_OUTPUT_DIR})

# llvm-lit (FileCheck)
set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

# CIRCEL directories (used by tests)
set(CIRCEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CIRCEL_BINARY_DIR ${CMAKE_BINARY_DIR}/bin)
set(CIRCEL_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# Subdirectories
add_subdirectory(lib)
add_subdirectory(test)
