cmake_minimum_required(VERSION 3.13.4)
project(esi_ram_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TODO: most of this stuff should be moved to a .cmake file that is included.

# fetch https://github.com/veselink1/refl-cpp
include(FetchContent)
FetchContent_Declare(
  refl-cpp
  GIT_REPOSITORY https://github.com/veselink1/refl-cpp
  GIT_TAG        v0.12.4
)
FetchContent_MakeAvailable(refl-cpp)

# Assert that CIRCT_DIR is defined.
if(NOT DEFINED CIRCT_DIR)
  message(FATAL_ERROR "CIRCT_DIR must be defined.")
endif()

if(NOT DEFINED PYCDE_OUT_DIR)
  message(FATAL_ERROR "PYCDE_OUT_DIR must be defined.")
endif()

message(STATUS "CIRCT_DIR= ${CIRCT_DIR}")
message(STATUS "PYCDE_OUT_DIR= ${PYCDE_OUT_DIR}")

set(CAPNP_SCHEMA "${PYCDE_OUT_DIR}/hw/schema.capnp")
set(ESI_CPP_API "${PYCDE_OUT_DIR}/hw/ESISystem.h")
set(ESI_HW_INCLUDE_DIR "${PYCDE_OUT_DIR}/hw")

# Ensure that the above files are present
if(NOT EXISTS ${CAPNP_SCHEMA})
  message(FATAL_ERROR "CAPNP_SCHEMA not found: ${CAPNP_SCHEMA}")
endif()

if(NOT EXISTS ${ESI_CPP_API})
  message(FATAL_ERROR "ESI_CPP_API not found: ${ESI_CPP_API}")
endif()


if(DEFINED CAPNP_PATH)
  set(ENV{PKG_CONFIG_PATH}
    "${CAPNP_PATH}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
  find_package(CapnProto CONFIG PATHS ${CAPNP_PATH})
  else()
  set(ENV{PKG_CONFIG_PATH}
    "${CIRCT_DIR}/ext/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
  find_package(CapnProto CONFIG PATHS "${CIRCT_DIR}/ext")
endif()

if (NOT CapnProto_FOUND)
  message(FATAL_ERROR "Cap'n Proto not found.")
endif()

# Move schema to the build directory - required by capnp_generate_cpp.
set(CAPNPC_SRC_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/capnp_generated)
set(CAPNPC_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/capnp_generated)
file(COPY ${CAPNP_SCHEMA} DESTINATION ${CAPNPC_OUTPUT_DIR})
get_filename_component(CAPNP_SCHEMA_BASENAME ${CAPNP_SCHEMA} NAME)
set(COPIED_CAPNP_SCHEMA ${CAPNP_OUTDIR}/${CAPNP_SCHEMA_BASENAME})
capnp_generate_cpp(
    ESI_RAM_SRCS
    ESI_RAM_HDRS
    ${CAPNPC_SRC_PREFIX}/${CAPNP_SCHEMA_BASENAME}
)

message(STATUS "CAPNP_OUTDIR= ${CAPNP_OUTDIR}")
add_executable(esi_ram_test
  esi_ram.cpp
  ${ESI_RAM_SRCS}
  ${ESI_RAM_HDRS}
)
target_link_libraries(esi_ram_test
  ${CAPNP_LIBRARIES}
  refl-cpp
)
target_include_directories(esi_ram_test PUBLIC
  # Include the copied ESI C++ runtime headers.
  ${PYCDE_OUT_DIR}/runtime/cpp/include)

message("ESI_RAM_SRCS: ${ESI_RAM_SRCS}")
message("ESI_RAM_HDRS: ${ESI_RAM_HDRS}")

target_include_directories(
    esi_ram_test
    PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CAPNP_INCLUDE_DIRS}
    ${CAPNP_OUTDIR}
    ${CIRCT_DIR}/include
    ${ESI_HW_INCLUDE_DIR}
)

target_compile_definitions(
    esi_ram_test
    PUBLIC
    -DESI_COSIM_CAPNP_H=\"${CAPNPC_OUTPUT_DIR}/schema.capnp.h\"
)
