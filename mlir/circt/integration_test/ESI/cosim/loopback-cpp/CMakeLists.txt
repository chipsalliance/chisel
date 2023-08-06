cmake_minimum_required(VERSION 3.13.4)
project(loopback_test)

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

if(NOT DEFINED CAPNP_SCHEMA)
  message(FATAL_ERROR "CAPNP_SCHEMA must be defined.")
endif()

message(STATUS "CIRCT_DIR= ${CIRCT_DIR}")
set(ESI_CPP_API "ESISystem.h")

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
    LOOPBACK_SRCS
    LOOPBACK_HDRS
    ${CAPNPC_SRC_PREFIX}/${CAPNP_SCHEMA_BASENAME}
)

message(STATUS "CAPNP_OUTDIR= ${CAPNP_OUTDIR}")
add_executable(loopback_test
  loopback.cpp
  ${LOOPBACK_SRCS}
  ${LOOPBACK_HDRS}
)
target_link_libraries(loopback_test
  ${CAPNP_LIBRARIES}
  refl-cpp
)
target_include_directories(loopback_test PUBLIC
  # Include the ESI C++ runtime headers directly from within the CIRCT
  # source tree.
  ${CIRCT_DIR}/lib/Dialect/ESI/runtime/cpp/include)

message("LOOPBACK_SRCS: ${LOOPBACK_SRCS}")
message("LOOPBACK_HDRS: ${LOOPBACK_HDRS}")

target_include_directories(
    loopback_test
    PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CAPNP_INCLUDE_DIRS}
    ${CAPNP_OUTDIR}
    ${CIRCT_DIR}/include
)

target_compile_definitions(
    loopback_test
    PUBLIC
    -DESI_COSIM_CAPNP_H=\"${CAPNPC_OUTPUT_DIR}/schema.capnp.h\"
)
