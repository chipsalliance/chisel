##===- CMakeLists.txt - Core cosim DPI library ----------------*- cmake -*-===//
##
## Define the cosim DPI library if it's enabled.
##
##===----------------------------------------------------------------------===//

if(ESI_COSIM)
  if (MSVC)
    string(REPLACE "/EHs-c-" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHa")
  else ()
    string(REPLACE "-fno-exceptions" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  endif ()

  add_library(EsiCosimDpiServer SHARED
    DpiEntryPoints.cpp
    Server.cpp
    Endpoint.cpp)

  set_target_properties(EsiCosimDpiServer
      PROPERTIES
          LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
          RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
          CXX_VISIBILITY_PRESET "default"
  )
  add_dependencies(EsiCosimDpiServer EsiCosimCapnp MtiPli)
  target_link_libraries(EsiCosimDpiServer PRIVATE
      CapnProto::kj CapnProto::kj-async CapnProto::kj-gzip
      CapnProto::capnp CapnProto::capnp-rpc 
      MtiPli EsiCosimCapnp)

  target_include_directories(EsiCosimDpiServer PRIVATE ${CAPNPC_OUTPUT_DIR})
  target_include_directories(EsiCosimDpiServer PRIVATE ${CAPNP_INCLUDE_DIRS})
  target_include_directories(EsiCosimDpiServer PRIVATE ${CIRCT_INCLUDE_DIR})
endif()
