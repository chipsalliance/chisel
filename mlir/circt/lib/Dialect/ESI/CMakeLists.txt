##===- CMakeLists.txt - ESI dialect code ----------------------*- cmake -*-===//
##
## Implementation files the ESI dialect. Optionally include the Capnp
## dependent code.
##
##===----------------------------------------------------------------------===//

set(srcs
  ESIDialect.cpp
  ESIFolds.cpp
  ESIOps.cpp
  ESIPasses.cpp
  ESIServices.cpp
  ESIStdServices.cpp
  ESITranslations.cpp
  ESITypes.cpp
  Passes/ESIEmitCollateral.cpp
  Passes/ESILowerPhysical.cpp
  Passes/ESILowerPorts.cpp
  Passes/ESILowerToHW.cpp
  Passes/ESILowerTypes.cpp
  Passes/ESICleanMetadata.cpp

  APIUtilities.cpp

  # C++ API
  Passes/CPPAPI/ESIAddCPPAPI.cpp
  Passes/CPPAPI/ESIEmitCPPAPI.cpp
  Passes/CPPAPI/CPPAPI.cpp
)

set(ESI_RUNTIME_DIR ${CIRCT_BINARY_DIR}/include/circt/Dialect/ESI)

set(ESI_LinkLibs
  CIRCTSupport
  CIRCTComb
  CIRCTMSFT
  CIRCTMSFTTransforms
  CIRCTSV
  CIRCTHW
  MLIRIR
  MLIRTransforms
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRTranslateLib
)

set(ESI_Deps
  ${ESI_LinkLibs}
  MLIRESITransformsIncGen
  MLIRESIInterfacesIncGen
)

if(CapnProto_FOUND)
  option(ESI_CAPNP "Enable ESI Capnp features" ON)
  list(APPEND srcs capnp/Schema.cpp)
  list(APPEND ESI_LinkLibs CapnProto::capnp CapnProto::capnpc)
endif()

add_circt_dialect_library(CIRCTESI
  ${srcs}

  DEPENDS
  MLIRESITransformsIncGen
  MLIRESIEnumsIncGen
  ${ESI_Deps}

  LINK_COMPONENTS
  Core
  Support

  LINK_LIBS PUBLIC
  ${ESI_LinkLibs}
  )

# Create a package of ESI collateral for any compiler to include with its
# distribution.
add_library(esi-collateral INTERFACE)

# A function which copies a file from the source tree to the build tree.
# This is an alternative to configure_file to ensure that the file is copied
# on every build (not just during CMake configure).
function(copy_esi_runtime FILENAME)
  # An optional "OUTDIR" argument can be passed to specify the relative location
  # within the ESI_RUNTIME_DIR. If not specified, the file will be copied to the
  # root of the ESI_RUNTIME_DIR.
  cmake_parse_arguments(ARG "" "OUTDIR" "" ${ARGN})
  if (NOT ARG_OUTDIR)
    set(ARG_OUTDIR "")
  else()
    set(ARG_OUTDIR "${ARG_OUTDIR}/")
  endif()

  get_filename_component(DST ${FILENAME} NAME)
  get_filename_component(SRC ${FILENAME} REALPATH)
  add_custom_command(TARGET CIRCTESI PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E
    copy ${SRC} ${ESI_RUNTIME_DIR}/${ARG_OUTDIR}${DST})
  target_sources(esi-collateral PUBLIC FILE_SET HEADERS
    BASE_DIRS "."
    FILES "${FILENAME}")
endfunction()


if (ESI_CAPNP)
  copy_esi_runtime(cosim/Cosim_DpiPkg.sv OUTDIR cosim)
  copy_esi_runtime(cosim/Cosim_Endpoint.sv OUTDIR cosim)
  install(FILES ${CapnProto_capnp_IMPORTED_LOCATION} DESTINATION lib)
  install(FILES ${CapnProto_capnpc_IMPORTED_LOCATION} DESTINATION lib)
  install(FILES ${CapnProto_kj_IMPORTED_LOCATION} DESTINATION lib)

  # These are not Capnp files, but they use a feature from a newer version of
  # CMake. Adding them here as a mitigation.
  # TODO: Figure out what to do longer term.
  copy_esi_runtime(ESIPrimitives.sv OUTDIR)
  copy_esi_runtime(runtime/cpp/include/esi/esi.h OUTDIR cpp/include/esi)
  copy_esi_runtime(runtime/cpp/include/esi/backends/capnp.h OUTDIR cpp/include/esi/backends)

  target_compile_definitions(obj.CIRCTESI PRIVATE CAPNP)
  target_link_libraries(obj.CIRCTESI CapnProto::capnp CapnProto::capnpc)
endif()

add_subdirectory(cosim)
