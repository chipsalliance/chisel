llvm_canonicalize_cmake_booleans(
  CIRCT_BINDINGS_PYTHON_ENABLED
  )

set(CIRCT_INTEGRATION_TEST_DEPENDS
  FileCheck count not split-file
  circt-opt
  circt-translate
  circt-rtl-sim
  esi-cosim-runner
  firtool
  hlstool
  handshake-runner
  )

# If Python bindings are available to build then enable the tests.
if(CIRCT_BINDINGS_PYTHON_ENABLED)
  list(APPEND CIRCT_INTEGRATION_TEST_DEPENDS CIRCTPythonModules)
endif()

# If ESI Cosim is available to build then enable its tests.
if (TARGET EsiCosimDpiServer)
  list(APPEND CIRCT_INTEGRATION_TEST_DEPENDS EsiCosimDpiServer)
  get_property(ESI_COSIM_LIB_DIR TARGET EsiCosimDpiServer PROPERTY LIBRARY_OUTPUT_DIRECTORY)
  set(ESI_COSIM_PATH ${ESI_COSIM_LIB_DIR}/libEsiCosimDpiServer.so)
endif()

# Enable circt-lec tests if it is built.
if(CIRCT_LEC_ENABLED)
  list(APPEND CIRCT_INTEGRATION_TEST_DEPENDS circt-lec)
endif()

set(CIRCT_INTEGRATION_TIMEOUT 60) # Set a 60s timeout on individual tests.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  INTEGRATION_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-circt-integration "Running the CIRCT integration tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CIRCT_INTEGRATION_TEST_DEPENDS}
  )
set_target_properties(check-circt-integration PROPERTIES FOLDER "IntegrationTests")

add_lit_testsuites(CIRCT_INTEGRATION ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${CIRCT_INTEGRATION_TEST_DEPS}
)
