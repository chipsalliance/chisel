add_subdirectory(lib)
add_subdirectory(CAPI)
add_subdirectory(Dialect)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
  )

set(CIRCT_TEST_DEPENDS
  FileCheck count not
  split-file
  arcilator
  circt-capi-ir-test
  circt-capi-om-test
  circt-as
  circt-dis
  circt-opt
  circt-translate
  circt-reduce
  esi-tester
  handshake-runner
  firtool
  hlstool
  om-linker
  )

if (CIRCT_GTEST_AVAILABLE)
  list(APPEND CIRCT_TEST_DEPENDS CIRCTUnitTests)
endif()

if(CIRCT_LLHD_SIM_ENABLED)
  list(APPEND CIRCT_TEST_DEPENDS llhd-sim)
  list(APPEND CIRCT_TEST_DEPENDS circt-llhd-signals-runtime-wrappers)
endif()

add_lit_testsuite(check-circt "Running the CIRCT regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CIRCT_TEST_DEPENDS}
  )
set_target_properties(check-circt PROPERTIES FOLDER "Tests")

add_lit_testsuites(CIRCT ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CIRCT_TEST_DEPENDS}
  )
