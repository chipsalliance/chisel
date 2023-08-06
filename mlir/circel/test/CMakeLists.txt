add_subdirectory(Bindings)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(CIRCEL_TEST_DEPENDS
  FileCheck count not
  mlir-bindings-support-test
  mlir-bindings-swift-test
  )

add_lit_testsuite(check-circel "Running the CIRCEL regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CIRCEL_TEST_DEPENDS}
  )
set_target_properties(check-circel PROPERTIES FOLDER "Tests")

add_lit_testsuites(CIRCEL ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${CIRCEL_TEST_DEPENDS}
)
