# ===- CMakeLists.txt - PyCDE integration tests cmake ---------*- cmake -*-===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===//


set(PYCDE_INTEGRATION_TEST_DEPENDS
  FileCheck count not split-file
  circt-rtl-sim
  esi-cosim-runner
  PyCDE
  mlir-opt
  hlstool
  CIRCTPythonModules)

# If ESI Cosim is available to build then enable its tests.
if (TARGET EsiCosimDpiServer)
  list(APPEND PYCDE_INTEGRATION_TEST_DEPENDS EsiCosimDpiServer)
  get_property(ESI_COSIM_LIB_DIR TARGET EsiCosimDpiServer PROPERTY LIBRARY_OUTPUT_DIRECTORY)
  set(ESI_COSIM_PATH ${ESI_COSIM_LIB_DIR}/libEsiCosimDpiServer.so)
endif()

set(PYCDE_INTEGRATION_TIMEOUT 60) # Set a 60s timeout on individual tests.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  PyCDE_INTEGRATION_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-pycde-integration "Running the PyCDE integration tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${PYCDE_INTEGRATION_TEST_DEPENDS}
  )
set_target_properties(check-pycde-integration PROPERTIES FOLDER "PyCDEIntegrationTests")
