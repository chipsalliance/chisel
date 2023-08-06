# ===- CMakeLists.txt - Simulation driver cmake ---------------*- cmake -*-===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===//
#
# Configure and copy a script to run ESI cosimulation tests.
#
# ===-----------------------------------------------------------------------===//

if (TARGET EsiCosimDpiServer)
  list(APPEND CIRCT_INTEGRATION_TEST_DEPENDS EsiCosimDpiServer CIRCTESI)
  get_property(ESI_COSIM_LIB_DIR TARGET EsiCosimDpiServer PROPERTY LIBRARY_OUTPUT_DIRECTORY)
  set(ESI_COSIM_PATH ${ESI_COSIM_LIB_DIR}/libEsiCosimDpiServer.so)
endif()

set(SOURCES esi-cosim-runner.py esi_cosim.py)
foreach(file IN ITEMS ${SOURCES})
  configure_file(${file}.in ${CIRCT_TOOLS_DIR}/${file})
  list(APPEND OUTPUTS ${CIRCT_TOOLS_DIR}/${file})
endforeach()
add_custom_target(esi-cosim-runner SOURCES ${OUTPUTS})
