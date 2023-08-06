# ===- CMakeLists.txt - Simulation driver cmake ---------------*- cmake -*-===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===//

# The Python script requires that it be configured.
configure_file("circt-rtl-sim.py.in" "${CIRCT_TOOLS_DIR}/circt-rtl-sim.py")
add_custom_target(circt-rtl-sim
  SOURCES "${CIRCT_TOOLS_DIR}/circt-rtl-sim.py"
  DEPENDS circt-std-sim-drivers)

# Separate out the drviers (which don't need to be configured) since they are
# useful beyond just circt-rtl-sim.
set(SOURCES
  driver.cpp
  driver.sv
)
foreach(SOURCE ${SOURCES})
  get_filename_component(DST ${SOURCE} NAME)
  get_filename_component(SRC ${SOURCE} REALPATH)
  list(APPEND ABS_SOURCES ${SRC})
  add_custom_command(TARGET circt-rtl-sim PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E
    copy ${SRC} ${CIRCT_TOOLS_DIR}/${DST})
endforeach()
add_library(circt-std-sim-drivers INTERFACE)
set_target_properties(circt-std-sim-drivers PROPERTIES
  PUBLIC_HEADER "${ABS_SOURCES}")
