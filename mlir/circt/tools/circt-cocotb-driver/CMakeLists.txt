# ===- CMakeLists.txt - Cocotb driver cmake -------------------*- cmake -*-===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===//

# The Python script requires that it be configured.
configure_file("circt-cocotb-driver.py.in" "${CIRCT_TOOLS_DIR}/circt-cocotb-driver.py")
add_custom_target(circt-cocotb-driver
  SOURCES "${CIRCT_TOOLS_DIR}/circt-cocotb-driver.py")
