# ===- CMakeLists.txt - python file splitter/executer ---------*- cmake -*-===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===//

set(SOURCES py-split-input-file.py)

foreach(file IN ITEMS ${SOURCES})
  configure_file(${file}.in ${CIRCT_TOOLS_DIR}/${file})
  list(APPEND OUTPUTS ${CIRCT_TOOLS_DIR}/${file})
endforeach()

add_custom_target(py-split-input-file SOURCES ${OUTPUTS})
