# ===- CMakeLists.txt - PyCDE top level cmake ------------------*- cmake -*-===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===//

llvm_canonicalize_cmake_booleans(
  CIRCT_BINDINGS_PYTHON_ENABLED
  )

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR "PyCDE cannot be built as separate project")
endif()

if (NOT CIRCT_BINDINGS_PYTHON_ENABLED)
  message(FATAL_ERROR "PyCDE requires that CIRCT python bindings be enabled")
endif()

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(integration_test)
