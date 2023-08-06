# ===- CMakeLists.txt - include frontends if they are enabled --*- cmake -*-===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===//

set (CIRCT_KNOWN_FRONTENDS
  PyCDE)
option(CIRCT_ENABLE_FRONTENDS "")
foreach(proj ${CIRCT_KNOWN_FRONTENDS})
  string(TOUPPER "${proj}" upper_proj)
  string(REGEX REPLACE "-" "_" upper_proj ${upper_proj})
  if ("${proj}" IN_LIST CIRCT_ENABLE_FRONTENDS)
    message(STATUS "${proj} project is enabled")
    set(PROJ_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${proj}")
    if(NOT EXISTS "${PROJ_DIR}" OR NOT IS_DIRECTORY "${PROJ_DIR}")
      message(FATAL_ERROR "CIRCT_ENABLE_FRONTENDS requests ${proj} but directory not found: ${PROJ_DIR}")
    endif()
    add_subdirectory(${PROJ_DIR})
  else()
    message(STATUS "${proj} project is disabled")
  endif()
endforeach()
