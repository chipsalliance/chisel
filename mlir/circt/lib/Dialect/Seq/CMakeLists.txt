##===- CMakeLists.txt - build definitions for Seq -------------*- cmake -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//
##
##
##===----------------------------------------------------------------------===//

add_circt_dialect_library(CIRCTSeq
  SeqAttributes.cpp
  SeqDialect.cpp
  SeqOpInterfaces.cpp
  SeqOps.cpp
  SeqTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Seq

  DEPENDS
  CIRCTHW
  CIRCTSV
  MLIRSeqIncGen
  MLIRSeqEnumsIncGen
  MLIRSeqAttributesIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSV
  MLIRIR
  MLIRPass
  MLIRTransforms
)

add_subdirectory(Transforms)
