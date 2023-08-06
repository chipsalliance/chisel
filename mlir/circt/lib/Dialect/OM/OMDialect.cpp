//===- OMDialect.cpp - Object Model dialect definition --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Object Model dialect definition.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/OM/OMOps.h"

#include "circt/Dialect/OM/OMDialect.cpp.inc"

void circt::om::OMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/OM/OM.cpp.inc"
      >();

  registerTypes();
  registerAttributes();
}
