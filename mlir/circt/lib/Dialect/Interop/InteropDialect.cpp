//===- InteropDialect.cpp - Implement the Interop dialect -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Interop dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Interop/InteropOps.h"

using namespace circt;
using namespace circt::interop;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void InteropDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Interop/Interop.cpp.inc"
      >();
}

#include "circt/Dialect/Interop/InteropDialect.cpp.inc"
