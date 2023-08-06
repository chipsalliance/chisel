//===- MooreDialect.cpp - Implement the Moore dialect ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Moore dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MIROps.h"

using namespace circt;
using namespace circt::moore;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void MooreDialect::initialize() {
  // Register types.
  registerTypes();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Moore/Moore.cpp.inc"
      >();
}

#include "circt/Dialect/Moore/MooreDialect.cpp.inc"
