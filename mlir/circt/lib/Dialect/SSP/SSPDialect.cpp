//===- SSPDialect.cpp - SSP dialect implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SSP (static scheduling problem) dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SSP/SSPDialect.h"
#include "circt/Dialect/SSP/SSPOps.h"

using namespace circt;
using namespace circt::ssp;

void SSPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/SSP/SSP.cpp.inc"
      >();
  registerAttributes();
}

#include "circt/Dialect/SSP/SSPDialect.cpp.inc"
