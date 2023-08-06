//===- LTLDialect.cpp - LTL dialect implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/LTL/LTLTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace ltl;

void LTLDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/LTL/LTLTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/LTL/LTL.cpp.inc"
      >();
}

#include "circt/Dialect/LTL/LTLDialect.cpp.inc"
#include "circt/Dialect/LTL/LTLEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/LTL/LTLTypes.cpp.inc"
