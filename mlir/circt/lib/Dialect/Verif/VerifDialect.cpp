//===- VerifDialect.cpp - Verif dialect implementation --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace verif;

void VerifDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Verif/Verif.cpp.inc"
      >();
}

Operation *VerifDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  if (auto intType = dyn_cast<IntegerType>(type))
    if (auto attrValue = dyn_cast<IntegerAttr>(value))
      return builder.create<hw::ConstantOp>(loc, type, attrValue);
  return nullptr;
}

#include "circt/Dialect/Verif/VerifDialect.cpp.inc"
