//===- SystemCAttributes.cpp - SystemC attribute code defs ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation for SystemC attributes.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SystemC/SystemCAttributes.h"
#include "circt/Dialect/SystemC/SystemCDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt::systemc;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/SystemC/SystemCAttributes.cpp.inc"
#include "circt/Dialect/SystemC/SystemCEnums.cpp.inc"

void SystemCDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/SystemC/SystemCAttributes.cpp.inc"
      >();
}
