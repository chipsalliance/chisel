//===- SeqAttributes.cpp - Implement Seq attributes -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Seq/SeqAttributes.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace seq;

#include "circt/Dialect/Seq/SeqEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/Seq/SeqAttributes.cpp.inc"

void SeqDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/Seq/SeqAttributes.cpp.inc"
      >();
}
