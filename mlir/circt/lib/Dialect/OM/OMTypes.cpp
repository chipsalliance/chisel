//===- OMTypes.cpp - Object Model type definitions ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Object Model type definitions.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OM/OMTypes.h"
#include "circt/Dialect/OM/OMDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/OM/OMTypes.cpp.inc"

void circt::om::OMDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/OM/OMTypes.cpp.inc"
      >();
}
