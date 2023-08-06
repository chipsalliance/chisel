//===- LLHDDialect.cpp - Implement the LLHD dialect -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLHD dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/IR/LLHDTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

using namespace circt;
using namespace circt::llhd;

//===----------------------------------------------------------------------===//
// LLHDDialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
/// This class defines the interface for handling inlining with LLHD operations.
struct LLHDInlinerInterface : public mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All operations within LLHD can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  bool isLegalToInline(Region *, Region *src, bool, IRMapping &) const final {
    // Don't inline processes and entities
    return !isa<llhd::ProcOp>(src->getParentOp()) &&
           !isa<llhd::EntityOp>(src->getParentOp());
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// LLHD Dialect
//===----------------------------------------------------------------------===//

void LLHDDialect::initialize() {
  registerTypes();
  registerAttributes();

  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/LLHD/IR/LLHD.cpp.inc"
      >();

  addInterfaces<LLHDInlinerInterface>();
}

Operation *LLHDDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  if (auto timeAttr = value.dyn_cast<TimeAttr>())
    return builder.create<llhd::ConstantTimeOp>(loc, type, timeAttr);

  if (auto intAttr = value.dyn_cast<IntegerAttr>())
    return builder.create<hw::ConstantOp>(loc, type, intAttr);

  return nullptr;
}

#include "circt/Dialect/LLHD/IR/LLHDDialect.cpp.inc"
