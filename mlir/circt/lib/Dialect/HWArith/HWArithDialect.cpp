//===- HWArithDialect.cpp - Implement the HWArith dialect -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the HWArith dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HWArith/HWArithDialect.h"
#include "circt/Dialect/HWArith/HWArithOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace circt;
using namespace circt::hwarith;

//===----------------------------------------------------------------------===//
// HWArithDialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct HWArithInlinerInterface : public mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;
  // Operations in the hwarith dialect are always legal to inline since they are
  // pure.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void HWArithDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/HWArith/HWArith.cpp.inc"
      >();

  // Register interface implementations
  addInterfaces<HWArithInlinerInterface>();
}

// Provide implementations for the enums we use.
#include "circt/Dialect/HWArith/HWArithEnums.cpp.inc"

#include "circt/Dialect/HWArith/HWArithDialect.cpp.inc"
