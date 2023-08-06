//===- FieldRef.cpp - Field Refs  -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines FieldRef and helpers for them.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/FieldRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"

using namespace circt;

Operation *FieldRef::getDefiningOp() const {
  if (auto *op = value.getDefiningOp())
    return op;
  return value.cast<BlockArgument>().getOwner()->getParentOp();
}

