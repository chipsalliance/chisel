//===- BackedgeBuilder.cpp - Support for building backedges ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provide support for building backedges.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace circt;

Backedge::Backedge(mlir::Operation *op) : value(op->getResult(0)) {}

void Backedge::setValue(mlir::Value newValue) {
  assert(value.getType() == newValue.getType());
  assert(!set && "backedge already set to a value!");
  value.replaceAllUsesWith(newValue);
  value = newValue; // In case the backedge is still referred to after setting.
  set = true;

  // If the backedge is referenced again, it should now point to the updated
  // value.
  value = newValue;
}

BackedgeBuilder::~BackedgeBuilder() { (void)clearOrEmitError(); }

LogicalResult BackedgeBuilder::clearOrEmitError() {
  unsigned numInUse = 0;
  for (Operation *op : edges) {
    if (!op->use_empty()) {
      auto diag = op->emitError("backedge of type `")
                  << op->getResult(0).getType() << "`still in use";
      for (auto user : op->getUsers())
        diag.attachNote(user->getLoc()) << "used by " << *user;
      ++numInUse;
      continue;
    }
    if (rewriter)
      rewriter->eraseOp(op);
    else
      op->erase();
  }
  edges.clear();
  if (numInUse > 0)
    mlir::emitRemark(loc, "abandoned ") << numInUse << " backedges";
  return success(numInUse == 0);
}

void BackedgeBuilder::abandon() { edges.clear(); }

BackedgeBuilder::BackedgeBuilder(OpBuilder &builder, Location loc)
    : builder(builder), rewriter(nullptr), loc(loc) {}
BackedgeBuilder::BackedgeBuilder(PatternRewriter &rewriter, Location loc)
    : builder(rewriter), rewriter(&rewriter), loc(loc) {}
Backedge BackedgeBuilder::get(Type t, mlir::LocationAttr optionalLoc) {
  if (!optionalLoc)
    optionalLoc = loc;
  Operation *op = builder.create<mlir::UnrealizedConversionCastOp>(
      optionalLoc, t, ValueRange{});
  edges.push_back(op);
  return Backedge(op);
}
