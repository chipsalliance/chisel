//===- ReductionUtils.cpp - Reduction pattern utilities -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Reduce/ReductionUtils.h"
#include "circt/Reduce/Reduction.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallSet.h"

using namespace circt;

void reduce::pruneUnusedOps(Operation *initialOp, Reduction &reduction) {
  SmallVector<Operation *> worklist;
  SmallSet<Operation *, 4> handled;
  worklist.push_back(initialOp);
  while (!worklist.empty()) {
    auto *op = worklist.pop_back_val();
    if (!op->use_empty())
      continue;
    for (auto arg : op->getOperands())
      if (auto *argOp = arg.getDefiningOp())
        if (handled.insert(argOp).second)
          worklist.push_back(argOp);
    reduction.notifyOpErased(op);
    op->erase();
  }
}
