//===- FIRRTLInstanceGraph.cpp - Instance Graph -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "mlir/IR/BuiltinOps.h"

using namespace circt;
using namespace firrtl;

static CircuitOp findCircuitOp(Operation *operation) {
  if (auto mod = dyn_cast<mlir::ModuleOp>(operation))
    for (auto &op : *mod.getBody())
      if (auto circuit = dyn_cast<CircuitOp>(&op))
        return circuit;
  return cast<CircuitOp>(operation);
}

InstanceGraph::InstanceGraph(Operation *operation)
    : InstanceGraphBase(findCircuitOp(operation)) {
  topLevelNode = lookup(cast<CircuitOp>(getParent()).getNameAttr());
}

bool circt::firrtl::allUnder(ArrayRef<InstanceRecord *> nodes,
                             InstanceGraphNode *top) {
  DenseSet<InstanceGraphNode *> seen;
  SmallVector<InstanceGraphNode *> worklist;
  worklist.reserve(nodes.size());
  seen.reserve(nodes.size());
  seen.insert(top);
  for (auto *n : nodes) {
    auto *mod = n->getParent();
    if (seen.insert(mod).second)
      worklist.push_back(mod);
  }

  while (!worklist.empty()) {
    auto *node = worklist.back();
    worklist.pop_back();

    assert(node != top);

    // If reach top-level node we're not covered by 'top', return.
    if (node->noUses())
      return false;

    // Otherwise, walk upwards.
    for (auto *use : node->uses()) {
      auto *mod = use->getParent();
      if (seen.insert(mod).second)
        worklist.push_back(mod);
    }
  }
  return true;
}
