//===- HWInstanceGraph.cpp - Instance Graph ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWInstanceGraph.h"

using namespace circt;
using namespace hw;

InstanceGraph::InstanceGraph(Operation *operation)
    : InstanceGraphBase(operation) {
  for (auto &node : nodes)
    if (node.getModule().isPublic())
      entry.addInstance({}, &node);
}

InstanceGraphNode *InstanceGraph::addModule(HWModuleLike module) {
  auto *node = InstanceGraphBase::addModule(module);
  if (module.isPublic())
    entry.addInstance({}, node);
  return node;
}

void InstanceGraph::erase(InstanceGraphNode *node) {
  for (auto *instance : llvm::make_early_inc_range(entry)) {
    if (instance->getTarget() == node)
      instance->erase();
  }
  InstanceGraphBase::erase(node);
}
