//===- InstanceGraph.h - Instance graph -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the HW InstanceGraph, which is similar to a CallGraph.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWINSTANCEGRAPH_H
#define CIRCT_DIALECT_HW_HWINSTANCEGRAPH_H

#include "circt/Dialect/HW/InstanceGraphBase.h"

namespace circt {
namespace hw {

/// HW-specific instance graph with a virtual entry node linking to
/// all publicly visible modules.
class InstanceGraph : public InstanceGraphBase {
public:
  InstanceGraph(Operation *operation);

  /// Return the entry node linking to all public modules.
  InstanceGraphNode *getTopLevelNode() override { return &entry; }

  /// Adds a module, updating links to entry.
  InstanceGraphNode *addModule(HWModuleLike module) override;

  /// Erases a module, updating links to entry.
  void erase(InstanceGraphNode *node) override;

private:
  InstanceGraphNode entry;
};

} // namespace hw
} // namespace circt

// Specialisation for the HW instance graph.
template <>
struct llvm::GraphTraits<circt::hw::InstanceGraph *>
    : public llvm::GraphTraits<circt::hw::InstanceGraphBase *> {};

template <>
struct llvm::DOTGraphTraits<circt::hw::InstanceGraph *>
    : public llvm::DOTGraphTraits<circt::hw::InstanceGraphBase *> {
  using llvm::DOTGraphTraits<circt::hw::InstanceGraphBase *>::DOTGraphTraits;
};

#endif // CIRCT_DIALECT_HW_HWINSTANCEGRAPH_H
