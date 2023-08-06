//===- HWModuleGraph.h - HWModule graph -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the HWModuleGraph.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWMODULEGRAPH_H
#define CIRCT_DIALECT_HW_HWMODULEGRAPH_H

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/InstanceGraphBase.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/LLVM.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"

namespace circt {
namespace hw {
namespace detail {

// Using declaration to avoid polluting global namespace with CIRCT-specific
// graph traits for mlir::Operation.
using HWOperation = mlir::Operation;

} // namespace detail
} // namespace hw
} // namespace circt

template <>
struct llvm::GraphTraits<circt::hw::detail::HWOperation *> {
  using NodeType = circt::hw::detail::HWOperation;
  using NodeRef = NodeType *;

  using ChildIteratorType = mlir::Operation::user_iterator;
  static NodeRef getEntryNode(NodeRef op) { return op; }
  static ChildIteratorType child_begin(NodeRef op) { return op->user_begin(); }
  static ChildIteratorType child_end(NodeRef op) { return op->user_end(); }
};

template <>
struct llvm::GraphTraits<circt::hw::HWModuleOp>
    : public llvm::GraphTraits<circt::hw::detail::HWOperation *> {
  using GraphType = circt::hw::HWModuleOp;

  static NodeRef getEntryNode(GraphType mod) {
    return &mod.getBodyBlock()->front();
  }

  using nodes_iterator = pointer_iterator<mlir::Block::iterator>;
  static nodes_iterator nodes_begin(GraphType mod) {
    return nodes_iterator{mod.getBodyBlock()->begin()};
  }
  static nodes_iterator nodes_end(GraphType mod) {
    return nodes_iterator{mod.getBodyBlock()->end()};
  }
};

template <>
struct llvm::DOTGraphTraits<circt::hw::HWModuleOp>
    : public llvm::DefaultDOTGraphTraits {
  using DefaultDOTGraphTraits::DefaultDOTGraphTraits;

  static std::string getNodeLabel(circt::hw::detail::HWOperation *node,
                                  circt::hw::HWModuleOp) {
    return llvm::TypeSwitch<mlir::Operation *, std::string>(node)
        .Case<circt::comb::AddOp>([&](auto) { return "+"; })
        .Case<circt::comb::SubOp>([&](auto) { return "-"; })
        .Case<circt::comb::AndOp>([&](auto) { return "&"; })
        .Case<circt::comb::OrOp>([&](auto) { return "|"; })
        .Case<circt::comb::XorOp>([&](auto) { return "^"; })
        .Case<circt::comb::MulOp>([&](auto) { return "*"; })
        .Case<circt::comb::MuxOp>([&](auto) { return "mux"; })
        .Case<circt::comb::ShrSOp, circt::comb::ShrUOp>(
            [&](auto) { return ">>"; })
        .Case<circt::comb::ShlOp>([&](auto) { return "<<"; })
        .Case<circt::comb::ICmpOp>([&](auto op) {
          switch (op.getPredicate()) {
          case circt::comb::ICmpPredicate::eq:
          case circt::comb::ICmpPredicate::ceq:
          case circt::comb::ICmpPredicate::weq:
            return "==";
          case circt::comb::ICmpPredicate::wne:
          case circt::comb::ICmpPredicate::cne:
          case circt::comb::ICmpPredicate::ne:
            return "!=";
          case circt::comb::ICmpPredicate::uge:
          case circt::comb::ICmpPredicate::sge:
            return ">=";
          case circt::comb::ICmpPredicate::ugt:
          case circt::comb::ICmpPredicate::sgt:
            return ">";
          case circt::comb::ICmpPredicate::ule:
          case circt::comb::ICmpPredicate::sle:
            return "<=";
          case circt::comb::ICmpPredicate::ult:
          case circt::comb::ICmpPredicate::slt:
            return "<";
          }
          llvm_unreachable("unhandled ICmp predicate");
        })
        .Case<circt::seq::CompRegOp, circt::seq::FirRegOp>(
            [&](auto op) { return op.getName().str(); })
        .Case<circt::hw::ConstantOp>([&](auto op) {
          llvm::SmallString<64> valueString;
          op.getValue().toString(valueString, 10, false);
          return valueString.str().str();
        })
        .Default([&](auto op) { return op->getName().getStringRef().str(); });
  }

  std::string getNodeAttributes(circt::hw::detail::HWOperation *node,
                                circt::hw::HWModuleOp) {
    return llvm::TypeSwitch<mlir::Operation *, std::string>(node)
        .Case<circt::hw::ConstantOp>(
            [&](auto) { return "fillcolor=darkgoldenrod1,style=filled"; })
        .Case<circt::comb::MuxOp>([&](auto) {
          return "shape=invtrapezium,fillcolor=bisque,style=filled";
        })
        .Case<circt::hw::OutputOp>(
            [&](auto) { return "fillcolor=lightblue,style=filled"; })
        .Default([&](auto op) {
          return llvm::TypeSwitch<mlir::Dialect *, std::string>(
                     op->getDialect())
              .Case<circt::comb::CombDialect>([&](auto) {
                return "shape=oval,fillcolor=bisque,style=filled";
              })
              .template Case<circt::seq::SeqDialect>([&](auto) {
                return "shape=folder,fillcolor=gainsboro,style=filled";
              })
              .Default([&](auto) { return ""; });
        });
  }

  static void
  addCustomGraphFeatures(circt::hw::HWModuleOp mod,
                         llvm::GraphWriter<circt::hw::HWModuleOp> &g) {

    // Add module input args.
    auto &os = g.getOStream();
    os << "subgraph cluster_entry_args {\n";
    os << "label=\"Input arguments\";\n";
    auto iports = mod.getPortList();
    for (auto [info, arg] : llvm::zip(iports.getInputs(), mod.getArguments())) {
      g.emitSimpleNode(reinterpret_cast<void *>(&arg), "",
                       info.getName().str());
    }
    os << "}\n";
    for (auto [info, arg] : llvm::zip(iports.getInputs(), mod.getArguments())) {
      for (auto *user : arg.getUsers()) {
        g.emitEdge(reinterpret_cast<void *>(&arg), 0, user, -1, "");
      }
    }
  }

  template <typename Iterator>
  static std::string getEdgeAttributes(circt::hw::detail::HWOperation *node,
                                       Iterator it, circt::hw::HWModuleOp mod) {

    mlir::OpOperand &operand = *it.getCurrent();
    mlir::Value v = operand.get();
    std::string str;
    llvm::raw_string_ostream os(str);
    auto verboseEdges = mod->getAttrOfType<mlir::BoolAttr>("dot_verboseEdges");
    if (verboseEdges.getValue()) {
      os << "label=\"" << operand.getOperandNumber() << " (" << v.getType()
         << ")\"";
    }

    int64_t width = circt::hw::getBitWidth(v.getType());
    if (width > 1)
      os << " style=bold";

    return os.str();
  }
};

#endif // CIRCT_DIALECT_HW_HWMODULEGRAPH_H
