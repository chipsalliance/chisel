//===- Utilities.cpp - Library of scheduling utilities --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains useful helpers for scheduler implementations.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Utilities.h"

#include "mlir/IR/Operation.h"
#include "mlir/Support/IndentedOstream.h"

using namespace circt;
using namespace circt::scheduling;

LogicalResult scheduling::handleOperationsInTopologicalOrder(Problem &prob,
                                                             HandleOpFn fun) {
  auto &allOps = prob.getOperations();
  SmallVector<Operation *> unhandledOps;
  unhandledOps.insert(unhandledOps.begin(), allOps.begin(), allOps.end());

  while (!unhandledOps.empty()) {
    // Remember how many unhandled operations we have at the beginning of this
    // attempt. This is a fail-safe for cyclic dependence graphs: If we do not
    // successfully handle at least one operation per attempt, we have
    // encountered a cycle.
    unsigned numUnhandledBefore = unhandledOps.size();

    // Set up the worklist for this attempt, and initialize it in reverse order
    // so that we can pop off its back later.
    SmallVector<Operation *> worklist;
    worklist.insert(worklist.begin(), unhandledOps.rbegin(),
                    unhandledOps.rend());
    unhandledOps.clear();

    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      auto res = fun(op);
      if (failed(res))
        unhandledOps.push_back(op);
    }

    if (numUnhandledBefore == unhandledOps.size())
      return prob.getContainingOp()->emitError() << "dependence cycle detected";
  }

  return success();
}

void scheduling::dumpAsDOT(Problem &prob, StringRef fileName) {
  std::error_code ec;
  llvm::raw_fd_ostream out(fileName, ec);
  scheduling::dumpAsDOT(prob, out);
  out.close();
}

void scheduling::dumpAsDOT(Problem &prob, raw_ostream &stream) {
  mlir::raw_indented_ostream os(stream);

  os << "digraph G {\n";
  os.indent();
  os << "rankdir = TB     // top to bottom\n";
  os << "splines = spline // draw edges and route around nodes\n";
  os << "nodesep = 0.2    // horizontal compression\n";
  os << "ranksep = 0.5    // vertical compression\n";
  os << "node [shape=box] // default node style\n";
  os << "compound = true  // allow edges between subgraphs\n";

  auto startHTMLLabel = [&os]() {
    os << "<<TABLE BORDER=\"0\">\n";
    os.indent();
  };
  auto emitTableHeader = [&os](std::string str) {
    os << "<TR><TD COLSPAN=\"2\"><B>" << str << "</B></TD></TR>\n";
  };
  auto emitTableRow = [&os](std::pair<std::string, std::string> &kv) {
    os << "<TR><TD ALIGN=\"LEFT\">" << std::get<0>(kv)
       << ":</TD><TD ALIGN=\"RIGHT\">" << std::get<1>(kv) << "</TD></TR>\n";
  };
  auto endHTMLLabel = [&os]() {
    os.unindent();
    os << "</TABLE>>";
  };

  DenseMap<Operation *, std::string> nodes;

  os << "\n// Operations\n";
  os << "subgraph dependence_graph {\n";
  os.indent();

  for (auto *op : prob.getOperations()) {
    auto id = std::to_string(nodes.size());
    auto node = "op" + id;
    nodes[op] = node;

    os << node << " [label = ";
    startHTMLLabel();
    emitTableHeader(("#" + id + " " + op->getName().getStringRef()).str());
    for (auto &kv : prob.getProperties(op))
      emitTableRow(kv);
    endHTMLLabel();
    os << "]\n";
  }

  os << "\n// Dependences\n";
  for (auto *op : prob.getOperations())
    for (auto &dep : prob.getDependences(op)) {
      os << nodes[dep.getSource()] << " -> " << nodes[dep.getDestination()]
         << " [";
      if (dep.isAuxiliary())
        os << "style = dashed ";
      if (auto props = prob.getProperties(dep); !props.empty()) {
        os << "label = ";
        startHTMLLabel();
        for (auto &kv : props)
          emitTableRow(kv);
        endHTMLLabel();
      }
      os << "]\n";
    }
  os.unindent();
  os << "}\n";

  os << "\n// Operator types\n";
  os << "subgraph cluster_operator_types {\n";
  os.indent();
  os << "label = \"Operator types\"\n";
  os << "style = filled fillcolor = lightgray\n";
  os << "node [style = \"rounded,filled\" fillcolor = white]\n";
  unsigned oprId = 0;
  for (auto opr : prob.getOperatorTypes()) {
    os << "opr" << oprId << " [label = ";
    startHTMLLabel();
    emitTableHeader(opr.str());
    for (auto &kv : prob.getProperties(opr))
      emitTableRow(kv);
    endHTMLLabel();
    os << "]\n";
    ++oprId;
  }
  os.unindent();
  os << "}\n";

  if (auto instanceProps = prob.getProperties(); !instanceProps.empty()) {
    os << "\n// Instance\n";
    os << "instance [shape = note, label = ";
    startHTMLLabel();
    emitTableHeader("Instance");
    for (auto &kv : instanceProps)
      emitTableRow(kv);
    endHTMLLabel();
    os << "]\n";
  }

  os.unindent();
  os << "}\n";
}
