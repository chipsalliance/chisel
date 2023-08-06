//===- LogicExporter.cpp - class to extrapolate CIRCT IR logic --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file defines the logic-exporting class for the `circt-lec` tool.
///
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE
#ifndef TOOLS_CIRCT_LEC_LOGICEXPORTER_H
#define TOOLS_CIRCT_LEC_LOGICEXPORTER_H

#include "Solver.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombVisitors.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWVisitors.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace circt {

/// A class traversing MLIR IR to extrapolate the logic of a given circuit.
///
/// This class implements a MLIR exporter which searches the IR for the
/// specified `hw.module` describing a circuit. It will then traverse its
/// operations and collect the underlying logical constraints within an
/// abstract circuit representation.
class LogicExporter {
public:
  LogicExporter(llvm::StringRef moduleName, Solver::Circuit *circuit)
      : moduleName(moduleName), circuit(circuit) {}

  /// Initializes the exporting by visiting the builtin module.
  mlir::LogicalResult run(mlir::ModuleOp &module);
  mlir::LogicalResult run(hw::HWModuleOp &module);

private:
  // For Solver::Circuit::addInstance to access Visitor::visitHW.
  friend Solver::Circuit;

  /// The specified module name to look for when traversing the input file.
  std::string moduleName;
  /// The circuit representation to hold the logical constraints extracted
  /// from the IR.
  Solver::Circuit *circuit;
};

} // namespace circt

#endif // TOOLS_CIRCT_LEC_LOGICEXPORTER_H
