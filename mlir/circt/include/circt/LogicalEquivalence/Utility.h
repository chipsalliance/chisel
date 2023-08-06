//===-- Utility.h - collection of utility functions and macros --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This header provides a variety of utility functions and macros for use
/// throughout the tool.
///
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE
#ifndef TOOLS_CIRCT_LEC_UTILITY_H
#define TOOLS_CIRCT_LEC_UTILITY_H

#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <z3++.h>

namespace lec {
// Defining persistent output streams such that text will be printed in
// accordance with the globally set indentation level.
inline mlir::raw_indented_ostream &dbgs() {
  static auto stream = mlir::raw_indented_ostream(llvm::dbgs());
  return stream;
}

inline mlir::raw_indented_ostream &errs() {
  static auto stream = mlir::raw_indented_ostream(llvm::errs());
  return stream;
}

inline mlir::raw_indented_ostream &outs() {
  static auto stream = mlir::raw_indented_ostream(llvm::outs());
  return stream;
}

/// RAII struct to indent the output streams.
struct Scope {
  mlir::raw_indented_ostream::DelimitedScope indentDbgs = lec::dbgs().scope();
  mlir::raw_indented_ostream::DelimitedScope indentErrs = lec::errs().scope();
  mlir::raw_indented_ostream::DelimitedScope indentOuts = lec::outs().scope();
};

/// Helper function to provide a common debug formatting for z3 expressions.
inline void printExpr(const z3::expr &expr) {
  lec::dbgs() << "symbol: " << expr.to_string() << "\n";
  lec::dbgs() << "sort: " << expr.get_sort().to_string() << "\n";
  lec::dbgs() << "expression id: " << expr.id() << "\n";
  lec::dbgs() << "expression hash: " << expr.hash() << "\n";
}

/// Helper function to provide a common debug formatting for MLIR values.
inline void printValue(const mlir::Value &value) {
  lec::dbgs() << "value: " << value << "\n";
  lec::dbgs() << "type: " << value.getType() << "\n";
  lec::dbgs() << "value hash: " << mlir::hash_value(value) << "\n";
}

/// Helper function to provide a common debug formatting for MLIR APInt'egers.
inline void printAPInt(const mlir::APInt &value) {
  lec::dbgs() << "APInt: " << value.getZExtValue() << "\n";
}
} // namespace lec

#endif // TOOLS_CIRCT_LEC_UTILITY_H
