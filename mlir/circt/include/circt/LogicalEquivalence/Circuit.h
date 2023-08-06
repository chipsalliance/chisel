//===-- Circuit.h - intermediate representation for circuits ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file defines an intermediate representation for circuits acting as
/// an abstraction for constraints defined over an SMT's solver context.
///
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE
#ifndef TOOLS_CIRCT_LEC_CIRCUIT_H
#define TOOLS_CIRCT_LEC_CIRCUIT_H

#include "Solver.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <string>
#include <z3++.h>

namespace circt {

/// The representation of a circuit within a logical engine.
///
/// This class defines a circuit as an abstraction of its underlying
/// logical constraints. Its various methods act in a builder pattern fashion,
/// declaring new constraints over a Z3 context.
class Solver::Circuit {
public:
  Circuit(llvm::Twine name, Solver &solver) : name(name.str()), solver(solver) {
    assignments = 0;
  };
  /// Add an input to the circuit; internally a new value gets allocated.
  void addInput(mlir::Value);
  /// Add an output to the circuit.
  void addOutput(mlir::Value);
  /// Recover the inputs.
  llvm::ArrayRef<z3::expr> getInputs();
  /// Recover the outputs.
  llvm::ArrayRef<z3::expr> getOutputs();

  // `hw` dialect operations.
  void addConstant(mlir::Value result, const mlir::APInt &value);
  void addInstance(llvm::StringRef instanceName, circt::hw::HWModuleOp op,
                   mlir::OperandRange arguments, mlir::ResultRange results);

  // `comb` dialect operations.
  void performAdd(mlir::Value result, mlir::OperandRange operands);
  void performAnd(mlir::Value result, mlir::OperandRange operands);
  void performConcat(mlir::Value result, mlir::OperandRange operands);
  void performDivS(mlir::Value result, mlir::Value lhs, mlir::Value rhs);
  void performDivU(mlir::Value result, mlir::Value lhs, mlir::Value rhs);
  void performExtract(mlir::Value result, mlir::Value input, uint32_t lowBit);
  mlir::LogicalResult performICmp(mlir::Value result,
                                  circt::comb::ICmpPredicate predicate,
                                  mlir::Value lhs, mlir::Value rhs);
  void performModS(mlir::Value result, mlir::Value lhs, mlir::Value rhs);
  void performModU(mlir::Value result, mlir::Value lhs, mlir::Value rhs);
  void performMul(mlir::Value result, mlir::OperandRange operands);
  void performMux(mlir::Value result, mlir::Value cond, mlir::Value trueValue,
                  mlir::Value falseValue);
  void performOr(mlir::Value result, mlir::OperandRange operands);
  void performParity(mlir::Value result, mlir::Value input);
  void performReplicate(mlir::Value result, mlir::Value input);
  void performShl(mlir::Value result, mlir::Value lhs, mlir::Value rhs);
  void performShrS(mlir::Value result, mlir::Value lhs, mlir::Value rhs);
  void performShrU(mlir::Value result, mlir::Value lhs, mlir::Value rhs);
  void performSub(mlir::Value result, mlir::OperandRange operands);
  void performXor(mlir::Value result, mlir::OperandRange operands);

private:
  /// Helper function for performing a variadic operation: it executes a lambda
  /// over a range of operands.
  void variadicOperation(
      mlir::Value result, mlir::OperandRange operands,
      llvm::function_ref<z3::expr(const z3::expr &, const z3::expr &)>
          operation);
  /// Allocates an IR value in the logical backend and returns its representing
  /// expression.
  z3::expr allocateValue(mlir::Value value);
  /// Allocates a constant value in the logical backend and returns its
  /// representing expression.
  void allocateConstant(mlir::Value opResult, const mlir::APInt &opValue);
  /// Fetches the corresponding logical expression for a given IR value.
  z3::expr fetchExpr(mlir::Value &value);
  /// Constrains the result of a MLIR operation to be equal a given logical
  /// express, simulating an assignment.
  void constrainResult(mlir::Value &result, z3::expr &expr);

  /// Convert from bitvector to bool sort.
  z3::expr bvToBool(const z3::expr &condition);
  /// Convert from a boolean sort to the corresponding 1-width bitvector.
  z3::expr boolToBv(const z3::expr &condition);

  /// The name of the circuit; it corresponds to its scope within the parsed IR.
  std::string name;
  /// A counter for how many assignments have occurred; it's used to uniquely
  /// name new values as they have to be represented within the logical engine's
  /// context.
  unsigned assignments;
  /// The solver environment the circuit belongs to.
  Solver &solver;
  /// The list for the circuit's inputs.
  llvm::SmallVector<z3::expr> inputs;
  /// The list for the circuit's outputs.
  llvm::SmallVector<z3::expr> outputs;
  /// A map from IR values to their corresponding logical representation.
  llvm::DenseMap<mlir::Value, z3::expr> exprTable;
};

} // namespace circt

#endif // TOOLS_CIRCT_LEC_CIRCUIT_H
