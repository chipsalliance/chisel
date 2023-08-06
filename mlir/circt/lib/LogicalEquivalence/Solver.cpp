//===-- Solver.h - SMT solver interface -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file defines a SMT solver interface for the 'circt-lec' tool.
///
//===----------------------------------------------------------------------===//

#include "circt/LogicalEquivalence/Solver.h"
#include "circt/LogicalEquivalence/Circuit.h"
#include "circt/LogicalEquivalence/LogicExporter.h"
#include "circt/LogicalEquivalence/Utility.h"
#include "mlir/IR/Builders.h"
#include <string>
#include <z3++.h>

#define DEBUG_TYPE "lec-solver"

using namespace circt;
using namespace mlir;

Solver::Solver(MLIRContext *mlirCtx, bool statisticsOpt)
    : circuits{}, mlirCtx(mlirCtx), context(), solver(context),
      statisticsOpt(statisticsOpt) {}

/// Solve the equivalence problem between the two circuits, then present the
/// results to the user.
LogicalResult Solver::solve() {
  // Constrain the circuits for equivalence checking to be made:
  // require them to produce different outputs starting from the same inputs.
  if (constrainCircuits().failed())
    return failure();

  // Instruct the logical engine to solve the constraints:
  // if they can't be satisfied it must mean the two circuits are functionally
  // equivalent. Otherwise, print a model to act as a counterexample.
  LogicalResult outcome = success();
  switch (solver.check()) {
  case z3::unsat:
    lec::outs() << "c1 == c2\n";
    break;
  case z3::sat:
    lec::outs() << "c1 != c2\n";
    printModel();
    outcome = failure();
    break;
  case z3::unknown:
    outcome = failure();
    lec::errs() << "circt-lec error: solver ran out of time\n";
  }

  // Print further relevant information as requested.
  LLVM_DEBUG(printAssertions());
  if (statisticsOpt)
    printStatistics();

  return outcome;
}

/// Create a new circuit to be compared and return it.
Solver::Circuit *Solver::addCircuit(llvm::StringRef name) {
  // NOLINTNEXTLINE
  assert(!(circuits[0] && circuits[1]) && "already added two circuits");
  // Hack: entities within the logical engine are namespaced by the circuit
  // they belong to, which may cause shadowing when parsing two files with a
  // similar module naming scheme.
  // To avoid that, they're differentiated by a prefix.
  unsigned n = circuits[0] ? 1 : 0;
  std::string prefix = n == 0 ? "c1@" : "c2@";
  circuits[n] = new Solver::Circuit(prefix + name, *this);
  return circuits[n];
}

/// Prints a model satisfying the solved constraints.
void Solver::printModel() {
  lec::outs() << "Model:\n";
  lec::Scope indent;
  z3::model model = solver.get_model();
  for (unsigned int i = 0; i < model.size(); i++) {
    // Recover the corresponding Value for the z3::expression
    // then emit a remark for its location.
    z3::func_decl f = model.get_const_decl(i);
    Builder builder(mlirCtx);
    std::string symbolStr = f.name().str();
    StringAttr symbol = builder.getStringAttr(symbolStr);
    Value value = symbolTable.find(symbol)->second;
    z3::expr e = model.get_const_interp(f);
    emitRemark(value.getLoc(), "");
    // Explicitly unfolded asm printing for `Value`.
    if (auto arg = value.dyn_cast<BlockArgument>()) {
      // Value is an argument rather than a SSA'ed value of an operation.
      Operation *parentOp = value.getParentRegion()->getParentOp();
      if (auto op = llvm::dyn_cast<hw::HWModuleOp>(parentOp)) {
        // Argument of a `hw.module`.
        lec::outs() << "argument name: " << op.getArgNames()[arg.getArgNumber()]
                    << "\n";
      } else {
        // Argument of a different operation.
        lec::outs() << arg << "\n";
      }
    }
    // Accompanying model information.
    lec::outs() << "internal symbol: " << symbol << "\n";
    lec::outs() << "model interpretation: " << e.to_string() << "\n\n";
  }
}

/// Prints the constraints which were added to the solver.
/// Compared to solver.assertions().to_string() this method exposes each
/// assertion as a z3::expression for eventual in-depth debugging.
void Solver::printAssertions() {
  lec::dbgs() << "Assertions:\n";
  lec::Scope indent;
  for (z3::expr assertion : solver.assertions()) {
    lec::dbgs() << assertion.to_string() << "\n";
  }
}

/// Prints the internal statistics of the SMT solver for benchmarking purposes
/// and operational insight.
void Solver::printStatistics() {
  lec::outs() << "SMT solver statistics:\n";
  lec::Scope indent;
  z3::stats stats = solver.statistics();
  for (unsigned i = 0; i < stats.size(); i++) {
    lec::outs() << stats.key(i) << " : " << stats.uint_value(i) << "\n";
  }
}

/// Formulates additional constraints which are satisfiable if only if the
/// two circuits which are being compared are NOT equivalent, in which case
/// there would be a model acting as a counterexample.
/// The procedure fails when detecting a mismatch of arity or type between
/// the inputs and outputs of the circuits.
LogicalResult Solver::constrainCircuits() {
  // TODO: Perform these failure checks before nalyzing the whole IR of the
  // modules during the pass.
  auto c1Inputs = circuits[0]->getInputs();
  auto c2Inputs = circuits[1]->getInputs();
  unsigned nc1Inputs = std::distance(c1Inputs.begin(), c1Inputs.end());
  unsigned nc2Inputs = std::distance(c2Inputs.begin(), c2Inputs.end());

  // Can't compare two circuits with different number of inputs.
  if (nc1Inputs != nc2Inputs) {
    lec::errs() << "circt-lec error: different input arity\n";
    return failure();
  }

  const auto *c1inIt = c1Inputs.begin();
  const auto *c2inIt = c2Inputs.begin();
  for (unsigned i = 0; i < nc1Inputs; i++) {
    // Can't compare two circuits when their ith inputs differ in type.
    if (c1inIt->get_sort().bv_size() != c2inIt->get_sort().bv_size()) {
      lec::errs() << "circt-lec error: input #" << i + 1 << " type mismatch\n";
      return failure();
    }
    // Their ith inputs have to be equivalent.
    solver.add(*c1inIt++ == *c2inIt++);
  }

  auto c1Outputs = circuits[0]->getOutputs();
  auto c2Outputs = circuits[1]->getOutputs();
  unsigned nc1Outputs = std::distance(c1Outputs.begin(), c1Outputs.end());
  unsigned nc2Outputs = std::distance(c2Outputs.begin(), c2Outputs.end());

  // Can't compare two circuits with different number of outputs.
  if (nc1Outputs != nc2Outputs) {
    lec::errs() << "circt-lec error: different output arity\n";
    return failure();
  }

  z3::expr_vector outputTerms(context);

  const auto *c1outIt = c1Outputs.begin();
  const auto *c2outIt = c2Outputs.begin();
  for (unsigned i = 0; i < nc1Outputs; i++) {
    // Can't compare two circuits when their ith outputs differ in type.
    if (c1outIt->get_sort().bv_size() != c2outIt->get_sort().bv_size()) {
      lec::errs() << "circt-lec error: output #" << i + 1 << " type mismatch\n";
      return failure();
    }
    // Their ith outputs have to be equivalent.
    outputTerms.push_back(*c1outIt++ != *c2outIt++);
  }

  // The circuits are not equivalent iff any of the outputs is not equal
  solver.add(z3::mk_or(outputTerms));

  return success();
}
