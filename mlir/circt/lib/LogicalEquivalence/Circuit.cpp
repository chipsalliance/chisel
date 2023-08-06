//===-- Circuit.cpp - intermediate representation for circuits --*- C++ -*-===//
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

#include "circt/LogicalEquivalence/Circuit.h"
#include "circt/LogicalEquivalence/LogicExporter.h"
#include "circt/LogicalEquivalence/Solver.h"
#include "circt/LogicalEquivalence/Utility.h"
#include "mlir/IR/Builders.h"

#define DEBUG_TYPE "lec-circuit"

using namespace mlir;
using namespace circt;

/// Add an input to the circuit; internally a new value gets allocated.
void Solver::Circuit::addInput(Value value) {
  LLVM_DEBUG(lec::dbgs() << name << " addInput\n");
  lec::Scope indent;
  z3::expr input = allocateValue(value);
  inputs.insert(inputs.end(), input);
}

/// Add an output to the circuit.
void Solver::Circuit::addOutput(Value value) {
  LLVM_DEBUG(lec::dbgs() << name << " addOutput\n");
  // Referenced value already assigned, fetching from expression table.
  z3::expr output = fetchExpr(value);
  outputs.insert(outputs.end(), output);
}

/// Recover the inputs.
llvm::ArrayRef<z3::expr> Solver::Circuit::getInputs() { return inputs; }

/// Recover the outputs.
llvm::ArrayRef<z3::expr> Solver::Circuit::getOutputs() { return outputs; }

//===----------------------------------------------------------------------===//
// `hw` dialect operations
//===----------------------------------------------------------------------===//

void Solver::Circuit::addConstant(Value opResult, const APInt &opValue) {
  LLVM_DEBUG(lec::dbgs() << name << " addConstant\n");
  lec::Scope indent;
  allocateConstant(opResult, opValue);
}

void Solver::Circuit::addInstance(llvm::StringRef instanceName,
                                  circt::hw::HWModuleOp op,
                                  OperandRange arguments, ResultRange results) {
  LLVM_DEBUG(lec::dbgs() << name << " addInstance\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "instance name: " << instanceName << "\n");
  LLVM_DEBUG(lec::dbgs() << "module name: " << op->getName() << "\n");
  // There is no preventing multiple instances holding the same name.
  // As an hack, a suffix is used to differentiate them.
  std::string suffix = "_" + std::to_string(assignments);
  Circuit instance(name + "@" + instanceName + suffix, solver);
  // Export logic to the instance's circuit by visiting the IR of the
  // instanced module.
  auto res = LogicExporter(op.getModuleName(), &instance).run(op);
  assert(res.succeeded() && "Instance visit failed");

  // Constrain the inputs and outputs of the instanced circuit to, respectively,
  // the arguments and results of the instance operation.
  {
    LLVM_DEBUG(lec::dbgs() << "instance inputs:\n");
    lec::Scope indent;
    auto *input = instance.inputs.begin();
    for (Value argument : arguments) {
      LLVM_DEBUG(lec::dbgs() << "input\n");
      z3::expr argExpr = fetchExpr(argument);
      solver.solver.add(argExpr == *input++);
    }
  }
  {
    LLVM_DEBUG(lec::dbgs() << "instance results:\n");
    lec::Scope indent;
    auto *output = instance.outputs.begin();
    for (circt::OpResult result : results) {
      z3::expr resultExpr = allocateValue(result);
      solver.solver.add(resultExpr == *output++);
    }
  }
}

//===----------------------------------------------------------------------===//
// `comb` dialect operations
//===----------------------------------------------------------------------===//

void Solver::Circuit::performAdd(Value result, OperandRange operands) {
  LLVM_DEBUG(lec::dbgs() << name << " perform Add\n");
  lec::Scope indent;
  variadicOperation(result, operands,
                    [](auto op1, auto op2) { return op1 + op2; });
}

void Solver::Circuit::performAnd(Value result, OperandRange operands) {
  LLVM_DEBUG(lec::dbgs() << name << " perform And\n");
  lec::Scope indent;
  variadicOperation(result, operands,
                    [](auto op1, auto op2) { return z3::operator&(op1, op2); });
}

void Solver::Circuit::performConcat(Value result, OperandRange operands) {
  LLVM_DEBUG(lec::dbgs() << name << " perform Concat\n");
  lec::Scope indent;
  variadicOperation(result, operands,
                    [](auto op1, auto op2) { return z3::concat(op1, op2); });
}

void Solver::Circuit::performDivS(Value result, Value lhs, Value rhs) {
  LLVM_DEBUG(lec::dbgs() << name << " perform DivS\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "lhs:\n");
  z3::expr lhsExpr = fetchExpr(lhs);
  LLVM_DEBUG(lec::dbgs() << "rhs:\n");
  z3::expr rhsExpr = fetchExpr(rhs);
  z3::expr op = z3::operator/(lhsExpr, rhsExpr);
  constrainResult(result, op);
}

void Solver::Circuit::performDivU(Value result, Value lhs, Value rhs) {
  LLVM_DEBUG(lec::dbgs() << name << " perform DivU\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "lhs:\n");
  z3::expr lhsExpr = fetchExpr(lhs);
  LLVM_DEBUG(lec::dbgs() << "rhs:\n");
  z3::expr rhsExpr = fetchExpr(rhs);
  z3::expr op = z3::udiv(lhsExpr, rhsExpr);
  constrainResult(result, op);
}

void Solver::Circuit::performExtract(Value result, Value input,
                                     uint32_t lowBit) {
  LLVM_DEBUG(lec::dbgs() << name << " performExtract\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "input:\n");
  z3::expr inputExpr = fetchExpr(input);
  unsigned width = result.getType().getIntOrFloatBitWidth();
  LLVM_DEBUG(lec::dbgs() << "width: " << width << "\n");
  z3::expr extract = inputExpr.extract(lowBit + width - 1, lowBit);
  constrainResult(result, extract);
}

LogicalResult Solver::Circuit::performICmp(Value result,
                                           circt::comb::ICmpPredicate predicate,
                                           Value lhs, Value rhs) {
  LLVM_DEBUG(lec::dbgs() << name << " performICmp\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "lhs:\n");
  z3::expr lhsExpr = fetchExpr(lhs);
  LLVM_DEBUG(lec::dbgs() << "rhs:\n");
  z3::expr rhsExpr = fetchExpr(rhs);
  z3::expr icmp(solver.context);

  switch (predicate) {
  case circt::comb::ICmpPredicate::eq:
    icmp = boolToBv(lhsExpr == rhsExpr);
    break;
  case circt::comb::ICmpPredicate::ne:
    icmp = boolToBv(lhsExpr != rhsExpr);
    break;
  case circt::comb::ICmpPredicate::slt:
    icmp = boolToBv(z3::slt(lhsExpr, rhsExpr));
    break;
  case circt::comb::ICmpPredicate::sle:
    icmp = boolToBv(z3::sle(lhsExpr, rhsExpr));
    break;
  case circt::comb::ICmpPredicate::sgt:
    icmp = boolToBv(z3::sgt(lhsExpr, rhsExpr));
    break;
  case circt::comb::ICmpPredicate::sge:
    icmp = boolToBv(z3::sge(lhsExpr, rhsExpr));
    break;
  case circt::comb::ICmpPredicate::ult:
    icmp = boolToBv(z3::ult(lhsExpr, rhsExpr));
    break;
  case circt::comb::ICmpPredicate::ule:
    icmp = boolToBv(z3::ule(lhsExpr, rhsExpr));
    break;
  case circt::comb::ICmpPredicate::ugt:
    icmp = boolToBv(z3::ugt(lhsExpr, rhsExpr));
    break;
  case circt::comb::ICmpPredicate::uge:
    icmp = boolToBv(z3::uge(lhsExpr, rhsExpr));
    break;
  // Multi-valued logic comparisons are not supported.
  case circt::comb::ICmpPredicate::ceq:
  case circt::comb::ICmpPredicate::weq:
  case circt::comb::ICmpPredicate::cne:
  case circt::comb::ICmpPredicate::wne:
    result.getDefiningOp()->emitError(
        "n-state logic predicates are not supported");
    return failure();
  };

  constrainResult(result, icmp);
  return success();
}

void Solver::Circuit::performModS(Value result, Value lhs, Value rhs) {
  LLVM_DEBUG(lec::dbgs() << name << " perform ModS\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "lhs:\n");
  z3::expr lhsExpr = fetchExpr(lhs);
  LLVM_DEBUG(lec::dbgs() << "rhs:\n");
  z3::expr rhsExpr = fetchExpr(rhs);
  z3::expr op = z3::smod(lhsExpr, rhsExpr);
  constrainResult(result, op);
}

void Solver::Circuit::performModU(Value result, Value lhs, Value rhs) {
  LLVM_DEBUG(lec::dbgs() << name << " perform ModU\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "lhs:\n");
  z3::expr lhsExpr = fetchExpr(lhs);
  LLVM_DEBUG(lec::dbgs() << "rhs:\n");
  z3::expr rhsExpr = fetchExpr(rhs);
  z3::expr op = z3::urem(lhsExpr, rhsExpr);
  constrainResult(result, op);
}

void Solver::Circuit::performMul(Value result, OperandRange operands) {
  LLVM_DEBUG(lec::dbgs() << name << " perform Mul\n");
  lec::Scope indent;
  variadicOperation(result, operands,
                    [](auto op1, auto op2) { return op1 * op2; });
}

void Solver::Circuit::performMux(Value result, Value cond, Value trueValue,
                                 Value falseValue) {
  LLVM_DEBUG(lec::dbgs() << name << " performMux\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "cond:\n");
  z3::expr condExpr = fetchExpr(cond);
  LLVM_DEBUG(lec::dbgs() << "trueValue:\n");
  z3::expr tvalue = fetchExpr(trueValue);
  LLVM_DEBUG(lec::dbgs() << "falseValue:\n");
  z3::expr fvalue = fetchExpr(falseValue);
  // Conversion due to z3::ite requiring a bool rather than a bitvector.
  z3::expr mux = z3::ite(bvToBool(condExpr), tvalue, fvalue);
  constrainResult(result, mux);
}

void Solver::Circuit::performOr(Value result, OperandRange operands) {
  LLVM_DEBUG(lec::dbgs() << name << " perform Or\n");
  lec::Scope indent;
  variadicOperation(result, operands,
                    [](auto op1, auto op2) { return op1 | op2; });
}

void Solver::Circuit::performParity(Value result, Value input) {
  LLVM_DEBUG(lec::dbgs() << name << " performParity\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "input:\n");
  z3::expr inputExpr = fetchExpr(input);

  unsigned width = inputExpr.get_sort().bv_size();

  // input has 1 or more bits
  z3::expr parity = inputExpr.extract(0, 0);
  // calculate parity with every other bit
  for (unsigned int i = 1; i < width; i++) {
    parity = parity ^ inputExpr.extract(i, i);
  }

  constrainResult(result, parity);
}

void Solver::Circuit::performReplicate(Value result, Value input) {
  LLVM_DEBUG(lec::dbgs() << name << " performReplicate\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "input:\n");
  z3::expr inputExpr = fetchExpr(input);

  unsigned int final = result.getType().getIntOrFloatBitWidth();
  unsigned int initial = input.getType().getIntOrFloatBitWidth();
  unsigned int times = final / initial;
  LLVM_DEBUG(lec::dbgs() << "replies: " << times << "\n");

  z3::expr replicate = inputExpr;
  for (unsigned int i = 1; i < times; i++) {
    replicate = z3::concat(replicate, inputExpr);
  }

  constrainResult(result, replicate);
}

void Solver::Circuit::performShl(Value result, Value lhs, Value rhs) {
  LLVM_DEBUG(lec::dbgs() << name << " perform Shl\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "lhs:\n");
  z3::expr lhsExpr = fetchExpr(lhs);
  LLVM_DEBUG(lec::dbgs() << "rhs:\n");
  z3::expr rhsExpr = fetchExpr(rhs);
  z3::expr op = z3::shl(lhsExpr, rhsExpr);
  constrainResult(result, op);
}

// Arithmetic shift right.
void Solver::Circuit::performShrS(Value result, Value lhs, Value rhs) {
  LLVM_DEBUG(lec::dbgs() << name << " perform ShrS\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "lhs:\n");
  z3::expr lhsExpr = fetchExpr(lhs);
  LLVM_DEBUG(lec::dbgs() << "rhs:\n");
  z3::expr rhsExpr = fetchExpr(rhs);
  z3::expr op = z3::ashr(lhsExpr, rhsExpr);
  constrainResult(result, op);
}

// Logical shift right.
void Solver::Circuit::performShrU(Value result, Value lhs, Value rhs) {
  LLVM_DEBUG(lec::dbgs() << name << " perform ShrU\n");
  lec::Scope indent;
  LLVM_DEBUG(lec::dbgs() << "lhs:\n");
  z3::expr lhsExpr = fetchExpr(lhs);
  LLVM_DEBUG(lec::dbgs() << "rhs:\n");
  z3::expr rhsExpr = fetchExpr(rhs);
  z3::expr op = z3::lshr(lhsExpr, rhsExpr);
  constrainResult(result, op);
}

void Solver::Circuit::performSub(Value result, OperandRange operands) {
  LLVM_DEBUG(lec::dbgs() << name << " perform Sub\n");
  lec::Scope indent;
  variadicOperation(result, operands,
                    [](auto op1, auto op2) { return op1 - op2; });
}

void Solver::Circuit::performXor(Value result, OperandRange operands) {
  LLVM_DEBUG(lec::dbgs() << name << " perform Xor\n");
  lec::Scope indent;
  variadicOperation(result, operands,
                    [](auto op1, auto op2) { return op1 ^ op2; });
}

/// Helper function for performing a variadic operation: it executes a lambda
/// over a range of operands.
void Solver::Circuit::variadicOperation(
    Value result, OperandRange operands,
    llvm::function_ref<z3::expr(const z3::expr &, const z3::expr &)>
        operation) {
  LLVM_DEBUG(lec::dbgs() << "variadic operation\n");
  lec::Scope indent;
  // Vacuous base case.
  auto it = operands.begin();
  Value operand = *it;
  z3::expr varOp = exprTable.find(operand)->second;
  {
    LLVM_DEBUG(lec::dbgs() << "first operand:\n");
    lec::Scope indent;
    LLVM_DEBUG(lec::printValue(operand));
  }
  ++it;
  // Inductive step.
  while (it != operands.end()) {
    operand = *it;
    varOp = operation(varOp, exprTable.find(operand)->second);
    {
      LLVM_DEBUG(lec::dbgs() << "next operand:\n");
      lec::Scope indent;
      LLVM_DEBUG(lec::printValue(operand));
    }
    ++it;
  };
  constrainResult(result, varOp);
}

/// Allocates an IR value in the logical backend and returns its representing
/// expression.
z3::expr Solver::Circuit::allocateValue(Value value) {
  std::string valueName = name + "%" + std::to_string(assignments++);
  LLVM_DEBUG(lec::dbgs() << "allocating value:\n");
  lec::Scope indent;
  Type type = value.getType();
  assert(type.isSignlessInteger() && "Unsupported type");
  unsigned int width = type.getIntOrFloatBitWidth();
  // Technically allowed for the `hw` dialect but
  // disallowed for `comb` operations; should check separately.
  assert(width > 0 && "0-width integers are not supported"); // NOLINT
  z3::expr expr = solver.context.bv_const(valueName.c_str(), width);
  LLVM_DEBUG(lec::printExpr(expr));
  LLVM_DEBUG(lec::printValue(value));
  auto exprInsertion = exprTable.insert(std::pair(value, expr));
  assert(exprInsertion.second && "Value not inserted in expression table");
  Builder builder(solver.mlirCtx);
  StringAttr symbol = builder.getStringAttr(valueName);
  auto symInsertion = solver.symbolTable.insert(std::pair(symbol, value));
  assert(symInsertion.second && "Value not inserted in symbol table");
  return expr;
}

/// Allocates a constant value in the logical backend and returns its
/// representing expression.
void Solver::Circuit::allocateConstant(Value result, const APInt &value) {
  // `The constant operation produces a constant value
  //  of standard integer type without a sign`
  const z3::expr constant =
      solver.context.bv_val(value.getZExtValue(), value.getBitWidth());
  auto insertion = exprTable.insert(std::pair(result, constant));
  assert(insertion.second && "Constant not inserted in expression table");
  LLVM_DEBUG(lec::printExpr(constant));
  LLVM_DEBUG(lec::printValue(result));
}

/// Fetches the corresponding logical expression for a given IR value.
z3::expr Solver::Circuit::fetchExpr(Value &value) {
  z3::expr expr = exprTable.find(value)->second;
  lec::Scope indent;
  LLVM_DEBUG(lec::printExpr(expr));
  LLVM_DEBUG(lec::printValue(value));
  return expr;
}

/// Constrains the result of a MLIR operation to be equal a given logical
/// express, simulating an assignment.
void Solver::Circuit::constrainResult(Value &result, z3::expr &expr) {
  LLVM_DEBUG(lec::dbgs() << "constraining result:\n");
  lec::Scope indent;
  {
    LLVM_DEBUG(lec::dbgs() << "result expression:\n");
    lec::Scope indent;
    LLVM_DEBUG(lec::printExpr(expr));
  }
  z3::expr resExpr = allocateValue(result);
  z3::expr constraint = resExpr == expr;
  {
    LLVM_DEBUG(lec::dbgs() << "adding constraint:\n");
    lec::Scope indent;
    LLVM_DEBUG(lec::dbgs() << constraint.to_string() << "\n");
  }
  solver.solver.add(constraint);
}

/// Convert from bitvector to bool sort.
z3::expr Solver::Circuit::bvToBool(const z3::expr &condition) {
  // bitvector is true if it's different from 0
  return condition != 0;
}

/// Convert from a boolean sort to the corresponding 1-width bitvector.
z3::expr Solver::Circuit::boolToBv(const z3::expr &condition) {
  return z3::ite(condition, solver.context.bv_val(1, 1),
                 solver.context.bv_val(0, 1));
}
