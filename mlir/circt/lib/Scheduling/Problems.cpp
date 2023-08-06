//===- Problems.cpp - Modeling of scheduling problems ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements base classes for scheduling problems.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Problems.h"
#include "circt/Scheduling/DependenceIterator.h"

#include "mlir/IR/Operation.h"

using namespace circt;
using namespace circt::scheduling;
using namespace circt::scheduling::detail;

//===----------------------------------------------------------------------===//
// Problem
//===----------------------------------------------------------------------===//

LogicalResult Problem::insertDependence(Dependence dep) {
  Operation *src = dep.getSource();
  Operation *dst = dep.getDestination();

  // Fail early on invalid dependences (src == dst == null), and def-use
  // dependences that cannot be added because the source value is not the result
  // of an operation (e.g., a BlockArgument).
  if (!src || !dst)
    return failure();

  // record auxiliary dependences explicitly
  if (dep.isAuxiliary())
    auxDependences[dst].insert(src);

  // auto-register the endpoints
  operations.insert(src);
  operations.insert(dst);

  return success();
}

Problem::OperatorType Problem::getOrInsertOperatorType(StringRef name) {
  auto opr = OperatorType::get(containingOp->getContext(), name);
  operatorTypes.insert(opr);
  return opr;
}

Problem::DependenceRange Problem::getDependences(Operation *op) {
  return DependenceRange(DependenceIterator(*this, op),
                         DependenceIterator(*this, op, /*end=*/true));
}

Problem::PropertyStringVector Problem::getProperties(Operation *op) {
  PropertyStringVector psv;
  if (auto linkedOpr = getLinkedOperatorType(op))
    psv.emplace_back("linkedOpr", (*linkedOpr).str());
  if (auto startTime = getStartTime(op))
    psv.emplace_back("startTime", std::to_string(*startTime));
  return psv;
}

Problem::PropertyStringVector Problem::getProperties(Dependence dep) {
  return {};
}

Problem::PropertyStringVector Problem::getProperties(OperatorType opr) {
  PropertyStringVector psv;
  if (auto latency = getLatency(opr))
    psv.emplace_back("latency", std::to_string(*latency));
  return psv;
}

Problem::PropertyStringVector Problem::getProperties() { return {}; }

LogicalResult Problem::checkLinkedOperatorType(Operation *op) {
  if (!getLinkedOperatorType(op))
    return op->emitError("Operation is not linked to an operator type");
  if (!hasOperatorType(*getLinkedOperatorType(op)))
    return op->emitError("Operation uses an unregistered operator type");
  return success();
}

LogicalResult Problem::checkLatency(OperatorType opr) {
  if (!getLatency(opr))
    return getContainingOp()->emitError()
           << "Operator type '" << opr.getValue() << "' has no latency";

  return success();
}

LogicalResult Problem::check() {
  for (auto *op : getOperations())
    if (failed(checkLinkedOperatorType(op)))
      return failure();

  for (auto opr : getOperatorTypes())
    if (failed(checkLatency(opr)))
      return failure();

  return success();
}

LogicalResult Problem::verifyStartTime(Operation *op) {
  if (!getStartTime(op))
    return op->emitError("Operation has no start time");
  return success();
}

LogicalResult Problem::verifyPrecedence(Dependence dep) {
  Operation *i = dep.getSource();
  Operation *j = dep.getDestination();

  unsigned stI = *getStartTime(i);
  unsigned latI = *getLatency(*getLinkedOperatorType(i));
  unsigned stJ = *getStartTime(j);

  // check if i's result is available before j starts
  if (!(stI + latI <= stJ))
    return getContainingOp()->emitError()
           << "Precedence violated for dependence."
           << "\n  from: " << *i << ", result available in t=" << (stI + latI)
           << "\n  to:   " << *j << ", starts in t=" << stJ;

  return success();
}

LogicalResult Problem::verify() {
  for (auto *op : getOperations())
    if (failed(verifyStartTime(op)))
      return failure();

  for (auto *op : getOperations())
    for (auto &dep : getDependences(op))
      if (failed(verifyPrecedence(dep)))
        return failure();

  return success();
}

std::optional<unsigned> Problem::getEndTime(Operation *op) {
  if (auto startTime = getStartTime(op))
    if (auto opType = getLinkedOperatorType(op))
      if (auto latency = getLatency(*opType))
        return startTime.value() + latency.value();
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// CyclicProblem
//===----------------------------------------------------------------------===//

Problem::PropertyStringVector CyclicProblem::getProperties(Dependence dep) {
  auto psv = Problem::getProperties(dep);
  if (auto distance = getDistance(dep))
    psv.emplace_back("distance", std::to_string(*distance));
  return psv;
}

Problem::PropertyStringVector CyclicProblem::getProperties() {
  auto psv = Problem::getProperties();
  if (auto ii = getInitiationInterval())
    psv.emplace_back("II", std::to_string(*ii));
  return psv;
}

LogicalResult CyclicProblem::verifyPrecedence(Dependence dep) {
  Operation *i = dep.getSource();
  Operation *j = dep.getDestination();

  unsigned stI = *getStartTime(i);
  unsigned latI = *getLatency(*getLinkedOperatorType(i));
  unsigned stJ = *getStartTime(j);
  unsigned dist = getDistance(dep).value_or(0); // optional property
  unsigned ii = *getInitiationInterval();

  // check if i's result is available before j starts (dist iterations later)
  if (!(stI + latI <= stJ + dist * ii))
    return getContainingOp()->emitError()
           << "Precedence violated for dependence."
           << "\n  from: " << *i << ", result available in t=" << (stI + latI)
           << "\n  to:   " << *j << ", starts in t=" << stJ
           << "\n  dist: " << dist << ", II=" << ii;

  return success();
}

LogicalResult CyclicProblem::verifyInitiationInterval() {
  if (!getInitiationInterval() || *getInitiationInterval() == 0)
    return getContainingOp()->emitError("Invalid initiation interval");
  return success();
}

LogicalResult CyclicProblem::verify() {
  if (failed(verifyInitiationInterval()) || failed(Problem::verify()))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// ChainingProblem
//===----------------------------------------------------------------------===//

Problem::PropertyStringVector ChainingProblem::getProperties(Operation *op) {
  auto psv = Problem::getProperties(op);
  if (auto stic = getStartTimeInCycle(op))
    psv.emplace_back("start time in cycle", std::to_string(*stic));
  return psv;
}

Problem::PropertyStringVector ChainingProblem::getProperties(OperatorType opr) {
  auto psv = Problem::getProperties(opr);
  if (auto incDelay = getIncomingDelay(opr))
    psv.emplace_back("incoming delay", std::to_string(*incDelay));
  if (auto outDelay = getOutgoingDelay(opr))
    psv.emplace_back("outgoing delay", std::to_string(*outDelay));
  return psv;
}

LogicalResult ChainingProblem::checkDelays(OperatorType opr) {
  auto incomingDelay = getIncomingDelay(opr);
  auto outgoingDelay = getOutgoingDelay(opr);

  if (!incomingDelay || !outgoingDelay)
    return getContainingOp()->emitError()
           << "Missing delays for operator type '" << opr << "'";

  float iDel = *incomingDelay;
  float oDel = *outgoingDelay;

  if (iDel < 0.0f || oDel < 0.0f)
    return getContainingOp()->emitError()
           << "Negative delays for operator type '" << opr << "'";

  if (*getLatency(opr) == 0 && iDel != oDel)
    return getContainingOp()->emitError()
           << "Incoming & outgoing delay must be equal for zero-latency "
              "operator type '"
           << opr << "'";

  return success();
}

LogicalResult ChainingProblem::verifyStartTimeInCycle(Operation *op) {
  auto startTimeInCycle = getStartTimeInCycle(op);
  if (!startTimeInCycle || *startTimeInCycle < 0.0f)
    return op->emitError("Operation has no non-negative start time in cycle");
  return success();
}

LogicalResult ChainingProblem::verifyPrecedenceInCycle(Dependence dep) {
  // Auxiliary edges don't transport values.
  if (dep.isAuxiliary())
    return success();

  Operation *i = dep.getSource();
  Operation *j = dep.getDestination();

  unsigned stI = *getStartTime(i);
  unsigned latI = *getLatency(*getLinkedOperatorType(i));
  unsigned stJ = *getStartTime(j);

  // If `i` finishes a full time step earlier than `j`, its value is registered
  // and thereby available at physical time 0.0 in `j`'s start cycle.
  if (stI + latI < stJ)
    return success();

  // We have stI + latI == stJ, i.e. `i` ends in the same cycle as `j` starts.
  // If `i` is combinational, both ops also start in the same cycle, and we must
  // include `i`'s start time in that cycle in the path delay. Otherwise, `i`
  // started in an earlier cycle and just contributes its outgoing delay to the
  // path.
  float sticI = latI == 0 ? *getStartTimeInCycle(i) : 0.0f;
  float oDelI = *getOutgoingDelay(*getLinkedOperatorType(i));
  float sticJ = *getStartTimeInCycle(j);

  if (!(sticI + oDelI <= sticJ))
    return getContainingOp()->emitError()
           << "Precedence violated in cycle " << stJ << " for dependence:"
           << "\n  from: " << *i << ", result after z=" << (sticI + oDelI)
           << "\n  to:   " << *j << ", starts in z=" << sticJ;

  return success();
}

LogicalResult ChainingProblem::check() {
  if (failed(Problem::check()))
    return failure();

  for (auto opr : getOperatorTypes())
    if (failed(checkDelays(opr)))
      return failure();

  return success();
}

LogicalResult ChainingProblem::verify() {
  if (failed(Problem::verify()))
    return failure();

  for (auto *op : getOperations())
    if (failed(verifyStartTimeInCycle(op)))
      return failure();

  for (auto *op : getOperations())
    for (auto dep : getDependences(op))
      if (failed(verifyPrecedenceInCycle(dep)))
        return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// SharedOperatorsProblem
//===----------------------------------------------------------------------===//

Problem::PropertyStringVector
SharedOperatorsProblem::getProperties(OperatorType opr) {
  auto psv = Problem::getProperties(opr);
  if (auto limit = getLimit(opr))
    psv.emplace_back("limit", std::to_string(*limit));
  return psv;
}

LogicalResult SharedOperatorsProblem::checkLatency(OperatorType opr) {
  if (failed(Problem::checkLatency(opr)))
    return failure();

  auto limit = getLimit(opr);
  if (limit && *limit > 0 && *getLatency(opr) == 0)
    return getContainingOp()->emitError()
           << "Limited operator type '" << opr.getValue()
           << "' has zero latency.";
  return success();
}

LogicalResult SharedOperatorsProblem::verifyUtilization(OperatorType opr) {
  auto limit = getLimit(opr);
  if (!limit)
    return success();

  llvm::SmallDenseMap<unsigned, unsigned> nOpsPerTimeStep;
  for (auto *op : getOperations())
    if (opr == *getLinkedOperatorType(op))
      ++nOpsPerTimeStep[*getStartTime(op)];

  for (auto &kv : nOpsPerTimeStep)
    if (kv.second > *limit)
      return getContainingOp()->emitError()
             << "Operator type '" << opr.getValue() << "' is oversubscribed."
             << "\n  time step: " << kv.first
             << "\n  #operations: " << kv.second << "\n  limit: " << *limit;

  return success();
}

LogicalResult SharedOperatorsProblem::verify() {
  if (failed(Problem::verify()))
    return failure();

  for (auto opr : getOperatorTypes())
    if (failed(verifyUtilization(opr)))
      return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ModuloProblem
//===----------------------------------------------------------------------===//

LogicalResult ModuloProblem::verifyUtilization(OperatorType opr) {
  auto limit = getLimit(opr);
  if (!limit)
    return success();

  unsigned ii = *getInitiationInterval();
  llvm::SmallDenseMap<unsigned, unsigned> nOpsPerCongruenceClass;
  for (auto *op : getOperations())
    if (opr == *getLinkedOperatorType(op))
      ++nOpsPerCongruenceClass[*getStartTime(op) % ii];

  for (auto &kv : nOpsPerCongruenceClass)
    if (kv.second > *limit)
      return getContainingOp()->emitError()
             << "Operator type '" << opr.getValue() << "' is oversubscribed."
             << "\n  congruence class: " << kv.first
             << "\n  #operations: " << kv.second << "\n  limit: " << *limit;

  return success();
}

LogicalResult ModuloProblem::verify() {
  if (failed(CyclicProblem::verify()))
    return failure();

  // Don't call SharedOperatorsProblem::verify() here to prevent redundant
  // verification of the base problem.
  for (auto opr : getOperatorTypes())
    if (failed(verifyUtilization(opr)))
      return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Dependence
//===----------------------------------------------------------------------===//

Operation *Dependence::getSource() const {
  return isDefUse() ? defUse->get().getDefiningOp() : auxSrc;
}

Operation *Dependence::getDestination() const {
  return isDefUse() ? defUse->getOwner() : auxDst;
}

std::optional<unsigned> Dependence::getSourceIndex() const {
  if (!isDefUse())
    return std::nullopt;

  assert(defUse->get().isa<OpResult>() && "source is not an operation");
  return defUse->get().dyn_cast<OpResult>().getResultNumber();
}

std::optional<unsigned> Dependence::getDestinationIndex() const {
  if (!isDefUse())
    return std::nullopt;
  return defUse->getOperandNumber();
}

Dependence::TupleRepr Dependence::getAsTuple() const {
  return TupleRepr(getSource(), getDestination(), getSourceIndex(),
                   getDestinationIndex());
}

bool Dependence::operator==(const Dependence &other) const {
  return getAsTuple() == other.getAsTuple();
}

//===----------------------------------------------------------------------===//
// DependenceIterator
//===----------------------------------------------------------------------===//

DependenceIterator::DependenceIterator(Problem &problem, Operation *op,
                                       bool end)
    : problem(problem), op(op), operandIdx(0), auxPredIdx(0), auxPreds(nullptr),
      dep() {
  if (!end) {
    if (problem.auxDependences.count(op))
      auxPreds = &problem.auxDependences[op];

    findNextDependence();
  }
}

void DependenceIterator::findNextDependence() {
  // Yield dependences corresponding to values used by `op`'s operands...
  while (operandIdx < op->getNumOperands()) {
    dep = Dependence(&op->getOpOperand(operandIdx++));
    Operation *src = dep.getSource();

    // ... but only if they are outgoing from operations that are registered in
    // the scheduling problem.
    if (src && problem.hasOperation(src))
      return;
  }

  // Then, yield auxiliary dependences, if present.
  if (auxPreds && auxPredIdx < auxPreds->size()) {
    dep = Dependence((*auxPreds)[auxPredIdx++], op);
    return;
  }

  // An invalid dependence signals the end of iteration.
  dep = Dependence();
}
