//===- IsolateClocks.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-isolate-clocks"

using namespace circt;
using namespace arc;

//===----------------------------------------------------------------------===//
// Datastructures
//===----------------------------------------------------------------------===//

namespace {
/// Represents a not-yet materialized clock-domain.
class ClockDomain {
public:
  ClockDomain(Value clock, MLIRContext *context)
      : clock(clock), domainBlock(std::make_unique<Block>()),
        builder(OpBuilder(context)) {
    builder.setInsertionPointToStart(domainBlock.get());
  }

  /// Moves an operation into the clock domain if it is not already in there.
  /// Returns true if the operation was moved.
  bool moveToDomain(Operation *op);
  /// Moves all non-clocked fan-in operations that are not also used outside the
  /// clock domain into the clock domain.
  void sinkFanIn(SmallVectorImpl<Value> &);
  /// Computes all values used from outside this clock domain and all values
  /// defined in this clock domain that are used outside.
  void computeCrossingValues(SmallVectorImpl<Value> &inputs,
                             SmallVectorImpl<Value> &outputs);
  /// Add the terminator, materialize the clock-domain and return the
  /// operation. After calling this function, the other member functions and
  /// fields should not be used anymore.
  ClockDomainOp materialize(OpBuilder &materializeBuilder, Location loc);

private:
  Value clock;
  std::unique_ptr<Block> domainBlock;
  OpBuilder builder;
};
} // namespace

bool ClockDomain::moveToDomain(Operation *op) {
  assert(op != nullptr);
  // Do not move if already in the block.
  if (op->getBlock() == domainBlock.get())
    return false;

  builder.setInsertionPointToStart(domainBlock.get());

  // Perform the move
  op->remove();
  builder.insert(op);

  return true;
}

void ClockDomain::sinkFanIn(SmallVectorImpl<Value> &worklist) {
  while (!worklist.empty()) {
    auto *op = worklist.pop_back_val().getDefiningOp();
    // Ignore block arguments
    if (!op)
      continue;
    // TODO: if we find a clock domain with the same clock we should merge it.
    // Otherwise ignore it.
    if (isa<ClockDomainOp>(op))
      continue;
    if (auto clockedOp = dyn_cast<ClockedOpInterface>(op);
        clockedOp && clockedOp.isClocked())
      continue;
    // Don't pull in operations that have other users outside this domain. We
    // don't want do duplicate ops to keep binary size as small as possible and
    // avoid computing the same thing multiple times. This is also important
    // because we want every path from domain input to domain output to have at
    // least one cycle latency, ideally every output is directly from a state op
    // with greater than 0 latency such that we don't have to insert additional
    // storage slots later on.
    if (llvm::any_of(op->getUsers(), [&](auto *user) {
          return user->getBlock() != domainBlock.get();
        }))
      continue;

    if (moveToDomain(op))
      worklist.append(op->getOperands().begin(), op->getOperands().end());
  }
}

ClockDomainOp ClockDomain::materialize(OpBuilder &materializeBuilder,
                                       Location loc) {
  builder.setInsertionPointToEnd(domainBlock.get());
  SmallVector<Value> inputs, outputs;
  computeCrossingValues(inputs, outputs);

  // Add the terminator and clock domain outputs and rewire the SSA value uses
  auto outputOp = builder.create<arc::OutputOp>(loc, outputs);
  auto clockDomainOp = materializeBuilder.create<ClockDomainOp>(
      loc, ValueRange(outputs).getTypes(), inputs, clock);
  for (auto [domainOutput, val] :
       llvm::zip(clockDomainOp.getOutputs(), outputOp->getOperands())) {
    val.replaceUsesWithIf(domainOutput, [&](OpOperand &operand) {
      return operand.getOwner()->getBlock() != domainBlock.get();
    });
  }

  // Add arguments and inputs to the clock domain operation and rewire the SSA
  // value uses.
  domainBlock->addArguments(ValueRange(inputs).getTypes(),
                            SmallVector<Location>(inputs.size(), loc));
  for (auto [domainArg, val] :
       llvm::zip(domainBlock->getArguments(), clockDomainOp.getInputs())) {
    val.replaceUsesWithIf(domainArg, [&](OpOperand &operand) {
      return operand.getOwner()->getBlock() == domainBlock.get();
    });
  }
  clockDomainOp->getRegion(0).push_back(domainBlock.release());

  return clockDomainOp;
}

void ClockDomain::computeCrossingValues(SmallVectorImpl<Value> &inputs,
                                        SmallVectorImpl<Value> &outputs) {
  DenseSet<Value> inputSet, outputSet;
  for (auto &op : *domainBlock) {
    for (auto operand : op.getOperands()) {
      auto *defOp = operand.getDefiningOp();
      if (!defOp || defOp->getBlock() != domainBlock.get()) {
        if (inputSet.insert(operand).second)
          inputs.push_back(operand);
      }
    }
    for (auto result : op.getResults()) {
      if (llvm::any_of(result.getUsers(), [&](auto *user) {
            return user->getBlock() != domainBlock.get();
          })) {
        if (outputSet.insert(result).second)
          outputs.push_back(result);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct IsolateClocksPass : public IsolateClocksBase<IsolateClocksPass> {
  void runOnOperation() override;
  LogicalResult runOnModule(hw::HWModuleOp module);
};
} // namespace

void IsolateClocksPass::runOnOperation() {
  for (auto module : getOperation().getOps<hw::HWModuleOp>())
    if (failed(runOnModule(module)))
      return signalPassFailure();
}

LogicalResult IsolateClocksPass::runOnModule(hw::HWModuleOp module) {
  llvm::MapVector<Value, SmallVector<ClockedOpInterface>> clocks;

  // Check preconditions and collect clocked operations
  for (auto &op : *module.getBodyBlock()) {
    // Nested regions not supported for now. Since different regions have
    // different semantics we need to special case all region operations we
    // want to support. An interesting one might be scf::IfOp or similar, but
    // when it contains clocked operations with different clocks, it needs to
    // be split up, etc.
    if (op.getNumRegions() != 0 && !isa<ClockDomainOp>(&op))
      return op.emitOpError("operations with regions not supported yet!");

    if (auto clockedOp = dyn_cast<ClockedOpInterface>(&op);
        clockedOp && clockedOp.isClocked())
      clocks[clockedOp.getClock()].push_back(clockedOp);
  }

  SmallVector<Value> worklist;
  // Construct the domains clock by clock. This makes handling of
  // inter-clock-domain connections considerably easier.
  for (auto [clock, clockedOps] : clocks) {
    ClockDomain domain(clock, module.getContext());

    // Move all the clocked operations into the domain, op by op, and pull in
    // their fan-in immediately after to have a nice op ordering inside the
    // domain.
    for (auto op : clockedOps) {
      worklist.clear();
      if (domain.moveToDomain(op)) {
        op.eraseClock();
        worklist.append(op->getOperands().begin(), op->getOperands().end());
        domain.sinkFanIn(worklist);
      }
    }

    // Materialize the actual clock domain operation.
    OpBuilder builder(module.getBodyBlock()->getTerminator());
    domain.materialize(builder, module.getLoc());
  }

  return success();
}

std::unique_ptr<Pass> arc::createIsolateClocksPass() {
  return std::make_unique<IsolateClocksPass>();
}
