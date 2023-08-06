//===- ExplicitRegs.cpp - Explicit regs pass --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the explicit regs pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Support/BackedgeBuilder.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace circt;
using namespace pipeline;

namespace {

class ExplicitRegsPass : public ExplicitRegsBase<ExplicitRegsPass> {
public:
  void runOnOperation() override;

private:
  // Recursively routes value v backwards through the pipeline, adding new
  // registers to 'stage' if the value was not already registered in the stage.
  // Returns the registerred version of 'v' through 'stage'.
  Value routeThroughStage(Value v, Block *stage);

  // Returns the distance between two stages in the pipeline. The distance is
  // defined wrt. the ordered stages of the pipeline.
  int64_t stageDistance(Block *from, Block *to);

  struct RoutedValue {
    Backedge v;
    // If true, this value is routed through a stage as a register, else
    // it is routed through a stage as a pass-through.e
    bool isReg;
  };
  // A mapping storing whether a given stage register constains a registerred
  // version of a given value. The registered version will be a backedge during
  // pipeline body analysis. Once the entire body has been analyzed, the
  // pipeline.stage operations will be replaced with pipeline.ss.reg
  // operations containing the requested regs, and the backedge will be
  // replaced. MapVector ensures deterministic iteration order, which in turn
  // ensures determinism during stage op IR emission.
  DenseMap<Block *, llvm::MapVector<Value, RoutedValue>> stageRegOrPassMap;

  // A mapping between stages and their index in the pipeline.
  llvm::DenseMap<Block *, unsigned> stageMap;

  std::shared_ptr<BackedgeBuilder> bb;
};

} // end anonymous namespace

int64_t ExplicitRegsPass::stageDistance(Block *from, Block *to) {
  int64_t fromStage = stageMap[from];
  int64_t toStage = stageMap[to];
  return toStage - fromStage;
}

// NOLINTNEXTLINE(misc-no-recursion)
Value ExplicitRegsPass::routeThroughStage(Value v, Block *stage) {
  Value retVal = v;
  Block *definingStage = retVal.getParentBlock();

  // Is the value defined in the current stage?
  if (definingStage == stage)
    return retVal;

  auto regIt = stageRegOrPassMap[stage].find(retVal);
  if (regIt != stageRegOrPassMap[stage].end()) {
    // 'v' is already routed through 'stage' - return the registered/passed
    // version.
    return regIt->second.v;
  }

  // Is the value a constant? If so, we allow it; constants are special cases
  // which are allowed to be used in any stage.
  auto *definingOp = retVal.getDefiningOp();
  if (definingOp && definingOp->hasTrait<OpTrait::ConstantLike>())
    return retVal;

  // Value is defined somewhere before the provided stage - route it through the
  // stage, and recurse to the predecessor stage.
  int64_t valueLatency = 0;
  if (auto latencyOp = dyn_cast_or_null<LatencyOp>(definingOp))
    valueLatency = latencyOp.getLatency();

  // A value should be registered in this stage if the latency of the value
  // is less than the distance between the current stage and the defining stage.
  bool isReg = valueLatency < stageDistance(definingStage, stage);
  auto valueBackedge = bb->get(retVal.getType());
  stageRegOrPassMap[stage].insert({retVal, {valueBackedge, isReg}});
  retVal = valueBackedge;

  // Recurse - recursion will only create a new backedge if necessary.
  Block *stagePred = stage->getSinglePredecessor();
  assert(stagePred && "Expected stage to have a single predecessor");
  routeThroughStage(v, stagePred);
  return retVal;
}

void ExplicitRegsPass::runOnOperation() {
  ScheduledPipelineOp pipeline = getOperation();
  OpBuilder b(getOperation().getContext());
  bb = std::make_shared<BackedgeBuilder>(b, getOperation().getLoc());

  // Cache external-like inputs in a set for fast lookup. This also includes
  // clock, reset, and stall.
  llvm::DenseSet<Value> extLikeInputs;
  for (auto extInput : pipeline.getInnerExtInputs())
    extLikeInputs.insert(extInput);
  extLikeInputs.insert(pipeline.getInnerClock());
  extLikeInputs.insert(pipeline.getInnerReset());
  if (pipeline.hasStall())
    extLikeInputs.insert(pipeline.getInnerStall());

  // Iterate over the pipeline body in-order (!).
  stageMap = pipeline.getStageMap();
  for (Block *stage : pipeline.getOrderedStages()) {
    // Walk the stage body - we do this since register materialization needs
    // to consider all levels of nesting within the stage.
    stage->walk([&](Operation *op) {
      // Check the operands of this operation to see if any of them cross a
      // stage boundary.
      for (OpOperand &operand : op->getOpOperands()) {
        if (extLikeInputs.contains(operand.get())) {
          // Never route external inputs through a stage.
          continue;
        }
        if (getParentStageInPipeline(pipeline, operand.get()) == stage) {
          // The operand is defined by some operation or block which ultimately
          // resides within the current pipeline stage. No routing needed.
          continue;
        }
        Value reroutedValue = routeThroughStage(operand.get(), stage);
        if (reroutedValue != operand.get())
          op->setOperand(operand.getOperandNumber(), reroutedValue);
      }
    });
  }

  auto *ctx = &getContext();

  // All values have been recorded through the stages. Now, add registers to the
  // stage blocks.
  for (auto &[stage, regMap] : stageRegOrPassMap) {
    // Gather register inputs to this stage, either from a predecessor stage
    // or from the original op.
    llvm::SmallVector<Value> regIns, passIns;
    Block *predecessorStage = stage->getSinglePredecessor();
    auto predStageRegOrPassMap = stageRegOrPassMap.find(predecessorStage);
    assert(predecessorStage && "Stage should always have a single predecessor");
    for (auto &[value, backedge] : regMap) {
      if (predStageRegOrPassMap != stageRegOrPassMap.end()) {
        // Grab the value if passed through the predecessor stage, else,
        // use the raw value.
        auto predRegIt = predStageRegOrPassMap->second.find(value);
        if (predRegIt != predStageRegOrPassMap->second.end()) {
          if (backedge.isReg)
            regIns.push_back(predRegIt->second.v);
          else
            passIns.push_back(predRegIt->second.v);
          continue;
        }
      }

      // Not passed through the stage - must be the original value.
      if (backedge.isReg)
        regIns.push_back(value);
      else
        passIns.push_back(value);
    }

    // Replace the predecessor stage terminator, which feeds this stage, with
    // a new terminator that has materialized arguments.
    StageOp terminator = cast<StageOp>(predecessorStage->getTerminator());
    b.setInsertionPoint(terminator);
    b.create<StageOp>(terminator.getLoc(), terminator.getNextStage(), regIns,
                      passIns);
    terminator.erase();

    // ... add arguments to the next stage. Registers first, then passthroughs.
    llvm::SmallVector<Type> regAndPassTypes;
    llvm::append_range(regAndPassTypes, ValueRange(regIns).getTypes());
    llvm::append_range(regAndPassTypes, ValueRange(passIns).getTypes());
    for (auto [i, type] : llvm::enumerate(regAndPassTypes))
      stage->insertArgument(i, type, UnknownLoc::get(ctx));

    // Replace backedges for the next stage with the new arguments.
    for (auto it : llvm::enumerate(regMap)) {
      auto index = it.index();
      auto &[value, backedge] = it.value();
      backedge.v.setValue(stage->getArgument(index));
    }
  }

  // Clear internal state. See https://github.com/llvm/circt/issues/3235
  stageRegOrPassMap.clear();
}

std::unique_ptr<mlir::Pass> circt::pipeline::createExplicitRegsPass() {
  return std::make_unique<ExplicitRegsPass>();
}
