//===- InferStateProperties.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Support/LLVM.h"

#define DEBUG_TYPE "arc-infer-state-properties"

using namespace circt;
using namespace arc;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static bool isConstZero(Value value) {
  if (auto constOp = value.getDefiningOp<hw::ConstantOp>())
    return constOp.getValue().isZero();

  return false;
}

static bool isConstTrue(Value value) {
  if (auto constOp = value.getDefiningOp<hw::ConstantOp>()) {
    return constOp.getValue().getBitWidth() == 1 &&
           constOp.getValue().isAllOnes();
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Reset and Enable property storages
//===----------------------------------------------------------------------===//

namespace {
/// Contains all the information needed to pass a detected reset to the rewriter
/// function.
struct ResetInfo {
  ResetInfo() = default;
  ResetInfo(std::function<Value(OpBuilder &)> &&constructInput,
            BlockArgument condition, bool isZeroReset)
      : constructInput(constructInput), condition(condition),
        isZeroReset(isZeroReset) {}

  ResetInfo(Value input, BlockArgument condition, bool isZeroReset)
      : ResetInfo([=](OpBuilder &) { return input; }, condition, isZeroReset) {}

  std::function<Value(OpBuilder &)> constructInput;
  BlockArgument condition;
  bool isZeroReset;

  operator bool() { return constructInput && condition; }
};

/// Contains all the information needed to pass a detected enable to the
/// rewriter function.
struct EnableInfo {
  EnableInfo() = default;
  EnableInfo(std::function<Value(OpBuilder &)> &&constructInput,
             BlockArgument condition, BlockArgument selfArg, bool isDisable)
      : constructInput(constructInput), condition(condition), selfArg(selfArg),
        isDisable(isDisable) {}

  EnableInfo(Value input, BlockArgument condition, BlockArgument selfArg,
             bool isDisable)
      : EnableInfo([=](OpBuilder &) { return input; }, condition, selfArg,
                   isDisable) {}

  std::function<Value(OpBuilder &)> constructInput;
  BlockArgument condition;
  BlockArgument selfArg;
  bool isDisable;

  operator bool() { return constructInput && condition && selfArg; }
};
} // namespace

//===----------------------------------------------------------------------===//
// Rewriter functions
//===----------------------------------------------------------------------===//

/// Take an arc and a detected reset per output value and apply it to the arc if
/// applicable (but does not change the state ops referring to the arc).
static LogicalResult applyResetTransformation(arc::DefineOp arcOp,
                                              ArrayRef<ResetInfo> resetInfos) {
  auto outputOp = cast<arc::OutputOp>(arcOp.getBodyBlock().getTerminator());

  assert(outputOp.getOutputs().size() == resetInfos.size() &&
         "required to pass the same amount of resets as outputs of the arc");

  for (auto info : resetInfos) {
    if (!info)
      return failure();

    // We can only pull out the reset to the whole arc when all the output
    // values have the same reset applied to them.
    // TODO: split the arcs such that there is one for each reset kind, however,
    // that requires a cost-model to not blow up binary-size too much
    if (!resetInfos.empty() &&
        (info.condition != resetInfos.back().condition ||
         info.isZeroReset != resetInfos.back().isZeroReset))
      return failure();

    // TODO: arc.state operation only supports resets to zero at the moment.
    if (!info.isZeroReset)
      return failure();
  }

  if (resetInfos.empty())
    return failure();

  OpBuilder builder(outputOp);

  for (size_t i = 0, e = outputOp.getOutputs().size(); i < e; ++i) {
    auto *defOp = outputOp.getOperands()[i].getDefiningOp();
    outputOp.getOperands()[i].replaceUsesWithIf(
        resetInfos[i].constructInput(builder),
        [](OpOperand &op) { return isa<arc::OutputOp>(op.getOwner()); });

    if (defOp && defOp->getResult(0).use_empty())
      defOp->erase();
  }

  return success();
}

/// Transform the given state operation to match the changes done to the arc in
/// 'applyResetTransformation' without any additional checks.
static void setResetOperandOfStateOp(arc::StateOp stateOp,
                                     unsigned resetConditionIndex) {
  Value resetCond = stateOp.getInputs()[resetConditionIndex];
  ImplicitLocOpBuilder builder(stateOp.getLoc(), stateOp);

  if (stateOp.getEnable())
    resetCond = builder.create<comb::AndOp>(stateOp.getEnable(), resetCond);

  if (stateOp.getReset())
    resetCond = builder.create<comb::OrOp>(stateOp.getReset(), resetCond);

  stateOp.getResetMutable().assign(resetCond);
}

/// Take an arc and a detected enable per output value and apply it to the given
/// state if applicable (no changes required to the arc::DefineOp operation for
/// enables).
static LogicalResult
applyEnableTransformation(arc::DefineOp arcOp, arc::StateOp stateOp,
                          ArrayRef<EnableInfo> enableInfos) {
  auto outputOp = cast<arc::OutputOp>(arcOp.getBodyBlock().getTerminator());

  assert(outputOp.getOutputs().size() == enableInfos.size() &&
         "required to pass the same amount of enables as outputs of the arc");

  for (auto info : enableInfos) {
    if (!info)
      return failure();

    // We can only pull out the enable to the whole arc when all the output
    // values have the same enable applied to them.
    // TODO: split the arcs such that there is one for each enable kind,
    // however, this requires a cost-model to not blow up binary-size too much.
    if (!enableInfos.empty() &&
        (info.condition != enableInfos.back().condition ||
         info.isDisable != enableInfos.back().isDisable))
      return failure();
  }

  if (enableInfos.empty())
    return failure();

  if (!enableInfos[0].condition.hasOneUse())
    return failure();

  ImplicitLocOpBuilder builder(stateOp.getLoc(), stateOp);
  SmallVector<Value> inputs(stateOp.getInputs());

  Value enableCond =
      stateOp.getInputs()[enableInfos[0].condition.getArgNumber()];
  Value one = builder.create<hw::ConstantOp>(builder.getI1Type(), -1);
  if (enableInfos[0].isDisable) {
    inputs[enableInfos[0].condition.getArgNumber()] =
        builder.create<hw::ConstantOp>(builder.getI1Type(), 0);
    enableCond = builder.create<comb::XorOp>(enableCond, one);
  } else {
    inputs[enableInfos[0].condition.getArgNumber()] = one;
  }

  if (stateOp.getEnable())
    enableCond = builder.create<comb::AndOp>(stateOp.getEnable(), enableCond);

  stateOp.getEnableMutable().assign(enableCond);

  for (size_t i = 0, e = outputOp.getOutputs().size(); i < e; ++i) {
    if (enableInfos[i].selfArg.hasOneUse())
      inputs[enableInfos[i].selfArg.getArgNumber()] =
          builder.create<hw::ConstantOp>(stateOp.getLoc(),
                                         enableInfos[i].selfArg.getType(), 0);
  }

  stateOp.getInputsMutable().assign(inputs);
  return success();
}

//===----------------------------------------------------------------------===//
// Pattern detectors
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Reset Patterns

/// A reset represented with a single mux operation.
/// out = mux(resetCondition, 0, arcArgument)
/// ==>
/// return arcArgument directly and add resetCondition to the StateOp
static ResetInfo getIfMuxBasedReset(OpOperand &output) {
  assert(isa<arc::OutputOp>(output.getOwner()) &&
         "value has to be returned by the arc");

  if (auto mux = output.get().getDefiningOp<comb::MuxOp>()) {
    if (!isConstZero(mux.getTrueValue()))
      return {};

    if (!mux.getResult().hasOneUse())
      return {};

    if (auto condArg = mux.getCond().dyn_cast<BlockArgument>())
      return ResetInfo(mux.getFalseValue(), condArg, true);
  }

  return {};
}

/// A reset represented by an AND and XOR operation for i1 values only.
/// out = and(X); X being a list containing all of
///   {xor(resetCond, true), arcArgument}
/// ==>
/// out = and(X\xor(resetCond, true)) + add resetCond to StateOp
static ResetInfo getIfAndBasedReset(OpOperand &output) {
  assert(isa<arc::OutputOp>(output.getOwner()) &&
         "value has to be returned by the arc");

  if (auto andOp = output.get().getDefiningOp<comb::AndOp>()) {
    if (!andOp.getResult().getType().isInteger(1))
      return {};

    if (!andOp.getResult().hasOneUse())
      return {};

    for (auto &operand : andOp->getOpOperands()) {
      if (auto xorOp = operand.get().getDefiningOp<comb::XorOp>();
          xorOp && xorOp->getNumOperands() == 2 &&
          xorOp.getResult().hasOneUse()) {
        if (auto condArg = xorOp.getInputs()[0].dyn_cast<BlockArgument>()) {
          if (xorOp.getInputs().size() != 2 ||
              !isConstTrue(xorOp.getInputs()[1]))
            continue;

          const unsigned condOutputNumber = operand.getOperandNumber();
          auto inputConstructor = [=](OpBuilder &builder) -> Value {
            if (andOp->getNumOperands() > 2) {
              builder.setInsertionPoint(andOp);
              auto copy = cast<comb::AndOp>(builder.clone(*andOp));
              copy.getInputsMutable().erase(condOutputNumber);
              return copy->getResult(0);
            }

            return andOp->getOperand(!condOutputNumber);
          };

          return ResetInfo(inputConstructor, condArg, true);
        }
      }
    }
  }

  return {};
}

//===----------------------------------------------------------------------===//
// Enable Patterns

/// Just a helper function for the following two patterns.
static EnableInfo checkOperandsForEnable(arc::StateOp stateOp, Value selfArg,
                                         Value cond, unsigned outputNr,
                                         bool isDisable) {
  if (auto trueArg = selfArg.dyn_cast<BlockArgument>()) {
    if (stateOp.getInputs()[trueArg.getArgNumber()] !=
        stateOp.getResult(outputNr))
      return {};

    if (auto condArg = cond.dyn_cast<BlockArgument>())
      return EnableInfo(selfArg, condArg, trueArg, isDisable);
  }

  return {};
}

/// An enable represented by a single mux operation.
/// out = mux(enableCond, x, arcArgument) where x is the 'out' of the last cycle
/// ==>
/// out = arcArgument + set enableCond as enable operand to the StateOp
static EnableInfo getIfMuxBasedEnable(OpOperand &output, StateOp stateOp) {
  assert(isa<arc::OutputOp>(output.getOwner()) &&
         "value has to be returned by the arc");

  if (auto mux = output.get().getDefiningOp<comb::MuxOp>()) {
    if (!mux.getResult().hasOneUse())
      return {};

    return checkOperandsForEnable(stateOp, mux.getFalseValue(), mux.getCond(),
                                  output.getOperandNumber(), false);
  }

  return {};
}

/// A negated enable represented by a single mux operation.
/// out = mux(enableCond, arcArgument, x) where x is the 'out' of the last cycle
/// ==>
/// out = arcArgument + set xor(enableCond, true) as enable operand to the
/// StateOp
static EnableInfo getIfMuxBasedDisable(OpOperand &output, StateOp stateOp) {
  assert(isa<arc::OutputOp>(output.getOwner()) &&
         "value has to be returned by the arc");

  if (auto mux = output.get().getDefiningOp<comb::MuxOp>()) {
    if (!mux.getResult().hasOneUse())
      return {};

    return checkOperandsForEnable(stateOp, mux.getTrueValue(), mux.getCond(),
                                  output.getOperandNumber(), true);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// Combine all the patterns
//===----------------------------------------------------------------------===//

/// Combine all the reset patterns to one.
ResetInfo computeResetInfoFromPattern(OpOperand &output) {
  auto resetInfo = getIfMuxBasedReset(output);

  if (!resetInfo)
    resetInfo = getIfAndBasedReset(output);

  return resetInfo;
}

/// Combine all the enable patterns to one.
EnableInfo computeEnableInfoFromPattern(OpOperand &output, StateOp stateOp) {
  auto enableInfo = getIfMuxBasedEnable(output, stateOp);

  if (!enableInfo)
    enableInfo = getIfMuxBasedDisable(output, stateOp);

  return enableInfo;
}

//===----------------------------------------------------------------------===//
// DetectResets pass
//===----------------------------------------------------------------------===//

namespace {
struct InferStatePropertiesPass
    : public InferStatePropertiesBase<InferStatePropertiesPass> {
  InferStatePropertiesPass() = default;
  InferStatePropertiesPass(const InferStatePropertiesPass &pass)
      : InferStatePropertiesPass() {}

  void runOnOperation() override;
  void runOnStateOp(arc::StateOp stateOp, arc::DefineOp arc,
                    DenseMap<arc::DefineOp, unsigned> &resetConditionMap);

  Statistic addedEnables{this, "added-enables",
                         "Enables added explicitly to a StateOp"};
  Statistic addedResets{this, "added-resets",
                        "Resets added explicitly to a StateOp"};
  Statistic missedEnables{
      this, "missed-enables",
      "Detected enables that could not be added explicitly to a StateOp"};
  Statistic missedResets{
      this, "missed-resets",
      "Detected resets that could not be added explicitly to a StateOp"};
};
} // namespace

void InferStatePropertiesPass::runOnOperation() {
  SymbolTableCollection symbolTable;

  DenseMap<arc::DefineOp, unsigned> resetConditionMap;
  getOperation()->walk([&](arc::StateOp stateOp) {
    auto arc =
        cast<arc::DefineOp>(cast<mlir::CallOpInterface>(stateOp.getOperation())
                                .resolveCallable(&symbolTable));
    runOnStateOp(stateOp, arc, resetConditionMap);
  });
}

void InferStatePropertiesPass::runOnStateOp(
    arc::StateOp stateOp, arc::DefineOp arc,
    DenseMap<arc::DefineOp, unsigned> &resetConditionMap) {

  if (stateOp.getLatency() < 1)
    return;

  auto outputOp = cast<arc::OutputOp>(arc.getBodyBlock().getTerminator());
  const unsigned visitedNoChange = -1;

  // Check for reset patterns, we only have to do this once per arc::DefineOp
  // and store the result for later arc::StateOps referring to the same arc.
  if (!resetConditionMap.count(arc)) {
    SmallVector<ResetInfo> resetInfos;
    int numResets = 0;
    ;
    for (auto &output : outputOp->getOpOperands()) {
      auto resetInfo = computeResetInfoFromPattern(output);
      resetInfos.push_back(resetInfo);
      if (resetInfo)
        ++numResets;
    }

    // Rewrite the arc::DefineOp if valid
    auto result = applyResetTransformation(arc, resetInfos);
    if ((succeeded(result) && resetInfos[0]))
      resetConditionMap[arc] = resetInfos[0].condition.getArgNumber();
    else
      resetConditionMap[arc] = visitedNoChange;

    if (failed(result))
      missedResets += numResets;
  }

  // Apply resets to the state operation.
  if (resetConditionMap.count(arc) &&
      resetConditionMap[arc] != visitedNoChange) {
    setResetOperandOfStateOp(stateOp, resetConditionMap[arc]);
    ++addedResets;
  }

  // Check for enable patterns.
  SmallVector<EnableInfo> enableInfos;
  int numEnables = 0;
  for (OpOperand &output : outputOp->getOpOperands()) {
    auto enableInfo = computeEnableInfoFromPattern(output, stateOp);
    enableInfos.push_back(enableInfo);
    if (enableInfo)
      ++numEnables;
  }

  // Apply enable patterns.
  if (!failed(applyEnableTransformation(arc, stateOp, enableInfos)))
    ++addedEnables;
  else
    missedEnables += numEnables;
}

std::unique_ptr<Pass> arc::createInferStatePropertiesPass() {
  return std::make_unique<InferStatePropertiesPass>();
}
