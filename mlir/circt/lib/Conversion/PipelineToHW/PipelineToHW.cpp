//===- PipelineToHW.cpp - Translate Pipeline into HW ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main Pipeline to HW Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/PipelineToHW.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Pipeline/PipelineOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace pipeline;

namespace {
static constexpr std::string_view kEnablePortName = "enable";
static constexpr std::string_view kStallPortName = "stall";
static constexpr std::string_view kClockPortName = "clk";
static constexpr std::string_view kResetPortName = "rst";
static constexpr std::string_view kValidPortName = "valid";
} // namespace

// Inlines the module pointed to by 'inst' if the module is empty. This assumes
// that 'inst' is the only user of the module. Furthermore, will remove the
// inlined module(!). Should probably implement some more generic inlining code
// for this, but it's simple enough to do when we know that the module is empty.
static void inlineAndEraseIfEmpty(hw::InstanceOp inst) {
  auto mod = cast<hw::HWModuleLike>(inst.getReferencedModule());
  if (mod->getNumRegions() == 0)
    return; // Nothing to do.

  Block &body = mod->getRegion(0).front();
  auto &ops = body.getOperations();
  if (ops.size() > 1)
    return; // non-empty

  hw::OutputOp output = cast<hw::OutputOp>(ops.front());
  DenseMap<Value, Value> valueMapping;
  for (auto [instOperand, modOperand] :
       llvm::zip(inst.getOperands(), output.getOperands()))
    valueMapping[modOperand] = instOperand;
  llvm::SmallVector<Value> returnValues;
  for (auto result : output.getOperands())
    returnValues.push_back(valueMapping[result]);

  // Replace the instance results with the mapped return values.
  inst.replaceAllUsesWith(returnValues);
  inst.erase();
  mod.erase();
}

// A class for generalizing pipeline lowering for both the inline and outlined
// implementation.
class PipelineLowering {
public:
  PipelineLowering(size_t pipelineID, ScheduledPipelineOp pipeline,
                   OpBuilder &builder, bool clockGateRegs)
      : pipelineID(pipelineID), pipeline(pipeline), builder(builder),
        clockGateRegs(clockGateRegs) {
    parentClk = pipeline.getClock();
    parentRst = pipeline.getReset();
    parentModule = pipeline->getParentOfType<hw::HWModuleOp>();
  }
  virtual ~PipelineLowering() = default;

  virtual LogicalResult run() = 0;

  // Arguments used for emitting the body of a stage module. These values must
  // be within the scope of the stage module body.
  struct StageArgs {
    ValueRange data;
    Value enable;
    Value stall;
    Value clock;
    Value reset;
  };

  // Arguments used for returning the results from a stage. These values must
  // be within the scope of the stage module body.
  struct StageReturns {
    llvm::SmallVector<Value> regs;
    llvm::SmallVector<Value> passthroughs;
    Value valid;
  };

  virtual FailureOr<StageReturns>
  lowerStage(Block *stage, StageArgs args, size_t stageIndex,
             llvm::ArrayRef<Attribute> inputNames = {}) = 0;

  StageReturns emitStageBody(Block *stage, StageArgs args,
                             llvm::ArrayRef<Attribute> registerNames,
                             size_t stageIndex = -1) {
    assert(args.enable && "enable not set");
    auto *terminator = stage->getTerminator();

    // Move the stage operations into the current insertion point.
    for (auto &op : llvm::make_early_inc_range(*stage)) {
      if (&op == terminator)
        continue;

      if (auto latencyOp = dyn_cast<LatencyOp>(op)) {
        // For now, just directly emit the body of the latency op. The latency
        // op is mainly used during register materialization. At a later stage,
        // we may want to add some TCL-related things here to communicate
        // multicycle paths.
        Block *latencyOpBody = latencyOp.getBodyBlock();
        for (auto &innerOp :
             llvm::make_early_inc_range(latencyOpBody->without_terminator()))
          innerOp.moveBefore(builder.getInsertionBlock(),
                             builder.getInsertionPoint());
        latencyOp.replaceAllUsesWith(
            latencyOpBody->getTerminator()->getOperands());
        latencyOp.erase();
      } else {
        op.moveBefore(builder.getInsertionBlock(), builder.getInsertionPoint());
      }
    }

    StageReturns rets;
    auto stageOp = dyn_cast<StageOp>(terminator);
    if (!stageOp) {
      assert(isa<ReturnOp>(terminator) && "expected ReturnOp");
      // This was the pipeline return op - the return op/last stage doesn't
      // register its operands, hence, all return operands are passthrough
      // and the valid signal is equal to the unregistered enable signal.
      rets.passthroughs = terminator->getOperands();
      rets.valid = args.enable;
      return rets;
    }

    assert(registerNames.size() == stageOp.getRegisters().size() &&
           "register names and registers must be the same size");

    // Build data registers.
    auto stageRegPrefix = getStageRegPrefix(stageIndex);
    auto loc = stageOp->getLoc();

    // Build the clock enable signal: enable && !stall (if applicable)
    Value stageValidAndNotStalled = args.enable;
    Value notStalled;
    bool hasStall = static_cast<bool>(args.stall);
    if (hasStall) {
      notStalled = comb::createOrFoldNot(loc, args.stall, builder);
      stageValidAndNotStalled =
          builder.create<comb::AndOp>(loc, stageValidAndNotStalled, notStalled);
    }

    Value notStalledClockGate;
    if (this->clockGateRegs) {
      // Create the top-level clock gate.
      notStalledClockGate = builder.create<seq::ClockGateOp>(
          loc, args.clock, stageValidAndNotStalled, /*test_enable=*/Value());
    }

    for (auto it : llvm::enumerate(stageOp.getRegisters())) {
      auto regIdx = it.index();
      auto regIn = it.value();

      StringAttr regName = registerNames[regIdx].cast<StringAttr>();
      Value dataReg;
      if (this->clockGateRegs) {
        // Use the clock gate instead of clock enable.
        Value currClockGate = notStalledClockGate;
        for (auto hierClockGateEnable : stageOp.getClockGatesForReg(regIdx)) {
          // Create clock gates for any hierarchically nested clock gates.
          currClockGate = builder.create<seq::ClockGateOp>(
              loc, currClockGate, hierClockGateEnable, /*test_enable=*/Value());
        }
        dataReg = builder.create<seq::CompRegOp>(stageOp->getLoc(), regIn,
                                                 currClockGate, regName);
      } else {
        // Only clock-enable the register if the pipeline is stallable.
        // For non-stallable pipelines, a data register can always be clocked.
        if (hasStall) {
          dataReg = builder.create<seq::CompRegClockEnabledOp>(
              stageOp->getLoc(), regIn, args.clock, stageValidAndNotStalled,
              regName);
        } else {
          dataReg = builder.create<seq::CompRegOp>(stageOp->getLoc(), regIn,
                                                   args.clock, regName);
        }
      }
      rets.regs.push_back(dataReg);
    }

    // Build valid register. The valid register is always reset to 0, and
    // clock enabled when not stalling.
    auto validRegName = (stageRegPrefix.strref() + "_valid").str();
    Value validRegResetVal =
        builder.create<hw::ConstantOp>(terminator->getLoc(), APInt(1, 0, false))
            .getResult();
    if (hasStall) {
      rets.valid = builder.create<seq::CompRegClockEnabledOp>(
          loc, args.enable, args.clock, notStalled, args.reset,
          validRegResetVal, validRegName);
    } else {
      rets.valid = builder.create<seq::CompRegOp>(loc, args.enable, args.clock,
                                                  args.reset, validRegResetVal,
                                                  validRegName);
    }

    rets.passthroughs = stageOp.getPassthroughs();
    return rets;
  }

  // A container carrying all-things stage output naming related.
  // To avoid overloading 'output's to much (i'm trying to keep that reserved
  // for "output" ports), this is named "egress".
  struct StageEgressNames {
    llvm::SmallVector<Attribute> regNames;
    llvm::SmallVector<Attribute> outNames;
    llvm::SmallVector<Attribute> inNames;
  };

  // Returns a set of names for the output values of a given stage (registers
  // and passthrough). If `withPipelinePrefix` is true, the names will be
  // prefixed with the pipeline name.
  void getStageEgressNames(size_t stageIndex, Operation *stageTerminator,
                           bool withPipelinePrefix,
                           StageEgressNames &egressNames) {
    StringAttr pipelineName;
    if (withPipelinePrefix)
      pipelineName = getPipelineBaseName();

    if (auto stageOp = dyn_cast<StageOp>(stageTerminator)) {
      // Registers...
      std::string assignedRegName, assignedOutName, assignedInName;
      for (size_t regi = 0; regi < stageOp.getRegisters().size(); ++regi) {
        if (auto regName = stageOp.getRegisterName(regi)) {
          assignedRegName = regName.str();
          assignedOutName = assignedRegName + "_out";
          assignedInName = assignedRegName + "_in";
        } else {
          assignedRegName =
              ("stage" + Twine(stageIndex) + "_reg" + Twine(regi)).str();
          assignedOutName = ("out" + Twine(regi)).str();
          assignedInName = ("in" + Twine(regi)).str();
        }

        if (pipelineName) {
          assignedRegName = pipelineName.str() + "_" + assignedRegName;
          assignedOutName = pipelineName.str() + "_" + assignedOutName;
          assignedInName = pipelineName.str() + "_" + assignedInName;
        }

        egressNames.regNames.push_back(builder.getStringAttr(assignedRegName));
        egressNames.outNames.push_back(builder.getStringAttr(assignedOutName));
        egressNames.inNames.push_back(builder.getStringAttr(assignedInName));
      }

      // Passthroughs
      for (size_t passi = 0; passi < stageOp.getPassthroughs().size();
           ++passi) {
        if (auto passName = stageOp.getPassthroughName(passi)) {
          assignedOutName = (passName.strref() + "_out").str();
          assignedInName = (passName.strref() + "_in").str();
        } else {
          assignedOutName = ("pass" + Twine(passi)).str();
          assignedInName = ("pass" + Twine(passi)).str();
        }

        if (pipelineName) {
          assignedOutName = pipelineName.str() + "_" + assignedOutName;
          assignedInName = pipelineName.str() + "_" + assignedInName;
        }

        egressNames.outNames.push_back(builder.getStringAttr(assignedOutName));
        egressNames.inNames.push_back(builder.getStringAttr(assignedInName));
      }
    } else {
      // For the return op, we just inherit the names of the top-level pipeline
      // as stage output names.
      llvm::copy(pipeline.getOutputNames().getAsRange<StringAttr>(),
                 std::back_inserter(egressNames.outNames));
    }
  }

  // Returns a string to be used as a prefix for all stage registers.
  virtual StringAttr getStageRegPrefix(size_t stageIdx) = 0;

protected:
  // Determine a reasonable name for the pipeline. This will affect naming
  // of things such as stage registers and outlined stage modules.
  StringAttr getPipelineBaseName() {
    if (auto nameAttr = pipeline.getNameAttr())
      return nameAttr;
    return StringAttr::get(pipeline.getContext(), "p" + Twine(pipelineID));
  }

  // Parent module clock.
  Value parentClk;
  // Parent module reset.
  Value parentRst;
  // ID of the current pipeline, used for naming.
  size_t pipelineID;
  // The current pipeline to be converted.
  ScheduledPipelineOp pipeline;

  // The module wherein the pipeline resides.
  hw::HWModuleOp parentModule;

  OpBuilder &builder;

  // If true, will use clock gating for registers instead of input muxing.
  bool clockGateRegs;

  // Name of this pipeline - used for naming stages and registers.
  // Implementation defined.
  StringAttr pipelineName;
};

class PipelineInlineLowering : public PipelineLowering {
public:
  using PipelineLowering::PipelineLowering;

  StringAttr getStageRegPrefix(size_t stageIdx) override {
    return builder.getStringAttr(pipelineName.strref() + "_stage" +
                                 Twine(stageIdx));
  }

  LogicalResult run() override {
    pipelineName = getPipelineBaseName();

    // Replace uses of the pipeline internal inputs with the pipeline inputs.
    for (auto [outer, inner] :
         llvm::zip(pipeline.getInputs(), pipeline.getInnerInputs()))
      inner.replaceAllUsesWith(outer);

    // Replace uses of the external inputs with the inner external inputs.
    for (auto [outer, inner] :
         llvm::zip(pipeline.getExtInputs(), pipeline.getInnerExtInputs()))
      inner.replaceAllUsesWith(outer);

    // All operations should go directly before the pipeline op, into the
    // parent module.
    builder.setInsertionPoint(pipeline);
    StageArgs args;
    args.data = pipeline.getInnerInputs();
    args.enable = pipeline.getGo();
    args.clock = pipeline.getClock();
    args.reset = pipeline.getReset();
    args.stall = pipeline.getStall();
    if (failed(lowerStage(pipeline.getEntryStage(), args, 0)))
      return failure();

    // Replace uses of clock, reset, and stall.
    pipeline.getInnerClock().replaceAllUsesWith(pipeline.getClock());
    pipeline.getInnerReset().replaceAllUsesWith(pipeline.getReset());
    if (auto stall = pipeline.getStall())
      pipeline.getInnerStall().replaceAllUsesWith(stall);

    pipeline.erase();
    return success();
  }

  /// NOLINTNEXTLINE(misc-no-recursion)
  FailureOr<StageReturns>
  lowerStage(Block *stage, StageArgs args, size_t stageIndex,
             llvm::ArrayRef<Attribute> /*inputNames*/ = {}) override {
    OpBuilder::InsertionGuard guard(builder);

    if (stage != pipeline.getEntryStage()) {
      // Replace the internal stage inputs with the provided arguments.
      for (auto [vInput, vArg] :
           llvm::zip(pipeline.getStageDataArgs(stage), args.data))
        vInput.replaceAllUsesWith(vArg);
    }

    // Replace the stage valid signal.
    pipeline.getStageEnableSignal(stage).replaceAllUsesWith(args.enable);

    // Determine stage egress info.
    auto nextStage = dyn_cast<StageOp>(stage->getTerminator());
    StageEgressNames egressNames;
    if (nextStage)
      getStageEgressNames(stageIndex, nextStage,
                          /*withPipelinePrefix=*/true, egressNames);

    // Move stage operations into the current module.
    builder.setInsertionPoint(pipeline);
    StageReturns stageRets =
        emitStageBody(stage, args, egressNames.regNames, stageIndex);

    if (nextStage) {
      // Lower the next stage.
      SmallVector<Value> nextStageArgs;
      llvm::append_range(nextStageArgs, stageRets.regs);
      llvm::append_range(nextStageArgs, stageRets.passthroughs);
      args.enable = stageRets.valid;
      args.data = nextStageArgs;
      return lowerStage(nextStage.getNextStage(), args, stageIndex + 1);
    }

    // Replace the pipeline results with the return op operands.
    auto returnOp = cast<pipeline::ReturnOp>(stage->getTerminator());
    llvm::SmallVector<Value> pipelineReturns;
    llvm::append_range(pipelineReturns, returnOp.getInputs());
    // The last stage valid signal is the 'done' output of the pipeline.
    pipelineReturns.push_back(args.enable);
    pipeline.replaceAllUsesWith(pipelineReturns);
    return stageRets;
  }
};

class PipelineOutlineLowering : public PipelineLowering {
public:
  using PipelineLowering::PipelineLowering;

  StringAttr getStageRegPrefix(size_t stageIdx) override {
    return builder.getStringAttr("stage" + std::to_string(stageIdx));
  }

  // Helper class to manage grabbing the various inputs for stage modules.
  struct PipelineStageMod {
    PipelineStageMod() = default;
    PipelineStageMod(PipelineOutlineLowering &parent, Block *stage,
                     hw::HWModuleOp mod, bool withStall,
                     bool isParentPipeline = false) {
      size_t nStageDataArgs = parent.pipeline.getStageDataArgs(stage).size();
      inputs = mod.getArguments().take_front(nStageDataArgs);
      if (isParentPipeline) {
        // The parent pipeline should always have all external inputs available.
        extInputs = mod.getArguments().slice(
            nStageDataArgs, parent.pipeline.getExtInputs().size());
      } else {
        llvm::SmallVector<Value> stageExtInputs;
        parent.getStageExtInputs(stage, stageExtInputs);
        extInputs =
            mod.getArguments().slice(nStageDataArgs, stageExtInputs.size());
      }

      auto portLookup = mod.getPortLookupInfo();
      if (withStall)
        stall = mod.getArgument(*portLookup.getInputPortIndex(kStallPortName));
      enable = mod.getArgument(*portLookup.getInputPortIndex(kEnablePortName));
      clock = mod.getArgument(*portLookup.getInputPortIndex(kClockPortName));
      reset = mod.getArgument(*portLookup.getInputPortIndex(kResetPortName));
    }

    ValueRange inputs;
    ValueRange extInputs;
    Value enable;
    Value stall;
    Value clock;
    Value reset;
  };

  // Iterate over the external inputs to the pipeline, and determine which
  // stages actually reference them. This will be used to generate the stage
  // module signatures.
  void gatherExtInputsToStages() {
    for (auto extIn : pipeline.getInnerExtInputs()) {
      for (auto *user : extIn.getUsers())
        stageExtInputs[user->getBlock()].insert(extIn);
    }
  }

  LogicalResult run() override {
    pipelineName = StringAttr::get(pipeline.getContext(),
                                   parentModule.getName() + "_" +
                                       getPipelineBaseName().strref());
    cloneConstantsToStages();

    // Map external inputs to names - we use this to generate nicer names for
    // the stage module arguments.
    if (!pipeline.getExtInputs().empty()) {
      for (auto [extIn, extName] :
           llvm::zip(pipeline.getInnerExtInputs(),
                     pipeline.getExtInputNames()->getAsRange<StringAttr>())) {
        extInputNames[extIn] = extName;
      }
    }

    // Build the top-level pipeline module.
    bool withStall = static_cast<bool>(pipeline.getStall());
    pipelineMod = buildPipelineLike(
        pipelineName.strref(), pipeline.getInputs().getTypes(),
        pipeline.getInnerExtInputs(), pipeline.getDataOutputs().getTypes(),
        withStall, pipeline.getInputNames().getValue(),
        pipeline.getOutputNames().getValue());
    auto portLookup = pipelineMod.getPortLookupInfo();
    pipelineClk = pipelineMod.getBody().front().getArgument(
        *portLookup.getInputPortIndex(kClockPortName));
    pipelineRst = pipelineMod.getBody().front().getArgument(
        *portLookup.getInputPortIndex(kResetPortName));

    if (withStall)
      pipelineStall = pipelineMod.getBody().front().getArgument(
          *portLookup.getInputPortIndex(kStallPortName));

    if (!pipeline.getExtInputs().empty()) {
      // Maintain a mapping between external inputs and their corresponding
      // block argument in the top-level pipeline.
      auto modInnerExtInputs =
          pipelineMod.getBody().front().getArguments().slice(
              pipeline.getExtInputs().getBeginOperandIndex(),
              pipeline.getExtInputs().size());
      for (auto [extIn, barg] :
           llvm::zip(pipeline.getInnerExtInputs(), modInnerExtInputs)) {
        toplevelExtInputs[extIn] = barg;
      }
    }

    // Instantiate the pipeline in the parent module.
    builder.setInsertionPoint(pipeline);
    llvm::SmallVector<Value, 4> pipelineOperands;
    llvm::append_range(pipelineOperands, pipeline.getInputs());
    llvm::append_range(pipelineOperands, pipeline.getExtInputs());
    pipelineOperands.push_back(pipeline.getGo());
    if (auto stall = pipeline.getStall())
      pipelineOperands.push_back(stall);
    llvm::append_range(pipelineOperands,
                       ValueRange{pipeline.getClock(), pipeline.getReset()});

    auto pipelineInst = builder.create<hw::InstanceOp>(
        pipeline.getLoc(), pipelineMod,
        builder.getStringAttr(pipelineMod.getName()), pipelineOperands);

    // Replace the top-level pipeline results with the pipeline instance
    // results.
    pipeline.replaceAllUsesWith(pipelineInst.getResults());

    // Determine the external inputs to each stage.
    gatherExtInputsToStages();

    // From now on, insertion point must point to the pipeline module body.
    // This ensures that pipeline stage instantiations and free-standing
    // operations are inserted into the pipeline module.
    builder.setInsertionPointToStart(pipelineMod.getBodyBlock());

    pipelineStageMod =
        PipelineStageMod(*this, pipeline.getEntryStage(), pipelineMod,
                         withStall, /*isParentPipeline*/ true);

    StageArgs args;
    args.data = pipelineStageMod.inputs;
    args.enable = pipelineStageMod.enable;
    args.clock = pipelineStageMod.clock;
    args.reset = pipelineStageMod.reset;
    args.stall = pipelineStageMod.stall;
    FailureOr<StageReturns> lowerRes = lowerStage(
        pipeline.getEntryStage(), args, 0, pipeline.getInputNames().getValue());
    if (failed(lowerRes))
      return failure();

    // Assign the output op of the top-level pipeline module.
    auto outputOp =
        cast<hw::OutputOp>(pipelineMod.getBodyBlock()->getTerminator());
    outputOp->setOperands(lowerRes->passthroughs);
    outputOp->insertOperands(lowerRes->passthroughs.size(), lowerRes->valid);
    pipeline.erase();

    // Mini-optimization: the lowerStage/emitBody logic is written to not do any
    // special-case logic for the final stage of the pipeline.
    // In many cases, there are no operations in the final stage, except for the
    // return op. In these cases, we'll just inline the exit stage into the
    // pipeline module, and erase the (empty) last stage module.
    inlineAndEraseIfEmpty(currentStageInst);

    return success();
  }

  /// NOLINTNEXTLINE(misc-no-recursion)
  FailureOr<StageReturns>
  lowerStage(Block *stage, StageArgs argsToStage, size_t stageIndex,
             llvm::ArrayRef<Attribute> inputNames = {}) override {
    hw::OutputOp stageOutputOp;
    ValueRange nextStageArgs;
    bool withStall = static_cast<bool>(argsToStage.stall);

    auto replaceValuesInStage = [&](Value src, Value dst) {
      src.replaceUsesWithIf(dst, [&](OpOperand &operand) {
        return operand.getOwner()->getBlock() == stage;
      });
    };

    // Anything but the last stage
    StageEgressNames egressNames;
    auto [stageMod, stageInst] =
        buildStage(stage, argsToStage, stageIndex, inputNames, egressNames);
    auto thisStageMod = PipelineStageMod{*this, stage, stageMod, withStall};
    currentStageInst = stageInst;

    // Remap the internal inputs of the stage to the stage module block
    // arguments.
    for (auto [vInput, vBarg] :
         llvm::zip(pipeline.getStageDataArgs(stage), thisStageMod.inputs))
      vInput.replaceAllUsesWith(vBarg);

    // Remap external inputs to the stage to the external inputs in the
    // module block arguments.
    llvm::SmallVector<Value> stageExtInputs;
    getStageExtInputs(stage, stageExtInputs);

    for (auto [vExtInput, vExtBarg] :
         llvm::zip(stageExtInputs, thisStageMod.extInputs))
      replaceValuesInStage(vExtInput, vExtBarg);

    // Move stage operations into the module.
    builder.setInsertionPointToStart(&stageMod.getBody().front());
    stageOutputOp =
        cast<hw::OutputOp>(stageMod.getBody().front().getTerminator());

    // As well as any use of clock, reset and stall within this stage.
    replaceValuesInStage(pipeline.getStageEnableSignal(stage),
                         thisStageMod.enable);
    replaceValuesInStage(pipeline.getInnerClock(), thisStageMod.clock);
    replaceValuesInStage(pipeline.getInnerReset(), thisStageMod.reset);
    if (pipeline.hasStall())
      replaceValuesInStage(pipeline.getInnerStall(), thisStageMod.stall);

    // Arguments passed to emitStageBody that map to the inner stage module
    // arguments.
    StageArgs innerArgs;
    innerArgs.data = thisStageMod.inputs;
    innerArgs.enable = thisStageMod.enable;
    innerArgs.clock = thisStageMod.clock;
    innerArgs.reset = thisStageMod.reset;
    innerArgs.stall = thisStageMod.stall;
    StageReturns stageRets =
        emitStageBody(stage, innerArgs, egressNames.regNames, stageIndex);

    // Assign the output operation to the stage return values.
    stageOutputOp->insertOperands(0, stageRets.regs);
    stageOutputOp->insertOperands(stageOutputOp.getNumOperands(),
                                  stageRets.passthroughs);
    stageOutputOp->insertOperands(stageOutputOp.getNumOperands(),
                                  stageRets.valid);

    if (auto stageOp = dyn_cast<StageOp>(stage->getTerminator())) {
      // Lower the next stage.
      StageArgs nextStageArgs;
      nextStageArgs.data = stageInst.getResults().drop_back();
      nextStageArgs.enable = stageInst.getResults().back();
      nextStageArgs.clock = pipelineClk;
      nextStageArgs.reset = pipelineRst;
      nextStageArgs.stall = pipelineStall;
      return lowerStage(stageOp.getNextStage(), nextStageArgs, stageIndex + 1,
                        // It is the current stages' egress info which
                        // determines the next stage input names.
                        egressNames.inNames);
    }

    // This was the final stage - forward the return values of the last stage
    // instance as the stage return value.
    stageRets.passthroughs = stageInst.getResults().drop_back();
    stageRets.valid = stageInst.getResults().back();

    return stageRets;
  }

private:
  // Creates a clone of constant-like operations within each stage that
  // references them. This ensures that, once outlined, each stage will
  // reference valid constants.
  void cloneConstantsToStages() {
    // Maintain a mapping of the constants already cloned to a stage.
    for (auto &constantOp : llvm::make_early_inc_range(pipeline.getOps())) {
      if (!constantOp.hasTrait<OpTrait::ConstantLike>())
        continue;

      llvm::DenseMap<Block *, llvm::SmallVector<OpOperand *>> stageUses;
      Block *stageWithConstant = constantOp.getBlock();
      for (auto &use : constantOp.getUses()) {
        Block *usingStage = use.getOwner()->getBlock();
        if (usingStage == stageWithConstant)
          continue;
        stageUses[usingStage].push_back(&use);
      }

      // Clone the constant into each stage that uses it, and replace usages
      // within that stage.
      for (auto &[stage, uses] : stageUses) {
        Operation *clonedConstant = constantOp.clone();
        builder.setInsertionPointToStart(stage);
        builder.insert(clonedConstant);

        clonedConstant->setLoc(constantOp.getLoc());
        clonedConstant->moveBefore(&stage->front());
        for (OpOperand *use : uses)
          use->set(clonedConstant->getResult(0));
      }
    }
  }

  // Builds a pipeline-like module (top level pipeline and pipeline stages).
  // The module signature is:
  // ($ins, $extIns, enable, (stall)? : i1, clk : i1, reset : i1)
  //    -> ($outs, valid : i1)
  // Optionally, explicit names can be provided for the inputs, outputs, and
  // external inputs.
  hw::HWModuleOp buildPipelineLike(Twine name, TypeRange inputs,
                                   ValueRange extInputs, TypeRange outputs,
                                   bool withStall,
                                   llvm::ArrayRef<Attribute> inputNames,
                                   llvm::ArrayRef<Attribute> outputNames) {
    assert(inputs.size() == inputNames.size());
    assert(outputs.size() == outputNames.size());
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(parentModule);
    llvm::SmallVector<hw::PortInfo> ports;

    // Data inputs
    for (auto [idx, in] : llvm::enumerate(inputs))
      ports.push_back(hw::PortInfo{{inputNames[idx].cast<StringAttr>(), in,
                                    hw::ModulePort::Direction::Input}});

    // External inputs
    for (auto extIn : extInputs) {
      ports.push_back(hw::PortInfo{{extInputNames.at(extIn), extIn.getType(),
                                    hw::ModulePort::Direction::Input}});
    }

    // Enable input
    ports.push_back(
        hw::PortInfo{{builder.getStringAttr(kEnablePortName),
                      builder.getI1Type(), hw::ModulePort::Direction::Input}});

    if (withStall) {
      // Stall input
      ports.push_back(hw::PortInfo{{builder.getStringAttr(kStallPortName),
                                    builder.getI1Type(),
                                    hw::ModulePort::Direction::Input}});
    }

    // clock and reset
    ports.push_back(hw::PortInfo{{
        builder.getStringAttr(kClockPortName),
        builder.getI1Type(),
        hw::ModulePort::Direction::Input,
    }});
    ports.push_back(hw::PortInfo{{
        builder.getStringAttr(kResetPortName),
        builder.getI1Type(),
        hw::ModulePort::Direction::Input,
    }});

    for (auto [idx, out] : llvm::enumerate(outputs))
      ports.push_back(hw::PortInfo{{outputNames[idx].cast<StringAttr>(), out,
                                    hw::ModulePort::Direction::Output}});

    // Valid output
    ports.push_back(
        hw::PortInfo{{builder.getStringAttr(kValidPortName),
                      builder.getI1Type(), hw::ModulePort::Direction::Output}});

    return builder.create<hw::HWModuleOp>(pipeline.getLoc(),
                                          builder.getStringAttr(name), ports);
  }

  std::tuple<hw::HWModuleOp, hw::InstanceOp>
  buildStage(Block *stage, StageArgs args, size_t stageIndex,
             llvm::ArrayRef<Attribute> inputNames,
             StageEgressNames &egressNames) {
    assert(args.data.size() == inputNames.size());
    builder.setInsertionPoint(parentModule);
    llvm::SmallVector<Type> outputTypes;
    auto *terminator = stage->getTerminator();
    if (auto stageOp = dyn_cast<StageOp>(terminator)) {
      // The return values of a stage are the inputs to the next stage.
      llvm::append_range(outputTypes, ValueRange(pipeline.getStageDataArgs(
                                                     stageOp.getNextStage()))
                                          .getTypes());
    } else {
      // The return values of the last stage are the outputs of the pipeline.
      auto returnOp = cast<ReturnOp>(terminator);
      llvm::append_range(outputTypes, returnOp.getOperandTypes());
    }
    getStageEgressNames(stageIndex, terminator,
                        /*withPipelinePrefix=*/false, egressNames);

    llvm::SmallVector<Value> stageExtInputs;
    getStageExtInputs(stage, stageExtInputs);
    bool hasStall = static_cast<bool>(args.stall);
    hw::HWModuleOp mod = buildPipelineLike(
        pipelineName.strref() + "_s" + Twine(stageIndex),
        ValueRange(pipeline.getStageDataArgs(stage)).getTypes(), stageExtInputs,
        outputTypes, hasStall, inputNames, egressNames.outNames);

    // instantiate...
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(pipelineMod.getBodyBlock()->getTerminator());
    llvm::SmallVector<Value, 4> stageOperands;
    llvm::append_range(stageOperands, args.data);

    // Gather external inputs for this stage from the top-level pipeline
    // module.
    for (auto extInput : stageExtInputs)
      stageOperands.push_back(toplevelExtInputs[extInput]);

    stageOperands.push_back(args.enable);
    if (hasStall)
      stageOperands.push_back(pipelineStall);
    stageOperands.push_back(pipelineClk);
    stageOperands.push_back(pipelineRst);
    auto inst = builder.create<hw::InstanceOp>(pipeline.getLoc(), mod,
                                               mod.getName(), stageOperands);

    return {mod, inst};
  }

  // Pipeline module clock.
  Value pipelineClk;
  // Pipeline module reset.
  Value pipelineRst;
  // Pipeline module stall.
  Value pipelineStall;

  // Handle to the instantiation of the current stage under processing.
  hw::InstanceOp currentStageInst;

  // Pipeline module, containing stage instantiations.
  hw::HWModuleOp pipelineMod;

  // Handle to the instantiation of the last stage in the pipeline.
  hw::InstanceOp lastStageInst;

  // Handle to the PipelineStageMod for the parent pipeline module.
  PipelineStageMod pipelineStageMod;

  // A mapping between stages and the external inputs which they reference.
  // A SetVector is used to ensure determinism in the order of the external
  // inputs to a stage.
  llvm::DenseMap<Block *, llvm::SetVector<Value>> stageExtInputs;

  // A mapping between external inputs and their corresponding name attribute.
  DenseMap<Value, StringAttr> extInputNames;

  // A mapping between external inputs and their corresponding top-level
  // input in the pipeline module.
  llvm::DenseMap<Value, Value> toplevelExtInputs;

  // Wrapper around stageExtInputs which returns a llvm::SmallVector<Value>,
  // which in turn can be used to get a ValueRange (and by extension,
  // TypeRange).
  void getStageExtInputs(Block *stage, llvm::SmallVector<Value> &extInputs) {
    llvm::append_range(extInputs, stageExtInputs[stage]);
  }
};

//===----------------------------------------------------------------------===//
// Pipeline to HW Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct PipelineToHWPass : public PipelineToHWBase<PipelineToHWPass> {
  void runOnOperation() override;
};

void PipelineToHWPass::runOnOperation() {
  OpBuilder builder(&getContext());
  // Iterate over each pipeline op in the module and convert.
  // Note: This pass matches on `hw::ModuleOp`s and not directly on the
  // `ScheduledPipelineOp` due to the `ScheduledPipelineOp` being erased during
  // this pass.
  size_t pipelinesSeen = 0;
  for (auto pipeline : llvm::make_early_inc_range(
           getOperation().getOps<ScheduledPipelineOp>())) {
    if (outlineStages) {
      if (failed(PipelineOutlineLowering(pipelinesSeen, pipeline, builder,
                                         clockGateRegs)
                     .run())) {
        signalPassFailure();
        return;
      }
    } else if (failed(PipelineInlineLowering(pipelinesSeen, pipeline, builder,
                                             clockGateRegs)
                          .run())) {
      signalPassFailure();
      return;
    }
    ++pipelinesSeen;
  }
}

} // namespace

std::unique_ptr<mlir::Pass> circt::createPipelineToHWPass() {
  return std::make_unique<PipelineToHWPass>();
}
