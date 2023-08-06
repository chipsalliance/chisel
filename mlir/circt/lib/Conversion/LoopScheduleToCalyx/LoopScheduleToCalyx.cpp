//=== LoopScheduleToCalyx.cpp - LoopSchedule to Calyx pass entry point-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main LoopSchedule to Calyx conversion pass implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/LoopScheduleToCalyx.h"
#include "../PassDetail.h"
#include "circt/Dialect/Calyx/CalyxHelpers.h"
#include "circt/Dialect/Calyx/CalyxLoweringUtils.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include <variant>

using namespace llvm;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::cf;
using namespace mlir::func;
using namespace circt::loopschedule;

namespace circt {
namespace pipelinetocalyx {

//===----------------------------------------------------------------------===//
// Utility types
//===----------------------------------------------------------------------===//

class PipelineWhileOp : public calyx::WhileOpInterface<LoopSchedulePipelineOp> {
public:
  explicit PipelineWhileOp(LoopSchedulePipelineOp op)
      : calyx::WhileOpInterface<LoopSchedulePipelineOp>(op) {}

  Block::BlockArgListType getBodyArgs() override {
    return getOperation().getStagesBlock().getArguments();
  }

  Block *getBodyBlock() override { return &getOperation().getStagesBlock(); }

  Block *getConditionBlock() override { return &getOperation().getCondBlock(); }

  Value getConditionValue() override {
    return getOperation().getCondBlock().getTerminator()->getOperand(0);
  }

  std::optional<int64_t> getBound() override {
    return getOperation().getTripCount();
  }
};

//===----------------------------------------------------------------------===//
// Lowering state classes
//===----------------------------------------------------------------------===//

struct PipelineScheduleable {
  /// While operation to schedule.
  PipelineWhileOp whileOp;
  /// The group(s) to schedule before the while operation These groups should
  /// set the initial value(s) of the loop init_args register(s).
  SmallVector<calyx::GroupOp> initGroups;
};

/// A variant of types representing scheduleable operations.
using Scheduleable = std::variant<calyx::GroupOp, PipelineScheduleable>;

/// Holds additional information required for scheduling Pipeline pipelines.
class PipelineScheduler : public calyx::SchedulerInterface<Scheduleable> {
public:
  /// Registers operations that may be used in a pipeline, but does not produce
  /// a value to be used in a further stage.
  void registerNonPipelineOperations(Operation *op,
                                     calyx::GroupInterface group) {
    operationToGroup[op] = group;
  }

  /// Returns the group registered for this non-pipelined value, and None
  /// otherwise.
  template <typename TGroupOp = calyx::GroupInterface>
  std::optional<TGroupOp> getNonPipelinedGroupFrom(Operation *op) {
    auto it = operationToGroup.find(op);
    if (it == operationToGroup.end())
      return std::nullopt;

    if constexpr (std::is_same<TGroupOp, calyx::GroupInterface>::value)
      return it->second;
    else {
      auto group = dyn_cast<TGroupOp>(it->second.getOperation());
      assert(group && "Actual group type differed from expected group type");
      return group;
    }
  }
  /// Register reg as being the idx'th pipeline register for the stage.
  void addPipelineReg(Operation *stage, calyx::RegisterOp reg, unsigned idx) {
    assert(pipelineRegs[stage].count(idx) == 0);
    assert(idx < stage->getNumResults());
    pipelineRegs[stage][idx] = reg;
  }

  /// Return a mapping of stage result indices to pipeline registers.
  const DenseMap<unsigned, calyx::RegisterOp> &
  getPipelineRegs(Operation *stage) {
    return pipelineRegs[stage];
  }

  /// Add a stage's groups to the pipeline prologue.
  void addPipelinePrologue(Operation *op, SmallVector<StringAttr> groupNames) {
    pipelinePrologue[op].push_back(groupNames);
  }

  /// Add a stage's groups to the pipeline epilogue.
  void addPipelineEpilogue(Operation *op, SmallVector<StringAttr> groupNames) {
    pipelineEpilogue[op].push_back(groupNames);
  }

  /// Get the pipeline prologue.
  SmallVector<SmallVector<StringAttr>> getPipelinePrologue(Operation *op) {
    return pipelinePrologue[op];
  }

  /// Create the pipeline prologue.
  void createPipelinePrologue(Operation *op, PatternRewriter &rewriter) {
    auto stages = pipelinePrologue[op];
    for (size_t i = 0, e = stages.size(); i < e; ++i) {
      PatternRewriter::InsertionGuard g(rewriter);
      auto parOp = rewriter.create<calyx::ParOp>(op->getLoc());
      rewriter.setInsertionPointToStart(parOp.getBodyBlock());
      for (size_t j = 0; j < i + 1; ++j)
        for (auto group : stages[j])
          rewriter.create<calyx::EnableOp>(op->getLoc(), group);
    }
  }

  /// Create the pipeline epilogue.
  void createPipelineEpilogue(Operation *op, PatternRewriter &rewriter) {
    auto stages = pipelineEpilogue[op];
    for (size_t i = 0, e = stages.size(); i < e; ++i) {
      PatternRewriter::InsertionGuard g(rewriter);
      auto parOp = rewriter.create<calyx::ParOp>(op->getLoc());
      rewriter.setInsertionPointToStart(parOp.getBodyBlock());
      for (size_t j = i, f = stages.size(); j < f; ++j)
        for (auto group : stages[j])
          rewriter.create<calyx::EnableOp>(op->getLoc(), group);
    }
  }

private:
  /// A mapping between operations and the group to which it was assigned. This
  /// is used for specific corner cases, such as pipeline stages that may not
  /// actually pipeline any values.
  DenseMap<Operation *, calyx::GroupInterface> operationToGroup;

  /// A mapping from pipeline stages to their registers.
  DenseMap<Operation *, DenseMap<unsigned, calyx::RegisterOp>> pipelineRegs;

  /// A mapping from pipeline ops to a vector of vectors of group names that
  /// constitute the pipeline prologue. Each inner vector consists of the groups
  /// for one stage.
  DenseMap<Operation *, SmallVector<SmallVector<StringAttr>>> pipelinePrologue;

  /// A mapping from pipeline ops to a vector of vectors of group names that
  /// constitute the pipeline epilogue. Each inner vector consists of the groups
  /// for one stage.
  DenseMap<Operation *, SmallVector<SmallVector<StringAttr>>> pipelineEpilogue;
};

/// Handles the current state of lowering of a Calyx component. It is mainly
/// used as a key/value store for recording information during partial lowering,
/// which is required at later lowering passes.
class ComponentLoweringState
    : public calyx::ComponentLoweringStateInterface,
      public calyx::LoopLoweringStateInterface<PipelineWhileOp>,
      public PipelineScheduler {
public:
  ComponentLoweringState(calyx::ComponentOp component)
      : calyx::ComponentLoweringStateInterface(component) {}
};

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

/// Iterate through the operations of a source function and instantiate
/// components or primitives based on the type of the operations.
class BuildOpGroups : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    /// We walk the operations of the funcOp to ensure that all def's have
    /// been visited before their uses.
    bool opBuiltSuccessfully = true;
    funcOp.walk([&](Operation *_op) {
      opBuiltSuccessfully &=
          TypeSwitch<mlir::Operation *, bool>(_op)
              .template Case<arith::ConstantOp, ReturnOp, BranchOpInterface,
                             /// memref
                             memref::AllocOp, memref::AllocaOp, memref::LoadOp,
                             memref::StoreOp,
                             /// standard arithmetic
                             AddIOp, SubIOp, CmpIOp, ShLIOp, ShRUIOp, ShRSIOp,
                             AndIOp, XOrIOp, OrIOp, ExtUIOp, TruncIOp, MulIOp,
                             DivUIOp, RemUIOp, IndexCastOp,
                             /// static logic
                             LoopScheduleTerminatorOp>(
                  [&](auto op) { return buildOp(rewriter, op).succeeded(); })
              .template Case<FuncOp, LoopSchedulePipelineOp,
                             LoopScheduleRegisterOp,
                             LoopSchedulePipelineStageOp>([&](auto) {
                /// Skip: these special cases will be handled separately.
                return true;
              })
              .Default([&](auto op) {
                op->emitError() << "Unhandled operation during BuildOpGroups()";
                return false;
              });

      return opBuiltSuccessfully ? WalkResult::advance()
                                 : WalkResult::interrupt();
    });

    return success(opBuiltSuccessfully);
  }

private:
  /// Op builder specializations.
  LogicalResult buildOp(PatternRewriter &rewriter,
                        BranchOpInterface brOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        arith::ConstantOp constOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter, AddIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, SubIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, MulIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, DivUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, RemUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ShRUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ShRSIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ShLIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, AndIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, OrIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, XOrIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, CmpIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, TruncIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ExtUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ReturnOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, IndexCastOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, memref::AllocOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, memref::AllocaOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, memref::LoadOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, memref::StoreOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        LoopScheduleTerminatorOp op) const;

  /// buildLibraryOp will build a TCalyxLibOp inside a TGroupOp based on the
  /// source operation TSrcOp.
  template <typename TGroupOp, typename TCalyxLibOp, typename TSrcOp>
  LogicalResult buildLibraryOp(PatternRewriter &rewriter, TSrcOp op,
                               TypeRange srcTypes, TypeRange dstTypes) const {
    SmallVector<Type> types;
    llvm::append_range(types, srcTypes);
    llvm::append_range(types, dstTypes);

    auto calyxOp =
        getState<ComponentLoweringState>().getNewLibraryOpInstance<TCalyxLibOp>(
            rewriter, op.getLoc(), types);

    auto directions = calyxOp.portDirections();
    SmallVector<Value, 4> opInputPorts;
    SmallVector<Value, 4> opOutputPorts;
    for (auto dir : enumerate(directions)) {
      if (dir.value() == calyx::Direction::Input)
        opInputPorts.push_back(calyxOp.getResult(dir.index()));
      else
        opOutputPorts.push_back(calyxOp.getResult(dir.index()));
    }
    assert(
        opInputPorts.size() == op->getNumOperands() &&
        opOutputPorts.size() == op->getNumResults() &&
        "Expected an equal number of in/out ports in the Calyx library op with "
        "respect to the number of operands/results of the source operation.");

    /// Create assignments to the inputs of the library op.
    auto group = createGroupForOp<TGroupOp>(rewriter, op);
    rewriter.setInsertionPointToEnd(group.getBodyBlock());
    for (auto dstOp : enumerate(opInputPorts))
      rewriter.create<calyx::AssignOp>(op.getLoc(), dstOp.value(),
                                       op->getOperand(dstOp.index()));

    /// Replace the result values of the source operator with the new operator.
    for (auto res : enumerate(opOutputPorts)) {
      getState<ComponentLoweringState>().registerEvaluatingGroup(res.value(),
                                                                 group);
      op->getResult(res.index()).replaceAllUsesWith(res.value());
    }
    return success();
  }

  /// buildLibraryOp which provides in- and output types based on the operands
  /// and results of the op argument.
  template <typename TGroupOp, typename TCalyxLibOp, typename TSrcOp>
  LogicalResult buildLibraryOp(PatternRewriter &rewriter, TSrcOp op) const {
    return buildLibraryOp<TGroupOp, TCalyxLibOp, TSrcOp>(
        rewriter, op, op.getOperandTypes(), op->getResultTypes());
  }

  /// Creates a group named by the basic block which the input op resides in.
  template <typename TGroupOp>
  TGroupOp createGroupForOp(PatternRewriter &rewriter, Operation *op) const {
    Block *block = op->getBlock();
    auto groupName = getState<ComponentLoweringState>().getUniqueName(
        loweringState().blockName(block));
    return calyx::createGroup<TGroupOp>(
        rewriter, getState<ComponentLoweringState>().getComponentOp(),
        op->getLoc(), groupName);
  }

  /// buildLibraryBinaryPipeOp will build a TCalyxLibBinaryPipeOp, to
  /// deal with MulIOp, DivUIOp and RemUIOp.
  template <typename TOpType, typename TSrcOp>
  LogicalResult buildLibraryBinaryPipeOp(PatternRewriter &rewriter, TSrcOp op,
                                         TOpType opPipe, Value out) const {
    StringRef opName = TSrcOp::getOperationName().split(".").second;
    Location loc = op.getLoc();
    Type width = op.getResult().getType();
    // Pass the result from the Operation to the Calyx primitive.
    op.getResult().replaceAllUsesWith(out);
    auto reg = createRegister(
        op.getLoc(), rewriter, getComponent(), width.getIntOrFloatBitWidth(),
        getState<ComponentLoweringState>().getUniqueName(opName));
    // Operation pipelines are not combinational, so a GroupOp is required.
    auto group = createGroupForOp<calyx::GroupOp>(rewriter, op);
    getState<ComponentLoweringState>().addBlockScheduleable(op->getBlock(),
                                                            group);

    rewriter.setInsertionPointToEnd(group.getBodyBlock());
    rewriter.create<calyx::AssignOp>(loc, opPipe.getLeft(), op.getLhs());
    rewriter.create<calyx::AssignOp>(loc, opPipe.getRight(), op.getRhs());
    // Write the output to this register.
    rewriter.create<calyx::AssignOp>(loc, reg.getIn(), out);
    // The write enable port is high when the pipeline is done.
    rewriter.create<calyx::AssignOp>(loc, reg.getWriteEn(), opPipe.getDone());
    rewriter.create<calyx::AssignOp>(
        loc, opPipe.getGo(),
        createConstant(loc, rewriter, getComponent(), 1, 1));
    // The group is done when the register write is complete.
    rewriter.create<calyx::GroupDoneOp>(loc, reg.getDone());

    // Register the values for the pipeline.
    getState<ComponentLoweringState>().registerEvaluatingGroup(out, group);
    getState<ComponentLoweringState>().registerEvaluatingGroup(opPipe.getLeft(),
                                                               group);
    getState<ComponentLoweringState>().registerEvaluatingGroup(
        opPipe.getRight(), group);

    return success();
  }

  /// Creates assignments within the provided group to the address ports of the
  /// memoryOp based on the provided addressValues.
  void assignAddressPorts(PatternRewriter &rewriter, Location loc,
                          calyx::GroupInterface group,
                          calyx::MemoryInterface memoryInterface,
                          Operation::operand_range addressValues) const {
    IRRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(group.getBody());
    auto addrPorts = memoryInterface.addrPorts();
    if (addressValues.empty()) {
      assert(
          addrPorts.size() == 1 &&
          "We expected a 1 dimensional memory of size 1 because there were no "
          "address assignment values");
      // Assign 1'd0 to the address port.
      rewriter.create<calyx::AssignOp>(
          loc, addrPorts[0],
          createConstant(loc, rewriter, getComponent(), 1, 0));
    } else {
      assert(addrPorts.size() == addressValues.size() &&
             "Mismatch between number of address ports of the provided memory "
             "and address assignment values");
      for (auto address : enumerate(addressValues))
        rewriter.create<calyx::AssignOp>(loc, addrPorts[address.index()],
                                         address.value());
    }
  }
};

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::LoadOp loadOp) const {
  Value memref = loadOp.getMemref();
  auto memoryInterface =
      getState<ComponentLoweringState>().getMemoryInterface(memref);
  if (calyx::noStoresToMemory(memref) && calyx::singleLoadFromMemory(memref)) {
    // Single load from memory; we do not need to write the
    // output to a register. This is essentially a "combinational read" under
    // current Calyx semantics with memory, and thus can be done in a
    // combinational group. Note that if any stores are done to this memory,
    // we require that the load and store be in separate non-combinational
    // groups to avoid reading and writing to the same memory in the same group.
    auto combGroup = createGroupForOp<calyx::CombGroupOp>(rewriter, loadOp);
    assignAddressPorts(rewriter, loadOp.getLoc(), combGroup, memoryInterface,
                       loadOp.getIndices());

    // We refrain from replacing the loadOp result with
    // memoryInterface.readData, since multiple loadOp's need to be converted
    // to a single memory's ReadData. If this replacement is done now, we lose
    // the link between which SSA memref::LoadOp values map to which groups for
    // loading a value from the Calyx memory. At this point of lowering, we
    // keep the memref::LoadOp SSA value, and do value replacement _after_
    // control has been generated (see LateSSAReplacement). This is *vital* for
    // things such as InlineCombGroups to be able to properly track which
    // memory assignment groups belong to which accesses.
    getState<ComponentLoweringState>().registerEvaluatingGroup(
        loadOp.getResult(), combGroup);
  } else {
    auto group = createGroupForOp<calyx::GroupOp>(rewriter, loadOp);
    assignAddressPorts(rewriter, loadOp.getLoc(), group, memoryInterface,
                       loadOp.getIndices());

    // Multiple loads from the same memory; In this case, we _may_ have a
    // structural hazard in the design we generate. To get around this, we
    // conservatively place a register in front of each load operation, and
    // replace all uses of the loaded value with the register output. Proper
    // handling of this requires the combinational group inliner/scheduler to
    // be aware of when a combinational expression references multiple loaded
    // values from the same memory, and then schedule assignments to temporary
    // registers to get around the structural hazard.
    auto reg = createRegister(
        loadOp.getLoc(), rewriter, getComponent(),
        loadOp.getMemRefType().getElementTypeBitWidth(),
        getState<ComponentLoweringState>().getUniqueName("load"));
    calyx::buildAssignmentsForRegisterWrite(
        rewriter, group, getState<ComponentLoweringState>().getComponentOp(),
        reg, memoryInterface.readData());
    loadOp.getResult().replaceAllUsesWith(reg.getOut());
    getState<ComponentLoweringState>().addBlockScheduleable(loadOp->getBlock(),
                                                            group);
  }
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::StoreOp storeOp) const {
  auto memoryInterface = getState<ComponentLoweringState>().getMemoryInterface(
      storeOp.getMemref());
  auto group = createGroupForOp<calyx::GroupOp>(rewriter, storeOp);

  // This is a sequential group, so register it as being scheduleable for the
  // block.
  getState<ComponentLoweringState>().addBlockScheduleable(storeOp->getBlock(),
                                                          group);
  assignAddressPorts(rewriter, storeOp.getLoc(), group, memoryInterface,
                     storeOp.getIndices());
  rewriter.setInsertionPointToEnd(group.getBodyBlock());
  rewriter.create<calyx::AssignOp>(
      storeOp.getLoc(), memoryInterface.writeData(), storeOp.getValueToStore());
  rewriter.create<calyx::AssignOp>(
      storeOp.getLoc(), memoryInterface.writeEn(),
      createConstant(storeOp.getLoc(), rewriter, getComponent(), 1, 1));
  rewriter.create<calyx::GroupDoneOp>(storeOp.getLoc(),
                                      memoryInterface.writeDone());

  getState<ComponentLoweringState>().registerNonPipelineOperations(storeOp,
                                                                   group);

  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     MulIOp mul) const {
  Location loc = mul.getLoc();
  Type width = mul.getResult().getType(), one = rewriter.getI1Type();
  auto mulPipe =
      getState<ComponentLoweringState>()
          .getNewLibraryOpInstance<calyx::MultPipeLibOp>(
              rewriter, loc, {one, one, one, width, width, width, one});
  return buildLibraryBinaryPipeOp<calyx::MultPipeLibOp>(
      rewriter, mul, mulPipe,
      /*out=*/mulPipe.getOut());
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     DivUIOp div) const {
  Location loc = div.getLoc();
  Type width = div.getResult().getType(), one = rewriter.getI1Type();
  auto divPipe =
      getState<ComponentLoweringState>()
          .getNewLibraryOpInstance<calyx::DivUPipeLibOp>(
              rewriter, loc, {one, one, one, width, width, width, width, one});
  return buildLibraryBinaryPipeOp<calyx::DivUPipeLibOp>(
      rewriter, div, divPipe,
      /*out=*/divPipe.getOut());
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     RemUIOp rem) const {
  Location loc = rem.getLoc();
  Type width = rem.getResult().getType(), one = rewriter.getI1Type();
  auto remPipe =
      getState<ComponentLoweringState>()
          .getNewLibraryOpInstance<calyx::DivUPipeLibOp>(
              rewriter, loc, {one, one, one, width, width, width, width, one});
  return buildLibraryBinaryPipeOp<calyx::DivUPipeLibOp>(
      rewriter, rem, remPipe,
      /*out=*/remPipe.getOut());
}

template <typename TAllocOp>
static LogicalResult buildAllocOp(ComponentLoweringState &componentState,
                                  PatternRewriter &rewriter, TAllocOp allocOp) {
  rewriter.setInsertionPointToStart(
      componentState.getComponentOp().getBodyBlock());
  MemRefType memtype = allocOp.getType();
  SmallVector<int64_t> addrSizes;
  SmallVector<int64_t> sizes;
  for (int64_t dim : memtype.getShape()) {
    sizes.push_back(dim);
    addrSizes.push_back(calyx::handleZeroWidth(dim));
  }
  // If memref has no size (e.g., memref<i32>) create a 1 dimensional memory of
  // size 1.
  if (sizes.empty() && addrSizes.empty()) {
    sizes.push_back(1);
    addrSizes.push_back(1);
  }
  auto memoryOp = rewriter.create<calyx::MemoryOp>(
      allocOp.getLoc(), componentState.getUniqueName("mem"),
      memtype.getElementType().getIntOrFloatBitWidth(), sizes, addrSizes);
  // Externalize memories by default. This makes it easier for the native
  // compiler to provide initialized memories.
  memoryOp->setAttr("external",
                    IntegerAttr::get(rewriter.getI1Type(), llvm::APInt(1, 1)));
  componentState.registerMemoryInterface(allocOp.getResult(),
                                         calyx::MemoryInterface(memoryOp));
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::AllocOp allocOp) const {
  return buildAllocOp(getState<ComponentLoweringState>(), rewriter, allocOp);
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::AllocaOp allocOp) const {
  return buildAllocOp(getState<ComponentLoweringState>(), rewriter, allocOp);
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     LoopScheduleTerminatorOp term) const {
  if (term.getOperands().size() == 0)
    return success();

  // Replace the pipeline's result(s) with the terminator's results.
  auto *pipeline = term->getParentOp();
  for (size_t i = 0, e = pipeline->getNumResults(); i < e; ++i)
    pipeline->getResult(i).replaceAllUsesWith(term.getResults()[i]);

  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     BranchOpInterface brOp) const {
  /// Branch argument passing group creation
  /// Branch operands are passed through registers. In BuildBasicBlockRegs we
  /// created registers for all branch arguments of each block. We now
  /// create groups for assigning values to these registers.
  Block *srcBlock = brOp->getBlock();
  for (auto succBlock : enumerate(brOp->getSuccessors())) {
    auto succOperands = brOp.getSuccessorOperands(succBlock.index());
    if (succOperands.empty())
      continue;
    // Create operand passing group
    std::string groupName = loweringState().blockName(srcBlock) + "_to_" +
                            loweringState().blockName(succBlock.value());
    auto groupOp = calyx::createGroup<calyx::GroupOp>(rewriter, getComponent(),
                                                      brOp.getLoc(), groupName);
    // Fetch block argument registers associated with the basic block
    auto dstBlockArgRegs =
        getState<ComponentLoweringState>().getBlockArgRegs(succBlock.value());
    // Create register assignment for each block argument
    for (auto arg : enumerate(succOperands.getForwardedOperands())) {
      auto reg = dstBlockArgRegs[arg.index()];
      calyx::buildAssignmentsForRegisterWrite(
          rewriter, groupOp,
          getState<ComponentLoweringState>().getComponentOp(), reg,
          arg.value());
    }
    /// Register the group as a block argument group, to be executed
    /// when entering the successor block from this block (srcBlock).
    getState<ComponentLoweringState>().addBlockArgGroup(
        srcBlock, succBlock.value(), groupOp);
  }
  return success();
}

/// For each return statement, we create a new group for assigning to the
/// previously created return value registers.
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ReturnOp retOp) const {
  if (retOp.getNumOperands() == 0)
    return success();

  std::string groupName =
      getState<ComponentLoweringState>().getUniqueName("ret_assign");
  auto groupOp = calyx::createGroup<calyx::GroupOp>(rewriter, getComponent(),
                                                    retOp.getLoc(), groupName);
  for (auto op : enumerate(retOp.getOperands())) {
    auto reg = getState<ComponentLoweringState>().getReturnReg(op.index());
    calyx::buildAssignmentsForRegisterWrite(
        rewriter, groupOp, getState<ComponentLoweringState>().getComponentOp(),
        reg, op.value());
  }
  /// Schedule group for execution for when executing the return op block.
  getState<ComponentLoweringState>().addBlockScheduleable(retOp->getBlock(),
                                                          groupOp);
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     arith::ConstantOp constOp) const {
  /// Move constant operations to the compOp body as hw::ConstantOp's.
  APInt value;
  calyx::matchConstantOp(constOp, value);
  auto hwConstOp = rewriter.replaceOpWithNewOp<hw::ConstantOp>(constOp, value);
  hwConstOp->moveAfter(getComponent().getBodyBlock(),
                       getComponent().getBodyBlock()->begin());
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     AddIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::AddLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     SubIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::SubLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ShRUIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::RshLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ShRSIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::SrshLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ShLIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::LshLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     AndIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::AndLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     OrIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::OrLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     XOrIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::XorLibOp>(rewriter, op);
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     CmpIOp op) const {
  switch (op.getPredicate()) {
  case CmpIPredicate::eq:
    return buildLibraryOp<calyx::CombGroupOp, calyx::EqLibOp>(rewriter, op);
  case CmpIPredicate::ne:
    return buildLibraryOp<calyx::CombGroupOp, calyx::NeqLibOp>(rewriter, op);
  case CmpIPredicate::uge:
    return buildLibraryOp<calyx::CombGroupOp, calyx::GeLibOp>(rewriter, op);
  case CmpIPredicate::ult:
    return buildLibraryOp<calyx::CombGroupOp, calyx::LtLibOp>(rewriter, op);
  case CmpIPredicate::ugt:
    return buildLibraryOp<calyx::CombGroupOp, calyx::GtLibOp>(rewriter, op);
  case CmpIPredicate::ule:
    return buildLibraryOp<calyx::CombGroupOp, calyx::LeLibOp>(rewriter, op);
  case CmpIPredicate::sge:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SgeLibOp>(rewriter, op);
  case CmpIPredicate::slt:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SltLibOp>(rewriter, op);
  case CmpIPredicate::sgt:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SgtLibOp>(rewriter, op);
  case CmpIPredicate::sle:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SleLibOp>(rewriter, op);
  }
  llvm_unreachable("unsupported comparison predicate");
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     TruncIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::SliceLibOp>(
      rewriter, op, {op.getOperand().getType()}, {op.getType()});
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ExtUIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::PadLibOp>(
      rewriter, op, {op.getOperand().getType()}, {op.getType()});
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     IndexCastOp op) const {
  Type sourceType = calyx::convIndexType(rewriter, op.getOperand().getType());
  Type targetType = calyx::convIndexType(rewriter, op.getResult().getType());
  unsigned targetBits = targetType.getIntOrFloatBitWidth();
  unsigned sourceBits = sourceType.getIntOrFloatBitWidth();
  LogicalResult res = success();

  if (targetBits == sourceBits) {
    /// Drop the index cast and replace uses of the target value with the source
    /// value.
    op.getResult().replaceAllUsesWith(op.getOperand());
  } else {
    /// pad/slice the source operand.
    if (sourceBits > targetBits)
      res = buildLibraryOp<calyx::CombGroupOp, calyx::SliceLibOp>(
          rewriter, op, {sourceType}, {targetType});
    else
      res = buildLibraryOp<calyx::CombGroupOp, calyx::PadLibOp>(
          rewriter, op, {sourceType}, {targetType});
  }
  rewriter.eraseOp(op);
  return res;
}

/// Creates a new Calyx component for each FuncOp in the program.
struct FuncOpConversion : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    /// Maintain a mapping between funcOp input arguments and the port index
    /// which the argument will eventually map to.
    DenseMap<Value, unsigned> funcOpArgRewrites;

    /// Maintain a mapping between funcOp output indexes and the component
    /// output port index which the return value will eventually map to.
    DenseMap<unsigned, unsigned> funcOpResultMapping;

    /// Maintain a mapping between an external memory argument (identified by a
    /// memref) and eventual component input- and output port indices that will
    /// map to the memory ports. The pair denotes the start index of the memory
    /// ports in the in- and output ports of the component. Ports are expected
    /// to be ordered in the same manner as they are added by
    /// calyx::appendPortsForExternalMemref.
    DenseMap<Value, std::pair<unsigned, unsigned>> extMemoryCompPortIndices;

    /// Create I/O ports. Maintain separate in/out port vectors to determine
    /// which port index each function argument will eventually map to.
    SmallVector<calyx::PortInfo> inPorts, outPorts;
    FunctionType funcType = funcOp.getFunctionType();
    unsigned extMemCounter = 0;
    for (auto arg : enumerate(funcOp.getArguments())) {
      if (arg.value().getType().isa<MemRefType>()) {
        /// External memories
        auto memName =
            "ext_mem" + std::to_string(extMemoryCompPortIndices.size());
        extMemoryCompPortIndices[arg.value()] = {inPorts.size(),
                                                 outPorts.size()};
        calyx::appendPortsForExternalMemref(rewriter, memName, arg.value(),
                                            extMemCounter++, inPorts, outPorts);
      } else {
        /// Single-port arguments
        auto inName = "in" + std::to_string(arg.index());
        funcOpArgRewrites[arg.value()] = inPorts.size();
        inPorts.push_back(calyx::PortInfo{
            rewriter.getStringAttr(inName),
            calyx::convIndexType(rewriter, arg.value().getType()),
            calyx::Direction::Input,
            DictionaryAttr::get(rewriter.getContext(), {})});
      }
    }
    for (auto res : enumerate(funcType.getResults())) {
      funcOpResultMapping[res.index()] = outPorts.size();
      outPorts.push_back(calyx::PortInfo{
          rewriter.getStringAttr("out" + std::to_string(res.index())),
          calyx::convIndexType(rewriter, res.value()), calyx::Direction::Output,
          DictionaryAttr::get(rewriter.getContext(), {})});
    }

    /// We've now recorded all necessary indices. Merge in- and output ports
    /// and add the required mandatory component ports.
    auto ports = inPorts;
    llvm::append_range(ports, outPorts);
    calyx::addMandatoryComponentPorts(rewriter, ports);

    /// Create a calyx::ComponentOp corresponding to the to-be-lowered function.
    auto compOp = rewriter.create<calyx::ComponentOp>(
        funcOp.getLoc(), rewriter.getStringAttr(funcOp.getSymName()), ports);

    /// Mark this component as the toplevel.
    compOp->setAttr("toplevel", rewriter.getUnitAttr());

    /// Store the function-to-component mapping.
    functionMapping[funcOp] = compOp;
    auto *compState = loweringState().getState<ComponentLoweringState>(compOp);
    compState->setFuncOpResultMapping(funcOpResultMapping);

    /// Rewrite funcOp SSA argument values to the CompOp arguments.
    for (auto &mapping : funcOpArgRewrites)
      mapping.getFirst().replaceAllUsesWith(
          compOp.getArgument(mapping.getSecond()));

    /// Register external memories
    for (auto extMemPortIndices : extMemoryCompPortIndices) {
      /// Create a mapping for the in- and output ports using the Calyx memory
      /// port structure.
      calyx::MemoryPortsImpl extMemPorts;
      unsigned inPortsIt = extMemPortIndices.getSecond().first;
      unsigned outPortsIt = extMemPortIndices.getSecond().second +
                            compOp.getInputPortInfo().size();
      extMemPorts.readData = compOp.getArgument(inPortsIt++);
      extMemPorts.writeDone = compOp.getArgument(inPortsIt);
      extMemPorts.writeData = compOp.getArgument(outPortsIt++);
      unsigned nAddresses = extMemPortIndices.getFirst()
                                .getType()
                                .cast<MemRefType>()
                                .getShape()
                                .size();
      for (unsigned j = 0; j < nAddresses; ++j)
        extMemPorts.addrPorts.push_back(compOp.getArgument(outPortsIt++));
      extMemPorts.writeEn = compOp.getArgument(outPortsIt);

      /// Register the external memory ports as a memory interface within the
      /// component.
      compState->registerMemoryInterface(extMemPortIndices.getFirst(),
                                         calyx::MemoryInterface(extMemPorts));
    }

    return success();
  }
};

/// In BuildWhileGroups, a register is created for each iteration argumenet of
/// the while op. These registers are then written to on the while op
/// terminating yield operation alongside before executing the whileOp in the
/// schedule, to set the initial values of the argument registers.
class BuildWhileGroups : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    LogicalResult res = success();
    funcOp.walk([&](Operation *op) {
      if (!isa<LoopSchedulePipelineOp>(op))
        return WalkResult::advance();

      PipelineWhileOp whileOp(cast<LoopSchedulePipelineOp>(op));

      getState<ComponentLoweringState>().setUniqueName(whileOp.getOperation(),
                                                       "while");

      /// Create iteration argument registers.
      /// The iteration argument registers will be referenced:
      /// - In the "before" part of the while loop, calculating the conditional,
      /// - In the "after" part of the while loop,
      /// - Outside the while loop, rewriting the while loop return values.
      for (auto arg : enumerate(whileOp.getBodyArgs())) {
        std::string name = getState<ComponentLoweringState>()
                               .getUniqueName(whileOp.getOperation())
                               .str() +
                           "_arg" + std::to_string(arg.index());
        auto reg =
            createRegister(arg.value().getLoc(), rewriter, getComponent(),
                           arg.value().getType().getIntOrFloatBitWidth(), name);
        getState<ComponentLoweringState>().addLoopIterReg(whileOp, reg,
                                                          arg.index());
        arg.value().replaceAllUsesWith(reg.getOut());

        /// Also replace uses in the "before" region of the while loop
        whileOp.getConditionBlock()
            ->getArgument(arg.index())
            .replaceAllUsesWith(reg.getOut());
      }

      /// Create iter args initial value assignment group(s), one per register.
      SmallVector<calyx::GroupOp> initGroups;
      auto numOperands = whileOp.getOperation()->getNumOperands();
      for (size_t i = 0; i < numOperands; ++i) {
        auto initGroupOp =
            getState<ComponentLoweringState>().buildLoopIterArgAssignments(
                rewriter, whileOp,
                getState<ComponentLoweringState>().getComponentOp(),
                getState<ComponentLoweringState>().getUniqueName(
                    whileOp.getOperation()) +
                    "_init_" + std::to_string(i),
                whileOp.getOperation()->getOpOperand(i));
        initGroups.push_back(initGroupOp);
      }

      /// Add the while op to the list of scheduleable things in the current
      /// block.
      getState<ComponentLoweringState>().addBlockScheduleable(
          whileOp.getOperation()->getBlock(), PipelineScheduleable{
                                                  whileOp,
                                                  initGroups,
                                              });
      return WalkResult::advance();
    });
    return res;
  }
};

/// Builds registers for each pipeline stage in the program.
class BuildPipelineRegs : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    funcOp.walk([&](LoopScheduleRegisterOp op) {
      // Condition registers are handled in BuildWhileGroups.
      auto *parent = op->getParentOp();
      auto stage = dyn_cast<LoopSchedulePipelineStageOp>(parent);
      if (!stage)
        return;

      // Create a register for each stage.
      for (auto &operand : op->getOpOperands()) {
        unsigned i = operand.getOperandNumber();
        // Iter args are created in BuildWhileGroups, so just mark the iter arg
        // register as the appropriate pipeline register.
        Value stageResult = stage.getResult(i);
        bool isIterArg = false;
        for (auto &use : stageResult.getUses()) {
          if (auto term = dyn_cast<LoopScheduleTerminatorOp>(use.getOwner())) {
            if (use.getOperandNumber() < term.getIterArgs().size()) {
              PipelineWhileOp whileOp(
                  dyn_cast<LoopSchedulePipelineOp>(stage->getParentOp()));
              auto reg = getState<ComponentLoweringState>().getLoopIterReg(
                  whileOp, use.getOperandNumber());
              getState<ComponentLoweringState>().addPipelineReg(stage, reg, i);
              isIterArg = true;
            }
          }
        }
        if (isIterArg)
          continue;

        // Create a register for passing this result to later stages.
        Value value = operand.get();
        Type resultType = value.getType();
        assert(resultType.isa<IntegerType>() &&
               "unsupported pipeline result type");
        auto name = SmallString<20>("stage_");
        name += std::to_string(stage.getStageNumber());
        name += "_register_";
        name += std::to_string(i);
        unsigned width = resultType.getIntOrFloatBitWidth();
        auto reg = createRegister(value.getLoc(), rewriter, getComponent(),
                                  width, name);
        getState<ComponentLoweringState>().addPipelineReg(stage, reg, i);

        // Note that we do not use replace all uses with here as in
        // BuildBasicBlockRegs. Instead, we wait until after BuildOpGroups, and
        // replace all uses inside BuildPipelineGroups, once the pipeline
        // register created here has been assigned to.
      }
    });
    return success();
  }
};

/// Builds groups for assigning registers for pipeline stages.
class BuildPipelineGroups : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    for (auto pipeline : funcOp.getOps<LoopSchedulePipelineOp>())
      for (auto stage :
           pipeline.getStagesBlock().getOps<LoopSchedulePipelineStageOp>())
        if (failed(buildStageGroups(pipeline, stage, rewriter)))
          return failure();

    return success();
  }

  LogicalResult buildStageGroups(LoopSchedulePipelineOp whileOp,
                                 LoopSchedulePipelineStageOp stage,
                                 PatternRewriter &rewriter) const {
    // Collect pipeline registers for stage.
    auto pipelineRegisters =
        getState<ComponentLoweringState>().getPipelineRegs(stage);
    // Get the number of pipeline stages in the stages block, excluding the
    // terminator. The verifier guarantees there is at least one stage followed
    // by a terminator.
    size_t numStages = whileOp.getStagesBlock().getOperations().size() - 1;
    assert(numStages > 0);

    // Collect group names for the prologue or epilogue.
    SmallVector<StringAttr> prologueGroups, epilogueGroups;

    auto updatePrologueAndEpilogue = [&](calyx::GroupOp group) {
      // Mark the group for scheduling in the pipeline's block.
      getState<ComponentLoweringState>().addBlockScheduleable(stage->getBlock(),
                                                              group);

      // Add the group to the prologue or epilogue for this stage as
      // necessary. The goal is to fill the pipeline so it will be in steady
      // state after the prologue, and drain the pipeline from steady state in
      // the epilogue. Every stage but the last should have its groups in the
      // prologue, and every stage but the first should have its groups in the
      // epilogue.
      unsigned stageNumber = stage.getStageNumber();
      if (stageNumber < numStages - 1)
        prologueGroups.push_back(group.getSymNameAttr());
      if (stageNumber > 0)
        epilogueGroups.push_back(group.getSymNameAttr());
    };

    MutableArrayRef<OpOperand> operands =
        stage.getBodyBlock().getTerminator()->getOpOperands();
    bool isStageWithNoPipelinedValues =
        operands.empty() && !stage.getBodyBlock().empty();
    if (isStageWithNoPipelinedValues) {
      // Covers the case where there are no values that need to be passed
      // through to the next stage, e.g., some intermediary store.
      for (auto &op : stage.getBodyBlock())
        if (auto group = getState<ComponentLoweringState>()
                             .getNonPipelinedGroupFrom<calyx::GroupOp>(&op))
          updatePrologueAndEpilogue(*group);
    }

    for (auto &operand : operands) {
      unsigned i = operand.getOperandNumber();
      Value value = operand.get();

      // Get the pipeline register for that result.
      auto pipelineRegister = pipelineRegisters[i];

      // Get the evaluating group for that value.
      calyx::GroupInterface evaluatingGroup =
          getState<ComponentLoweringState>().getEvaluatingGroup(value);

      // Remember the final group for this stage result.
      calyx::GroupOp group;

      // Stitch the register in, depending on whether the group was
      // combinational or sequential.
      if (auto combGroup =
              dyn_cast<calyx::CombGroupOp>(evaluatingGroup.getOperation()))
        group =
            convertCombToSeqGroup(combGroup, pipelineRegister, value, rewriter);
      else
        group =
            replaceGroupRegister(evaluatingGroup, pipelineRegister, rewriter);

      // Replace the stage result uses with the register out.
      stage.getResult(i).replaceAllUsesWith(pipelineRegister.getOut());

      updatePrologueAndEpilogue(group);
    }

    // Append the stage to the prologue or epilogue list of stages if any groups
    // were added for this stage. We append a list of groups for each stage, so
    // we can group by stage later, when we generate the schedule.
    if (!prologueGroups.empty())
      getState<ComponentLoweringState>().addPipelinePrologue(whileOp,
                                                             prologueGroups);
    if (!epilogueGroups.empty())
      getState<ComponentLoweringState>().addPipelineEpilogue(whileOp,
                                                             epilogueGroups);

    return success();
  }

  calyx::GroupOp convertCombToSeqGroup(calyx::CombGroupOp combGroup,
                                       calyx::RegisterOp pipelineRegister,
                                       Value value,
                                       PatternRewriter &rewriter) const {
    // Create a sequential group and replace the comb group.
    PatternRewriter::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(combGroup);
    auto group = rewriter.create<calyx::GroupOp>(combGroup.getLoc(),
                                                 combGroup.getName());
    rewriter.cloneRegionBefore(combGroup.getBodyRegion(),
                               &group.getBody().front());
    group.getBodyRegion().back().erase();
    rewriter.eraseOp(combGroup);

    // Stitch evaluating group to register.
    calyx::buildAssignmentsForRegisterWrite(
        rewriter, group, getState<ComponentLoweringState>().getComponentOp(),
        pipelineRegister, value);

    // Mark the new group as the evaluating group.
    for (auto assign : group.getOps<calyx::AssignOp>())
      getState<ComponentLoweringState>().registerEvaluatingGroup(
          assign.getSrc(), group);

    return group;
  }

  calyx::GroupOp replaceGroupRegister(calyx::GroupInterface evaluatingGroup,
                                      calyx::RegisterOp pipelineRegister,
                                      PatternRewriter &rewriter) const {
    auto group = cast<calyx::GroupOp>(evaluatingGroup.getOperation());

    // Get the group and register that is temporarily being written to.
    auto doneOp = group.getDoneOp();
    auto tempReg =
        cast<calyx::RegisterOp>(doneOp.getSrc().cast<OpResult>().getOwner());
    auto tempIn = tempReg.getIn();
    auto tempWriteEn = tempReg.getWriteEn();

    // Replace the register write with a write to the pipeline register.
    for (auto assign : group.getOps<calyx::AssignOp>()) {
      if (assign.getDest() == tempIn)
        assign.getDestMutable().assign(pipelineRegister.getIn());
      else if (assign.getDest() == tempWriteEn)
        assign.getDestMutable().assign(pipelineRegister.getWriteEn());
    }
    doneOp.getSrcMutable().assign(pipelineRegister.getDone());

    // Remove the old register completely.
    rewriter.eraseOp(tempReg);

    return group;
  }
};

/// Builds a control schedule by traversing the CFG of the function and
/// associating this with the previously created groups.
/// For simplicity, the generated control flow is expanded for all possible
/// paths in the input DAG. This elaborated control flow is later reduced in
/// the runControlFlowSimplification passes.
class BuildControl : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    auto *entryBlock = &funcOp.getBlocks().front();
    rewriter.setInsertionPointToStart(
        getComponent().getControlOp().getBodyBlock());
    auto topLevelSeqOp = rewriter.create<calyx::SeqOp>(funcOp.getLoc());
    DenseSet<Block *> path;
    return buildCFGControl(path, rewriter, topLevelSeqOp.getBodyBlock(),
                           nullptr, entryBlock);
  }

private:
  /// Sequentially schedules the groups that registered themselves with
  /// 'block'.
  LogicalResult scheduleBasicBlock(PatternRewriter &rewriter,
                                   const DenseSet<Block *> &path,
                                   mlir::Block *parentCtrlBlock,
                                   mlir::Block *block) const {
    auto compBlockScheduleables =
        getState<ComponentLoweringState>().getBlockScheduleables(block);
    auto loc = block->front().getLoc();

    if (compBlockScheduleables.size() > 1) {
      auto seqOp = rewriter.create<calyx::SeqOp>(loc);
      parentCtrlBlock = seqOp.getBodyBlock();
    }

    for (auto &group : compBlockScheduleables) {
      rewriter.setInsertionPointToEnd(parentCtrlBlock);
      if (auto groupPtr = std::get_if<calyx::GroupOp>(&group); groupPtr) {
        rewriter.create<calyx::EnableOp>(groupPtr->getLoc(),
                                         groupPtr->getSymName());
      } else if (auto *pipeSchedPtr = std::get_if<PipelineScheduleable>(&group);
                 pipeSchedPtr) {
        auto &whileOp = pipeSchedPtr->whileOp;

        auto whileCtrlOp =
            buildWhileCtrlOp(whileOp, pipeSchedPtr->initGroups, rewriter);
        rewriter.setInsertionPointToEnd(whileCtrlOp.getBodyBlock());
        auto whileBodyOp =
            rewriter.create<calyx::ParOp>(whileOp.getOperation()->getLoc());
        rewriter.setInsertionPointToEnd(whileBodyOp.getBodyBlock());

        /// Schedule pipeline stages in the parallel group directly.
        auto bodyBlockScheduleables =
            getState<ComponentLoweringState>().getBlockScheduleables(
                whileOp.getBodyBlock());
        for (auto &group : bodyBlockScheduleables)
          if (auto *groupPtr = std::get_if<calyx::GroupOp>(&group); groupPtr)
            rewriter.create<calyx::EnableOp>(groupPtr->getLoc(),
                                             groupPtr->getSymName());
          else
            return whileOp.getOperation()->emitError(
                "Unsupported block schedulable");

        // Add any prologue or epilogue.
        PatternRewriter::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(whileCtrlOp);
        getState<ComponentLoweringState>().createPipelinePrologue(
            whileOp.getOperation(), rewriter);
        rewriter.setInsertionPointAfter(whileCtrlOp);
        getState<ComponentLoweringState>().createPipelineEpilogue(
            whileOp.getOperation(), rewriter);
      } else
        llvm_unreachable("Unknown scheduleable");
    }
    return success();
  }

  /// Schedules a block by inserting a branch argument assignment block (if any)
  /// before recursing into the scheduling of the block innards.
  /// Blocks 'from' and 'to' refer to blocks in the source program.
  /// parentCtrlBlock refers to the control block wherein control operations are
  /// to be inserted.
  LogicalResult schedulePath(PatternRewriter &rewriter,
                             const DenseSet<Block *> &path, Location loc,
                             Block *from, Block *to,
                             Block *parentCtrlBlock) const {
    /// Schedule any registered block arguments to be executed before the body
    /// of the branch.
    rewriter.setInsertionPointToEnd(parentCtrlBlock);
    auto preSeqOp = rewriter.create<calyx::SeqOp>(loc);
    rewriter.setInsertionPointToEnd(preSeqOp.getBodyBlock());
    for (auto barg :
         getState<ComponentLoweringState>().getBlockArgGroups(from, to))
      rewriter.create<calyx::EnableOp>(barg.getLoc(), barg.getSymName());

    return buildCFGControl(path, rewriter, parentCtrlBlock, from, to);
  }

  LogicalResult buildCFGControl(DenseSet<Block *> path,
                                PatternRewriter &rewriter,
                                mlir::Block *parentCtrlBlock,
                                mlir::Block *preBlock,
                                mlir::Block *block) const {
    if (path.count(block) != 0)
      return preBlock->getTerminator()->emitError()
             << "CFG backedge detected. Loops must be raised to 'scf.while' or "
                "'scf.for' operations.";

    rewriter.setInsertionPointToEnd(parentCtrlBlock);
    LogicalResult bbSchedResult =
        scheduleBasicBlock(rewriter, path, parentCtrlBlock, block);
    if (bbSchedResult.failed())
      return bbSchedResult;

    path.insert(block);
    auto successors = block->getSuccessors();
    auto nSuccessors = successors.size();
    if (nSuccessors > 0) {
      auto brOp = dyn_cast<BranchOpInterface>(block->getTerminator());
      assert(brOp);
      if (nSuccessors > 1) {
        /// TODO(mortbopet): we could choose to support ie. std.switch, but it
        /// would probably be easier to just require it to be lowered
        /// beforehand.
        assert(nSuccessors == 2 &&
               "only conditional branches supported for now...");
        /// Wrap each branch inside an if/else.
        auto cond = brOp->getOperand(0);
        auto condGroup = getState<ComponentLoweringState>()
                             .getEvaluatingGroup<calyx::CombGroupOp>(cond);
        auto symbolAttr = FlatSymbolRefAttr::get(
            StringAttr::get(getContext(), condGroup.getSymName()));

        auto ifOp = rewriter.create<calyx::IfOp>(
            brOp->getLoc(), cond, symbolAttr, /*initializeElseBody=*/true);
        rewriter.setInsertionPointToStart(ifOp.getThenBody());
        auto thenSeqOp = rewriter.create<calyx::SeqOp>(brOp.getLoc());
        rewriter.setInsertionPointToStart(ifOp.getElseBody());
        auto elseSeqOp = rewriter.create<calyx::SeqOp>(brOp.getLoc());

        bool trueBrSchedSuccess =
            schedulePath(rewriter, path, brOp.getLoc(), block, successors[0],
                         thenSeqOp.getBodyBlock())
                .succeeded();
        bool falseBrSchedSuccess = true;
        if (trueBrSchedSuccess) {
          falseBrSchedSuccess =
              schedulePath(rewriter, path, brOp.getLoc(), block, successors[1],
                           elseSeqOp.getBodyBlock())
                  .succeeded();
        }

        return success(trueBrSchedSuccess && falseBrSchedSuccess);
      } else {
        /// Schedule sequentially within the current parent control block.
        return schedulePath(rewriter, path, brOp.getLoc(), block,
                            successors.front(), parentCtrlBlock);
      }
    }
    return success();
  }

  calyx::WhileOp buildWhileCtrlOp(PipelineWhileOp whileOp,
                                  SmallVector<calyx::GroupOp> initGroups,
                                  PatternRewriter &rewriter) const {
    Location loc = whileOp.getLoc();
    /// Insert while iter arg initialization group(s). Emit a
    /// parallel group to assign one or more registers all at once.
    {
      PatternRewriter::InsertionGuard g(rewriter);
      auto parOp = rewriter.create<calyx::ParOp>(loc);
      rewriter.setInsertionPointToStart(parOp.getBodyBlock());
      for (calyx::GroupOp group : initGroups)
        rewriter.create<calyx::EnableOp>(group.getLoc(), group.getName());
    }

    /// Insert the while op itself.
    auto cond = whileOp.getConditionValue();
    auto condGroup = getState<ComponentLoweringState>()
                         .getEvaluatingGroup<calyx::CombGroupOp>(cond);
    auto symbolAttr = FlatSymbolRefAttr::get(
        StringAttr::get(getContext(), condGroup.getSymName()));
    auto whileCtrlOp = rewriter.create<calyx::WhileOp>(loc, cond, symbolAttr);

    /// If a bound was specified, add it.
    if (auto bound = whileOp.getBound()) {
      // Subtract the number of iterations unrolled into the prologue.
      auto prologue = getState<ComponentLoweringState>().getPipelinePrologue(
          whileOp.getOperation());
      auto unrolledBound = *bound - prologue.size();
      whileCtrlOp->setAttr("bound", rewriter.getI64IntegerAttr(unrolledBound));
    }

    return whileCtrlOp;
  }
};

/// LateSSAReplacement contains various functions for replacing SSA values that
/// were not replaced during op construction.
class LateSSAReplacement : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult partiallyLowerFuncToComp(FuncOp funcOp,
                                         PatternRewriter &) const override {
    funcOp.walk([&](memref::LoadOp loadOp) {
      if (calyx::singleLoadFromMemory(loadOp)) {
        /// In buildOpGroups we did not replace loadOp's results, to ensure a
        /// link between evaluating groups (which fix the input addresses of a
        /// memory op) and a readData result. Now, we may replace these SSA
        /// values with their memoryOp readData output.
        loadOp.getResult().replaceAllUsesWith(
            getState<ComponentLoweringState>()
                .getMemoryInterface(loadOp.getMemref())
                .readData());
      }
    });

    return success();
  }
};

/// Erases FuncOp operations.
class CleanupFuncOps : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult matchAndRewrite(FuncOp funcOp,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(funcOp);
    return success();
  }

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//
class LoopScheduleToCalyxPass
    : public LoopScheduleToCalyxBase<LoopScheduleToCalyxPass> {
public:
  LoopScheduleToCalyxPass()
      : LoopScheduleToCalyxBase<LoopScheduleToCalyxPass>(),
        partialPatternRes(success()) {}
  void runOnOperation() override;

  LogicalResult setTopLevelFunction(mlir::ModuleOp moduleOp,
                                    std::string &topLevelFunction) {
    if (!topLevelFunctionOpt.empty()) {
      if (SymbolTable::lookupSymbolIn(moduleOp, topLevelFunctionOpt) ==
          nullptr) {
        moduleOp.emitError() << "Top level function '" << topLevelFunctionOpt
                             << "' not found in module.";
        return failure();
      }
      topLevelFunction = topLevelFunctionOpt;
    } else {
      /// No top level function set; infer top level if the module only contains
      /// a single function, else, throw error.
      auto funcOps = moduleOp.getOps<FuncOp>();
      if (std::distance(funcOps.begin(), funcOps.end()) == 1)
        topLevelFunction = (*funcOps.begin()).getSymName().str();
      else {
        moduleOp.emitError()
            << "Module contains multiple functions, but no top level "
               "function was set. Please see --top-level-function";
        return failure();
      }
    }
    return success();
  }

  struct LoweringPattern {
    enum class Strategy { Once, Greedy };
    RewritePatternSet pattern;
    Strategy strategy;
  };

  //// Labels the entry point of a Calyx program.
  /// Furthermore, this function performs validation on the input function,
  /// to ensure that we've implemented the capabilities necessary to convert
  /// it.
  LogicalResult labelEntryPoint(StringRef topLevelFunction) {
    // Program legalization - the partial conversion driver will not run
    // unless some pattern is provided - provide a dummy pattern.
    struct DummyPattern : public OpRewritePattern<mlir::ModuleOp> {
      using OpRewritePattern::OpRewritePattern;
      LogicalResult matchAndRewrite(mlir::ModuleOp,
                                    PatternRewriter &) const override {
        return failure();
      }
    };

    ConversionTarget target(getContext());
    target.addLegalDialect<calyx::CalyxDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addIllegalDialect<hw::HWDialect>();
    target.addIllegalDialect<comb::CombDialect>();

    // For loops should have been lowered to while loops
    target.addIllegalOp<scf::ForOp>();

    // Only accept std operations which we've added lowerings for
    target.addIllegalDialect<FuncDialect>();
    target.addIllegalDialect<ArithDialect>();
    target.addLegalOp<AddIOp, SubIOp, CmpIOp, ShLIOp, ShRUIOp, ShRSIOp, AndIOp,
                      XOrIOp, OrIOp, ExtUIOp, TruncIOp, CondBranchOp, BranchOp,
                      MulIOp, DivUIOp, DivSIOp, RemUIOp, RemSIOp, ReturnOp,
                      arith::ConstantOp, IndexCastOp, FuncOp, ExtSIOp>();

    RewritePatternSet legalizePatterns(&getContext());
    legalizePatterns.add<DummyPattern>(&getContext());
    DenseSet<Operation *> legalizedOps;
    if (applyPartialConversion(getOperation(), target,
                               std::move(legalizePatterns))
            .failed())
      return failure();

    // Program conversion
    return calyx::applyModuleOpConversion(getOperation(), topLevelFunction);
  }

  /// 'Once' patterns are expected to take an additional LogicalResult&
  /// argument, to forward their result state (greedyPatternRewriteDriver
  /// results are skipped for Once patterns).
  template <typename TPattern, typename... PatternArgs>
  void addOncePattern(SmallVectorImpl<LoweringPattern> &patterns,
                      PatternArgs &&...args) {
    RewritePatternSet ps(&getContext());
    ps.add<TPattern>(&getContext(), partialPatternRes, args...);
    patterns.push_back(
        LoweringPattern{std::move(ps), LoweringPattern::Strategy::Once});
  }

  template <typename TPattern, typename... PatternArgs>
  void addGreedyPattern(SmallVectorImpl<LoweringPattern> &patterns,
                        PatternArgs &&...args) {
    RewritePatternSet ps(&getContext());
    ps.add<TPattern>(&getContext(), args...);
    patterns.push_back(
        LoweringPattern{std::move(ps), LoweringPattern::Strategy::Greedy});
  }

  LogicalResult runPartialPattern(RewritePatternSet &pattern, bool runOnce) {
    assert(pattern.getNativePatterns().size() == 1 &&
           "Should only apply 1 partial lowering pattern at once");

    // During component creation, the function body is inlined into the
    // component body for further processing. However, proper control flow
    // will only be established later in the conversion process, so ensure
    // that rewriter optimizations (especially DCE) are disabled.
    GreedyRewriteConfig config;
    config.enableRegionSimplification = false;
    if (runOnce)
      config.maxIterations = 1;

    /// Can't return applyPatternsAndFoldGreedily. Root isn't
    /// necessarily erased so it will always return failed(). Instead,
    /// forward the 'succeeded' value from PartialLoweringPatternBase.
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(pattern),
                                       config);
    return partialPatternRes;
  }

private:
  LogicalResult partialPatternRes;
  std::shared_ptr<calyx::CalyxLoweringState> loweringState = nullptr;
};

void LoopScheduleToCalyxPass::runOnOperation() {
  // Clear internal state. See https://github.com/llvm/circt/issues/3235
  loweringState.reset();
  partialPatternRes = LogicalResult::failure();

  std::string topLevelFunction;
  if (failed(setTopLevelFunction(getOperation(), topLevelFunction))) {
    signalPassFailure();
    return;
  }

  /// Start conversion
  if (failed(labelEntryPoint(topLevelFunction))) {
    signalPassFailure();
    return;
  }
  loweringState = std::make_shared<calyx::CalyxLoweringState>(getOperation(),
                                                              topLevelFunction);

  /// --------------------------------------------------------------------------
  /// If you are a developer, it may be helpful to add a
  /// 'getOperation()->dump()' call after the execution of each stage to
  /// view the transformations that's going on.
  /// --------------------------------------------------------------------------

  /// A mapping is maintained between a function operation and its corresponding
  /// Calyx component.
  DenseMap<FuncOp, calyx::ComponentOp> funcMap;
  SmallVector<LoweringPattern, 8> loweringPatterns;
  calyx::PatternApplicationState patternState;

  /// Creates a new Calyx component for each FuncOp in the inpurt module.
  addOncePattern<FuncOpConversion>(loweringPatterns, patternState, funcMap,
                                   *loweringState);

  /// This pattern converts all index typed values to an i32 integer.
  addOncePattern<calyx::ConvertIndexTypes>(loweringPatterns, patternState,
                                           funcMap, *loweringState);

  /// This pattern creates registers for all basic-block arguments.
  addOncePattern<calyx::BuildBasicBlockRegs>(loweringPatterns, patternState,
                                             funcMap, *loweringState);

  /// This pattern creates registers for the function return values.
  addOncePattern<calyx::BuildReturnRegs>(loweringPatterns, patternState,
                                         funcMap, *loweringState);

  /// This pattern creates registers for iteration arguments of scf.while
  /// operations. Additionally, creates a group for assigning the initial
  /// value of the iteration argument registers.
  addOncePattern<BuildWhileGroups>(loweringPatterns, patternState, funcMap,
                                   *loweringState);

  /// This pattern creates registers for all pipeline stages.
  addOncePattern<BuildPipelineRegs>(loweringPatterns, patternState, funcMap,
                                    *loweringState);

  /// This pattern converts operations within basic blocks to Calyx library
  /// operators. Combinational operations are assigned inside a
  /// calyx::CombGroupOp, and sequential inside calyx::GroupOps.
  /// Sequential groups are registered with the Block* of which the operation
  /// originated from. This is used during control schedule generation. By
  /// having a distinct group for each operation, groups are analogous to SSA
  /// values in the source program.
  addOncePattern<BuildOpGroups>(loweringPatterns, patternState, funcMap,
                                *loweringState);

  /// This pattern creates groups for all pipeline stages.
  addOncePattern<BuildPipelineGroups>(loweringPatterns, patternState, funcMap,
                                      *loweringState);

  /// This pattern traverses the CFG of the program and generates a control
  /// schedule based on the calyx::GroupOp's which were registered for each
  /// basic block in the source function.
  addOncePattern<BuildControl>(loweringPatterns, patternState, funcMap,
                               *loweringState);

  /// This pass recursively inlines use-def chains of combinational logic (from
  /// non-stateful groups) into groups referenced in the control schedule.
  addOncePattern<calyx::InlineCombGroups>(loweringPatterns, patternState,
                                          *loweringState);

  /// This pattern performs various SSA replacements that must be done
  /// after control generation.
  addOncePattern<LateSSAReplacement>(loweringPatterns, patternState, funcMap,
                                     *loweringState);

  /// Eliminate any unused combinational groups. This is done before
  /// calyx::RewriteMemoryAccesses to avoid inferring slice components for
  /// groups that will be removed.
  addGreedyPattern<calyx::EliminateUnusedCombGroups>(loweringPatterns);

  /// This pattern rewrites accesses to memories which are too wide due to
  /// index types being converted to a fixed-width integer type.
  addOncePattern<calyx::RewriteMemoryAccesses>(loweringPatterns, patternState,
                                               *loweringState);

  /// This pattern removes the source FuncOp which has now been converted into
  /// a Calyx component.
  addOncePattern<CleanupFuncOps>(loweringPatterns, patternState, funcMap,
                                 *loweringState);

  /// Sequentially apply each lowering pattern.
  for (auto &pat : loweringPatterns) {
    LogicalResult partialPatternRes = runPartialPattern(
        pat.pattern,
        /*runOnce=*/pat.strategy == LoweringPattern::Strategy::Once);
    if (succeeded(partialPatternRes))
      continue;
    signalPassFailure();
    return;
  }

  //===----------------------------------------------------------------------===//
  // Cleanup patterns
  //===----------------------------------------------------------------------===//
  RewritePatternSet cleanupPatterns(&getContext());
  cleanupPatterns.add<calyx::MultipleGroupDonePattern,
                      calyx::NonTerminatingGroupDonePattern>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                          std::move(cleanupPatterns)))) {
    signalPassFailure();
    return;
  }

  if (ciderSourceLocationMetadata) {
    // Debugging information for the Cider debugger.
    // Reference: https://docs.calyxir.org/debug/cider.html
    SmallVector<Attribute, 16> sourceLocations;
    getOperation()->walk([&](calyx::ComponentOp component) {
      return getCiderSourceLocationMetadata(component, sourceLocations);
    });

    MLIRContext *context = getOperation()->getContext();
    getOperation()->setAttr("calyx.metadata",
                            ArrayAttr::get(context, sourceLocations));
  }
}

} // namespace pipelinetocalyx

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createLoopScheduleToCalyxPass() {
  return std::make_unique<pipelinetocalyx::LoopScheduleToCalyxPass>();
}

} // namespace circt
