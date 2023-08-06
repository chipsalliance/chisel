//===- LowerState.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-lower-state"

using namespace circt;
using namespace arc;
using namespace hw;
using namespace mlir;
using llvm::SmallDenseSet;

//===----------------------------------------------------------------------===//
// Data Structures
//===----------------------------------------------------------------------===//

namespace {

/// Statistics gathered throughout the execution of this pass.
struct Statistics {
  Pass *parent;
  Statistics(Pass *parent) : parent(parent) {}
  using Statistic = Pass::Statistic;

  Statistic matOpsMoved{parent, "mat-ops-moved",
                        "Ops moved during value materialization"};
  Statistic matOpsCloned{parent, "mat-ops-cloned",
                         "Ops cloned during value materialization"};
  Statistic opsPruned{parent, "ops-pruned", "Ops removed as dead code"};
};

/// Lowering info associated with a single primary clock.
struct ClockLowering {
  /// The root clock this lowering is for.
  Value clock;
  /// A `ClockTreeOp` or `PassThroughOp`.
  Operation *treeOp;
  /// Pass statistics.
  Statistics &stats;
  OpBuilder builder;
  /// A mapping from values outside the clock tree to their materialize form
  /// inside the clock tree.
  IRMapping materializedValues;
  /// A cache of AND gates created for aggregating enable conditions.
  DenseMap<std::pair<Value, Value>, Value> andCache;

  ClockLowering(Value clock, Operation *treeOp, Statistics &stats)
      : clock(clock), treeOp(treeOp), stats(stats), builder(treeOp) {
    assert((isa<ClockTreeOp, PassThroughOp>(treeOp)));
    builder.setInsertionPointToStart(&treeOp->getRegion(0).front());
  }

  Value materializeValue(Value value);
  Value getOrCreateAnd(Value lhs, Value rhs, Location loc);
};

struct GatedClockLowering {
  /// Lowering info of the primary clock.
  ClockLowering &clock;
  /// An optional enable condition of the primary clock. May be null.
  Value enable;
};

/// State lowering for a single `HWModuleOp`.
struct ModuleLowering {
  HWModuleOp moduleOp;
  /// Pass statistics.
  Statistics &stats;
  MLIRContext *context;
  DenseMap<Value, std::unique_ptr<ClockLowering>> clockLowerings;
  DenseMap<Value, GatedClockLowering> gatedClockLowerings;
  Value storageArg;

  ModuleLowering(HWModuleOp moduleOp, Statistics &stats)
      : moduleOp(moduleOp), stats(stats), context(moduleOp.getContext()),
        builder(moduleOp) {
    builder.setInsertionPointToStart(moduleOp.getBodyBlock());
  }

  GatedClockLowering getOrCreateClockLowering(Value clock);
  ClockLowering &getOrCreatePassThrough();
  void replaceValueWithStateRead(Value value, Value state);

  void addStorageArg();
  LogicalResult lowerPrimaryInputs();
  LogicalResult lowerPrimaryOutputs();
  LogicalResult lowerStates();
  LogicalResult lowerState(StateOp stateOp);
  LogicalResult lowerState(MemoryOp memOp);
  LogicalResult lowerState(MemoryWritePortOp memWriteOp);
  LogicalResult lowerState(TapOp tapOp);
  LogicalResult lowerExtModules(SymbolTable &symtbl);
  LogicalResult lowerExtModule(InstanceOp instOp);

  LogicalResult cleanup();

private:
  OpBuilder builder;
  SmallVector<StateReadOp, 0> readsToSink;
};
} // namespace

//===----------------------------------------------------------------------===//
// Clock Lowering
//===----------------------------------------------------------------------===//

static bool shouldMaterialize(Operation *op) {
  // Don't materialize arc uses with latency >0, since we handle these in a
  // second pass once all other operations have been moved to their respective
  // clock trees.
  if (auto stateOp = dyn_cast<StateOp>(op); stateOp && stateOp.getLatency() > 0)
    return false;

  if (isa<MemoryOp, AllocStateOp, AllocMemoryOp, AllocStorageOp, ClockTreeOp,
          PassThroughOp, RootInputOp, RootOutputOp, StateWriteOp,
          MemoryWritePortOp, HWInstanceLike>(op))
    return false;

  return true;
}

static bool shouldMaterialize(Value value) {
  assert(value);

  // Block arguments are just used as they are.
  auto *op = value.getDefiningOp();
  if (!op)
    return false;

  return shouldMaterialize(op);
}

/// Materialize a value within this clock tree. This clones or moves all
/// operations required to produce this value inside the clock tree.
Value ClockLowering::materializeValue(Value value) {
  if (!value)
    return {};
  if (!shouldMaterialize(value))
    return value;
  if (auto mapped = materializedValues.lookupOrNull(value))
    return mapped;

  struct WorkItem {
    Operation *op;
    SmallVector<Value, 2> operands;
    WorkItem(Operation *op) : op(op) {}
  };

  SmallPtrSet<Operation *, 8> seen;
  SmallVector<WorkItem> worklist;

  auto addToWorklist = [&](Operation *outerOp) {
    SmallDenseSet<Value> seenOperands;
    auto &workItem = worklist.emplace_back(outerOp);
    outerOp->walk([&](Operation *innerOp) {
      for (auto operand : innerOp->getOperands()) {
        // Skip operands that are defined within the operation itself.
        if (!operand.getParentBlock()->getParentOp()->isProperAncestor(outerOp))
          continue;

        // Skip operands that we have already seen.
        if (!seenOperands.insert(operand).second)
          continue;

        // Skip operands that we have already materialized or that should not
        // be materialized at all.
        if (materializedValues.contains(operand) || !shouldMaterialize(operand))
          continue;

        workItem.operands.push_back(operand);
      }
    });
  };

  seen.insert(value.getDefiningOp());
  addToWorklist(value.getDefiningOp());

  while (!worklist.empty()) {
    auto &workItem = worklist.back();
    if (!workItem.operands.empty()) {
      auto operand = workItem.operands.pop_back_val();
      if (materializedValues.contains(operand) || !shouldMaterialize(operand))
        continue;
      auto *defOp = operand.getDefiningOp();
      if (!seen.insert(defOp).second) {
        defOp->emitError("combinational loop detected");
        return {};
      }
      addToWorklist(defOp);
    } else {
      builder.clone(*workItem.op, materializedValues);
      seen.erase(workItem.op);
      worklist.pop_back();
    }
  }

  return materializedValues.lookup(value);
}

/// Create an AND gate if none with the given operands already exists. Note that
/// the operands may be null, in which case the function will return the
/// non-null operand, or null if both operands are null.
Value ClockLowering::getOrCreateAnd(Value lhs, Value rhs, Location loc) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  auto &slot = andCache[std::make_pair(lhs, rhs)];
  if (!slot)
    slot = builder.create<comb::AndOp>(loc, lhs, rhs);
  return slot;
}

//===----------------------------------------------------------------------===//
// Module Lowering
//===----------------------------------------------------------------------===//

GatedClockLowering ModuleLowering::getOrCreateClockLowering(Value clock) {
  // Look through clock gates.
  if (auto ckgOp = clock.getDefiningOp<ClockGateOp>()) {
    // Reuse the existing lowering for this clock gate if possible.
    if (auto it = gatedClockLowerings.find(clock);
        it != gatedClockLowerings.end())
      return it->second;

    // Get the lowering for the parent clock gate's input clock. This will give
    // us the clock tree to emit things into, alongside the compound enable
    // condition of all the clock gates along the way to the primary clock. All
    // we have to do is to add this clock gate's condition to that list.
    auto info = getOrCreateClockLowering(ckgOp.getInput());
    auto ckgEnable = info.clock.materializeValue(ckgOp.getEnable());
    info.enable =
        info.clock.getOrCreateAnd(info.enable, ckgEnable, ckgOp.getLoc());
    gatedClockLowerings.insert({clock, info});
    return info;
  }

  // Create the `ClockTreeOp` that corresponds to this ungated clock.
  auto &slot = clockLowerings[clock];
  if (!slot) {
    auto treeOp = builder.create<ClockTreeOp>(clock.getLoc(), clock);
    treeOp.getBody().emplaceBlock();
    slot = std::make_unique<ClockLowering>(clock, treeOp, stats);
  }
  return GatedClockLowering{*slot, Value{}};
}

ClockLowering &ModuleLowering::getOrCreatePassThrough() {
  auto &slot = clockLowerings[Value{}];
  if (!slot) {
    auto treeOp = builder.create<PassThroughOp>(moduleOp.getLoc());
    treeOp.getBody().emplaceBlock();
    slot = std::make_unique<ClockLowering>(Value{}, treeOp, stats);
  }
  return *slot;
}

/// Replace all uses of a value with a `StateReadOp` on a state.
void ModuleLowering::replaceValueWithStateRead(Value value, Value state) {
  OpBuilder builder(state.getContext());
  builder.setInsertionPointAfterValue(state);
  auto readOp = builder.create<StateReadOp>(value.getLoc(), state);
  value.replaceAllUsesWith(readOp);
  readsToSink.push_back(readOp);
}

/// Add the global state as an argument to the module's body block.
void ModuleLowering::addStorageArg() {
  assert(!storageArg);
  storageArg = moduleOp.getBodyBlock()->addArgument(
      StorageType::get(context, {}), moduleOp.getLoc());
}

/// Lower the primary inputs of the module to dedicated ops that allocate the
/// inputs in the model's storage.
LogicalResult ModuleLowering::lowerPrimaryInputs() {
  builder.setInsertionPointToStart(moduleOp.getBodyBlock());
  for (auto blockArg : moduleOp.getArguments()) {
    if (blockArg == storageArg)
      continue;
    auto name =
        moduleOp.getArgNames()[blockArg.getArgNumber()].cast<StringAttr>();
    auto intType = blockArg.getType().dyn_cast<IntegerType>();
    if (!intType)
      return mlir::emitError(blockArg.getLoc(), "input ")
             << name << " is of non-integer type " << blockArg.getType();
    auto state = builder.create<RootInputOp>(
        blockArg.getLoc(), StateType::get(intType), name, storageArg);
    replaceValueWithStateRead(blockArg, state);
  }
  return success();
}

/// Lower the primary outputs of the module to dedicated ops that allocate the
/// outputs in the model's storage.
LogicalResult ModuleLowering::lowerPrimaryOutputs() {
  auto outputOp = cast<hw::OutputOp>(moduleOp.getBodyBlock()->getTerminator());
  if (outputOp.getNumOperands() > 0) {
    auto &passThrough = getOrCreatePassThrough();
    for (auto [value, name] :
         llvm::zip(outputOp.getOperands(), moduleOp.getResultNames())) {
      auto intType = value.getType().dyn_cast<IntegerType>();
      if (!intType)
        return mlir::emitError(outputOp.getLoc(), "output ")
               << name << " is of non-integer type " << value.getType();
      auto materializedValue = passThrough.materializeValue(value);
      auto state = builder.create<RootOutputOp>(
          outputOp.getLoc(), StateType::get(intType), name.cast<StringAttr>(),
          storageArg);
      passThrough.builder.create<StateWriteOp>(outputOp.getLoc(), state,
                                               materializedValue, Value{});
    }
  }
  return success();
}

LogicalResult ModuleLowering::lowerStates() {
  for (auto &op : llvm::make_early_inc_range(*moduleOp.getBodyBlock())) {
    auto result = TypeSwitch<Operation *, LogicalResult>(&op)
                      .Case<StateOp, MemoryOp, MemoryWritePortOp, TapOp>(
                          [&](auto op) { return lowerState(op); })
                      .Default(success());
    if (failed(result))
      return failure();
  }
  return success();
}

LogicalResult ModuleLowering::lowerState(StateOp stateOp) {
  // Latency zero arcs incur no state and remain in the IR unmodified.
  if (stateOp.getLatency() == 0)
    return success();

  // We don't support arcs beyond latency 1 yet. These should be easy to add in
  // the future though.
  if (stateOp.getLatency() > 1)
    return stateOp.emitError("state with latency > 1 not supported");

  // Get the clock tree and enable condition for this state's clock. If this arc
  // carries an explicit enable condition, fold that into the enable provided by
  // the clock gates in the arc's clock tree.
  auto info = getOrCreateClockLowering(stateOp.getClock());
  info.enable = info.clock.getOrCreateAnd(
      info.enable, info.clock.materializeValue(stateOp.getEnable()),
      stateOp.getLoc());

  // Allocate the necessary state within the model.
  SmallVector<Value> allocatedStates;
  for (unsigned stateIdx = 0; stateIdx < stateOp.getNumResults(); ++stateIdx) {
    auto type = stateOp.getResult(stateIdx).getType();
    auto intType = dyn_cast<IntegerType>(type);
    if (!intType)
      return stateOp.emitOpError("result ")
             << stateIdx << " has non-integer type " << type
             << "; only integer types are supported";
    auto stateType = StateType::get(intType);
    auto state =
        builder.create<AllocStateOp>(stateOp.getLoc(), stateType, storageArg);
    if (auto names = stateOp->getAttrOfType<ArrayAttr>("names"))
      state->setAttr("name", names[stateIdx]);
    allocatedStates.push_back(state);
  }

  // Create a copy of the arc use with latency zero. This will effectively be
  // the computation of the arc's transfer function, while the latency is
  // implemented through read and write functions.
  SmallVector<Value> materializedOperands;
  materializedOperands.reserve(stateOp.getInputs().size());
  SmallVector<Value> inputs(stateOp.getInputs());

  for (auto input : inputs)
    materializedOperands.push_back(info.clock.materializeValue(input));

  OpBuilder nonResetBuilder = info.clock.builder;
  if (stateOp.getReset()) {
    auto materializedReset = info.clock.materializeValue(stateOp.getReset());
    auto ifOp = info.clock.builder.create<scf::IfOp>(stateOp.getLoc(),
                                                     materializedReset, true);

    for (auto [alloc, resTy] :
         llvm::zip(allocatedStates, stateOp.getResultTypes())) {
      if (!resTy.isa<IntegerType>())
        stateOp->emitOpError("Non-integer result not supported yet!");

      auto thenBuilder = ifOp.getThenBodyBuilder();
      Value constZero =
          thenBuilder.create<hw::ConstantOp>(stateOp.getLoc(), resTy, 0);
      thenBuilder.create<StateWriteOp>(stateOp.getLoc(), alloc, constZero,
                                       Value());
    }

    nonResetBuilder = ifOp.getElseBodyBuilder();
  }

  stateOp->dropAllReferences();

  auto newStateOp = nonResetBuilder.create<StateOp>(
      stateOp.getLoc(), stateOp.getArcAttr(), stateOp.getResultTypes(), Value{},
      Value{}, 0, materializedOperands);

  // Create the write ops that write the result of the transfer function to the
  // allocated state storage.
  for (auto [alloc, result] :
       llvm::zip(allocatedStates, newStateOp.getResults()))
    nonResetBuilder.create<StateWriteOp>(stateOp.getLoc(), alloc, result,
                                         info.enable);

  // Replace all uses of the arc with reads from the allocated state.
  for (auto [alloc, result] : llvm::zip(allocatedStates, stateOp.getResults()))
    replaceValueWithStateRead(result, alloc);
  if (&*builder.getInsertionPoint() == stateOp)
    builder.setInsertionPointAfter(stateOp);
  stateOp.erase();
  return success();
}

LogicalResult ModuleLowering::lowerState(MemoryOp memOp) {
  auto allocMemOp = builder.create<AllocMemoryOp>(
      memOp.getLoc(), memOp.getType(), storageArg, memOp->getAttrs());
  memOp.replaceAllUsesWith(allocMemOp.getResult());
  if (&*builder.getInsertionPoint() == memOp)
    builder.setInsertionPointAfter(memOp);
  memOp.erase();
  return success();
}

LogicalResult ModuleLowering::lowerState(MemoryWritePortOp memWriteOp) {
  if (memWriteOp.getLatency() > 1)
    return memWriteOp->emitOpError("latencies > 1 not supported yet");

  // Get the clock tree and enable condition for this write port's clock. If the
  // port carries an explicit enable condition, fold that into the enable
  // provided by the clock gates in the port's clock tree.
  auto info = getOrCreateClockLowering(memWriteOp.getClock());

  SmallVector<Value> materializedInputs;
  for (auto input : memWriteOp.getInputs())
    materializedInputs.push_back(info.clock.materializeValue(input));
  ValueRange results =
      info.clock.builder
          .create<CallOp>(memWriteOp.getLoc(), memWriteOp.getArcResultTypes(),
                          memWriteOp.getArc(), materializedInputs)
          ->getResults();

  auto enable =
      memWriteOp.getEnable() ? results[memWriteOp.getEnableIdx()] : Value();
  info.enable =
      info.clock.getOrCreateAnd(info.enable, enable, memWriteOp.getLoc());

  // Materialize the operands for the write op within the surrounding clock
  // tree.
  auto address = results[memWriteOp.getAddressIdx()];
  auto data = results[memWriteOp.getDataIdx()];
  if (memWriteOp.getMask()) {
    Value mask = results[memWriteOp.getMaskIdx(static_cast<bool>(enable))];
    Value oldData = info.clock.builder.create<arc::MemoryReadOp>(
        mask.getLoc(), data.getType(), memWriteOp.getMemory(), address);
    Value allOnes = info.clock.builder.create<hw::ConstantOp>(
        mask.getLoc(), oldData.getType(), -1);
    Value negatedMask = info.clock.builder.create<comb::XorOp>(
        mask.getLoc(), mask, allOnes, true);
    Value maskedOldData = info.clock.builder.create<comb::AndOp>(
        mask.getLoc(), negatedMask, oldData, true);
    Value maskedNewData =
        info.clock.builder.create<comb::AndOp>(mask.getLoc(), mask, data, true);
    data = info.clock.builder.create<comb::OrOp>(mask.getLoc(), maskedOldData,
                                                 maskedNewData, true);
  }

  // Filter the list of reads such that they only contain read ports within the
  // same clock domain.
  // TODO: This really needs to go now that we have the `LegalizeStateUpdates`
  // pass. Instead, we should just have memory read and write accesses all over
  // the place, then lower them into proper reads and writes, and let the
  // legalization pass insert any necessary temporaries.
  // SmallVector<Value> newReads;
  // for (auto read : memWriteOp.getReads()) {
  //   if (auto readOp = read.getDefiningOp<MemoryReadPortOp>())
  //     // HACK: This check for a constant clock is ugly. The read ops should
  //     // instead be replicated for every clock domain that they are used in,
  //     // and then dependencies should be tracked between reads and writes
  //     // within that clock domain. Lack of a clock (comb mem) should be
  //     // handled properly as well. Presence of a clock should group the read
  //     // under that clock as expected, and write to a "read buffer" that can
  //     // be read again by actual uses in different clock domains. LLVM
  //     // lowering already has such a read buffer. Just need to formalize it.
  //     if (!readOp.getClock().getDefiningOp<hw::ConstantOp>() &&
  //         getOrCreateClockLowering(readOp.getClock()).clock.clock !=
  //             info.clock.clock)
  //       continue;
  //   newReads.push_back(info.clock.materializeValue(read));
  // }
  // TODO: This just creates a write without any reads. Instead, there should be
  // a separate memory write op that we can lower to here which doesn't need to
  // track its reads, but will get legalized by the LegalizeStateUpdate pass.
  info.clock.builder.create<MemoryWriteOp>(
      memWriteOp.getLoc(), memWriteOp.getMemory(), address, info.enable, data);
  if (&*builder.getInsertionPoint() == memWriteOp)
    builder.setInsertionPointAfter(memWriteOp);
  memWriteOp.erase();
  return success();
}

// Add state for taps into the passthrough block.
LogicalResult ModuleLowering::lowerState(TapOp tapOp) {
  auto intType = tapOp.getValue().getType().dyn_cast<IntegerType>();
  if (!intType)
    return mlir::emitError(tapOp.getLoc(), "tapped value ")
           << tapOp.getNameAttr() << " is of non-integer type "
           << tapOp.getValue().getType();
  auto &passThrough = getOrCreatePassThrough();
  auto materializedValue = passThrough.materializeValue(tapOp.getValue());
  auto state = builder.create<AllocStateOp>(
      tapOp.getLoc(), StateType::get(intType), storageArg, true);
  state->setAttr("name", tapOp.getNameAttr());
  passThrough.builder.create<StateWriteOp>(tapOp.getLoc(), state,
                                           materializedValue, Value{});
  if (&*builder.getInsertionPoint() == tapOp)
    builder.setInsertionPointAfter(tapOp);
  tapOp.erase();
  return success();
}

/// Lower all instances of external modules to internal inputs/outputs to be
/// driven from outside of the design.
LogicalResult ModuleLowering::lowerExtModules(SymbolTable &symtbl) {
  for (auto op : llvm::make_early_inc_range(moduleOp.getOps<InstanceOp>()))
    if (isa<HWModuleExternOp>(symtbl.lookup(op.getModuleNameAttr().getAttr())))
      if (failed(lowerExtModule(op)))
        return failure();
  return success();
}

LogicalResult ModuleLowering::lowerExtModule(InstanceOp instOp) {
  LLVM_DEBUG(llvm::dbgs() << "- Lowering extmodule "
                          << instOp.getInstanceNameAttr() << "\n");

  SmallString<32> baseName(instOp.getInstanceName());
  auto baseNameLen = baseName.size();

  // Lower the inputs of the extmodule as state that is only written.
  for (auto [operand, name] :
       llvm::zip(instOp.getOperands(), instOp.getArgNames())) {
    LLVM_DEBUG(llvm::dbgs()
               << "  - Input " << name << " : " << operand.getType() << "\n");
    auto intType = operand.getType().dyn_cast<IntegerType>();
    if (!intType)
      return mlir::emitError(operand.getLoc(), "input ")
             << name << " of extern module " << instOp.getModuleNameAttr()
             << " instance " << instOp.getInstanceNameAttr()
             << " is of non-integer type " << operand.getType();
    baseName.resize(baseNameLen);
    baseName += '/';
    baseName += cast<StringAttr>(name).getValue();
    auto &passThrough = getOrCreatePassThrough();
    auto state = builder.create<AllocStateOp>(
        instOp.getLoc(), StateType::get(intType), storageArg);
    state->setAttr("name", builder.getStringAttr(baseName));
    passThrough.builder.create<StateWriteOp>(
        instOp.getLoc(), state, passThrough.materializeValue(operand), Value{});
  }

  // Lower the outputs of the extmodule as state that is only read.
  for (auto [result, name] :
       llvm::zip(instOp.getResults(), instOp.getResultNames())) {
    LLVM_DEBUG(llvm::dbgs()
               << "  - Output " << name << " : " << result.getType() << "\n");
    auto intType = result.getType().dyn_cast<IntegerType>();
    if (!intType)
      return mlir::emitError(result.getLoc(), "output ")
             << name << " of extern module " << instOp.getModuleNameAttr()
             << " instance " << instOp.getInstanceNameAttr()
             << " is of non-integer type " << result.getType();
    baseName.resize(baseNameLen);
    baseName += '/';
    baseName += cast<StringAttr>(name).getValue();
    auto state = builder.create<AllocStateOp>(
        result.getLoc(), StateType::get(intType), storageArg);
    state->setAttr("name", builder.getStringAttr(baseName));
    replaceValueWithStateRead(result, state);
  }

  if (&*builder.getInsertionPoint() == instOp)
    builder.setInsertionPointAfter(instOp);
  instOp.erase();
  return success();
}

LogicalResult ModuleLowering::cleanup() {

  // Establish an order among all operations (to avoid an O(n²) pathological
  // pattern with `moveBefore`) and replicate read operations into the blocks
  // where they have uses. The established order is used to create the read
  // operation as late in the block as possible, just before the first use.
  DenseMap<Operation *, unsigned> opOrder;
  moduleOp.walk([&](Operation *op) { opOrder.insert({op, opOrder.size()}); });
  for (auto readToSink : readsToSink) {
    SmallDenseMap<Block *, std::pair<StateReadOp, unsigned>> readsByBlock;
    for (auto &use : llvm::make_early_inc_range(readToSink->getUses())) {
      auto *user = use.getOwner();
      auto userOrder = opOrder.lookup(user);
      auto &localRead = readsByBlock[user->getBlock()];
      if (!localRead.first) {
        localRead.first = OpBuilder(user).cloneWithoutRegions(readToSink);
        localRead.second = userOrder;
      } else if (userOrder < localRead.second) {
        localRead.first->moveBefore(user);
        localRead.second = userOrder;
      }
      use.set(localRead.first);
    }
    readToSink.erase();
  }

  // For each operation left in the module body, make sure that all uses are in
  // the module body as well, not inside any clock trees. The clock trees should
  // have created copies of the operations that they need for computation.
  for (auto &op : llvm::make_early_inc_range(*moduleOp.getBodyBlock())) {
    if (!shouldMaterialize(&op) || isa<hw::OutputOp>(&op))
      continue;
    for (auto *user : op.getUsers()) {
      auto *clockParent = user->getParentOp();
      while (clockParent && !isa<ClockTreeOp, PassThroughOp>(clockParent))
        clockParent = clockParent->getParentOp();
      if (!clockParent)
        continue;
      auto d = op.emitError(
          "has uses in clock trees but has been left outside the clock tree");
      d.attachNote(user->getLoc()) << "use is here";
      return failure();
    }

    // Delete ops as we go.
    if (!llvm::any_of(op.getUsers(),
                      [&](auto *user) { return isa<ClockTreeOp>(user); })) {
      op.dropAllReferences();
      op.dropAllUses();
      op.erase();
      ++stats.opsPruned;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerStatePass : public LowerStateBase<LowerStatePass> {
  LowerStatePass() = default;
  LowerStatePass(const LowerStatePass &pass) : LowerStatePass() {}

  void runOnOperation() override;
  LogicalResult runOnModule(HWModuleOp moduleOp, SymbolTable &symtbl);

  Statistics stats{this};
};
} // namespace

void LowerStatePass::runOnOperation() {
  auto &symtbl = getAnalysis<SymbolTable>();
  SmallVector<HWModuleExternOp> extModules;
  for (auto &op : llvm::make_early_inc_range(getOperation().getOps())) {
    if (auto moduleOp = dyn_cast<HWModuleOp>(&op)) {
      if (failed(runOnModule(moduleOp, symtbl)))
        return signalPassFailure();
    } else if (auto extModuleOp = dyn_cast<HWModuleExternOp>(&op)) {
      extModules.push_back(extModuleOp);
    }
  }
  for (auto op : extModules)
    op.erase();

  // Lower remaining MemoryReadPort ops to MemoryRead ops. This can occur when
  // the fan-in of a MemoryReadPortOp contains another such operation and is
  // materialized before the one in the fan-in as the MemoryReadPortOp is not
  // marked as a fan-in blocking/termination operation in `shouldMaterialize`.
  // Adding it there can lead to dominance issues which would then have to be
  // resolved instead.
  SetVector<DefineOp> arcsToLower;
  OpBuilder builder(getOperation());
  getOperation()->walk([&](MemoryReadPortOp memReadOp) {
    if (auto defOp = memReadOp->getParentOfType<DefineOp>())
      arcsToLower.insert(defOp);

    builder.setInsertionPoint(memReadOp);
    Value newRead = builder.create<MemoryReadOp>(
        memReadOp.getLoc(), memReadOp.getMemory(), memReadOp.getAddress());
    memReadOp.replaceAllUsesWith(newRead);
    memReadOp.erase();
  });

  SymbolTableCollection symbolTable;
  mlir::SymbolUserMap userMap(symbolTable, getOperation());
  for (auto defOp : arcsToLower) {
    auto *terminator = defOp.getBodyBlock().getTerminator();
    builder.setInsertionPoint(terminator);
    builder.create<func::ReturnOp>(terminator->getLoc(),
                                   terminator->getOperands());
    terminator->erase();
    builder.setInsertionPoint(defOp);
    auto funcOp = builder.create<func::FuncOp>(defOp.getLoc(), defOp.getName(),
                                               defOp.getFunctionType());
    funcOp->setAttr("llvm.linkage",
                    LLVM::LinkageAttr::get(builder.getContext(),
                                           LLVM::linkage::Linkage::Internal));
    funcOp.getBody().takeBody(defOp.getBody());

    for (auto *user : userMap.getUsers(defOp)) {
      builder.setInsertionPoint(user);
      ValueRange results = builder
                               .create<func::CallOp>(
                                   user->getLoc(), funcOp,
                                   cast<CallOpInterface>(user).getArgOperands())
                               ->getResults();
      user->replaceAllUsesWith(results);
      user->erase();
    }

    defOp.erase();
  }
}

LogicalResult LowerStatePass::runOnModule(HWModuleOp moduleOp,
                                          SymbolTable &symtbl) {
  LLVM_DEBUG(llvm::dbgs() << "Lowering state in `" << moduleOp.getModuleName()
                          << "`\n");
  ModuleLowering lowering(moduleOp, stats);
  lowering.addStorageArg();
  if (failed(lowering.lowerStates()))
    return failure();
  if (failed(lowering.lowerExtModules(symtbl)))
    return failure();

  // Since we don't yet support derived clocks, simply delete all clock trees
  // that are not driven by a primary input and emit a warning.
  for (auto clockTreeOp :
       llvm::make_early_inc_range(moduleOp.getOps<ClockTreeOp>())) {
    if (clockTreeOp.getClock().isa<BlockArgument>())
      continue;
    auto d = mlir::emitWarning(clockTreeOp.getClock().getLoc(),
                               "unsupported derived clock ignored");
    d.attachNote()
        << "only clocks through a top-level input are supported at the moment";
    clockTreeOp.erase();
  }

  if (failed(lowering.lowerPrimaryInputs()))
    return failure();
  if (failed(lowering.lowerPrimaryOutputs()))
    return failure();

  // Clean up the module body which contains a lot of operations that the
  // pessimistic value materialization has left behind because it couldn't
  // reliably determine that the ops were no longer needed.
  if (failed(lowering.cleanup()))
    return failure();

  // Replace the `HWModuleOp` with a `ModelOp`.
  moduleOp.getBodyBlock()->eraseArguments(
      [&](auto arg) { return arg != lowering.storageArg; });
  moduleOp.getBodyBlock()->getTerminator()->erase();
  ImplicitLocOpBuilder builder(moduleOp.getLoc(), moduleOp);
  auto modelOp =
      builder.create<ModelOp>(moduleOp.getLoc(), moduleOp.getModuleNameAttr());
  modelOp.getBody().takeBody(moduleOp.getBody());
  moduleOp->erase();
  return success();
}

std::unique_ptr<Pass> arc::createLowerStatePass() {
  return std::make_unique<LowerStatePass>();
}
