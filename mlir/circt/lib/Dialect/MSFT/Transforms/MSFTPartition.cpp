//===- MSFTPartion.cpp - MSFT Partitioning pass -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MSFTPassCommon.h"
#include "PassDetails.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "circt/Dialect/MSFT/MSFTOpInterfaces.h"
#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/MSFT/MSFTPasses.h"
#include "circt/Support/Namespace.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace circt;
using namespace msft;

/// Is this operation "free" and copy-able?
static bool isWireManipulationOp(Operation *op) {
  return isa<hw::ArrayConcatOp, hw::ArrayCreateOp, hw::ArrayGetOp,
             hw::ArraySliceOp, hw::StructCreateOp, hw::StructExplodeOp,
             hw::StructExtractOp, hw::StructInjectOp, hw::StructCreateOp,
             hw::ConstantOp>(op);
}

static SymbolRefAttr getPart(Operation *op) {
  return op->getAttrOfType<SymbolRefAttr>("targetDesignPartition");
}

namespace {
struct PartitionPass : public PartitionBase<PartitionPass>, MSFTPassCommon {
  void runOnOperation() override;

private:
  void partition(MSFTModuleOp mod);
  MSFTModuleOp partition(DesignPartitionOp part, Block *partBlock);

  void bubbleUp(MSFTModuleOp mod, Block *partBlock);
  void bubbleUpGlobalRefs(Operation *op, StringAttr parentMod,
                          StringAttr parentName,
                          llvm::DenseSet<hw::GlobalRefAttr> &refsMoved);
  void pushDownGlobalRefs(Operation *op, DesignPartitionOp partOp,
                          llvm::SetVector<Attribute> &newGlobalRefs);

  // Tag wire manipulation ops in this module.
  static void
  copyWireOps(MSFTModuleOp,
              DenseMap<SymbolRefAttr, DenseSet<Operation *>> &perPartOpsToMove);

  MLIRContext *ctxt;
};
} // anonymous namespace

void PartitionPass::runOnOperation() {
  ModuleOp outerMod = getOperation();
  ctxt = outerMod.getContext();
  topLevelSyms.addDefinitions(outerMod);
  if (failed(verifyInstances(outerMod))) {
    signalPassFailure();
    return;
  }

  // Get a properly sorted list, then partition the mods in order.
  SmallVector<MSFTModuleOp, 64> sortedMods;
  getAndSortModules(outerMod, sortedMods);

  for (auto mod : sortedMods) {
    // Make partition's job easier by cleaning up first.
    (void)mlir::applyPatternsAndFoldGreedily(mod,
                                             mlir::FrozenRewritePatternSet());
    // Do the partitioning.
    partition(mod);
    // Cleanup whatever mess we made.
    (void)mlir::applyPatternsAndFoldGreedily(mod,
                                             mlir::FrozenRewritePatternSet());
  }
}

/// Determine if 'op' is driven exclusively by other tagged ops or wires which
/// are themselves exclusively driven by tagged ops. Recursive but memoized via
/// `seen`.
static bool isDrivenByPartOpsOnly(Operation *op, const IRMapping &partOps,
                                  DenseMap<Operation *, bool> &seen) {
  auto prevResult = seen.find(op);
  if (prevResult != seen.end())
    return prevResult->second;
  bool &result = seen[op];
  // Default to true.
  result = true;

  for (Value oper : op->getOperands()) {
    if (partOps.contains(oper))
      continue;
    if (oper.isa<BlockArgument>())
      continue;
    Operation *defOp = oper.getDefiningOp();
    if (!isWireManipulationOp(defOp) ||
        !isDrivenByPartOpsOnly(defOp, partOps, seen))
      result = false;
  }
  return result;
}

/// Move the list of tagged operations in to 'partBlock' and copy/move any free
/// (wire) ops connecting them in also. If 'extendMaximalUp` is specified,
/// attempt to copy all the way up to the block args.
void copyIntoPart(ArrayRef<Operation *> taggedOps, Block *partBlock,
                  bool extendMaximalUp) {
  IRMapping map;
  if (taggedOps.empty())
    return;
  OpBuilder b(taggedOps[0]->getContext());
  // Copy all of the ops listed.
  for (Operation *op : taggedOps) {
    op->moveBefore(partBlock, partBlock->end());
    for (Value result : op->getResults())
      map.map(result, result);
  }

  // Memoization space.
  DenseMap<Operation *, bool> seen;

  // Treat the 'partBlock' as a queue, iterating through and appending as
  // necessary.
  for (Operation &op : *partBlock) {
    // Make sure we are always appending.
    b.setInsertionPointToEnd(partBlock);

    // Go through the operands and copy any which we can.
    for (auto &opOper : op.getOpOperands()) {
      Value operValue = opOper.get();
      assert(operValue);

      // Check if there's already a copied op for this value.
      Value existingValue = map.lookupOrNull(operValue);
      if (existingValue) {
        opOper.set(existingValue);
        continue;
      }

      // Determine if we can copy the op into our partition.
      Operation *defOp = operValue.getDefiningOp();
      if (!defOp)
        continue;

      // We don't copy anything which isn't "free".
      if (!isWireManipulationOp(defOp))
        continue;

      // Copy operand wire ops into the partition.
      //   If `extendMaximalUp` is set, we want to copy unconditionally.
      //   Otherwise, we only want to copy wire ops which connect this operation
      //   to another in the partition.
      if (extendMaximalUp || isDrivenByPartOpsOnly(defOp, map, seen)) {
        // Optimization: if all the consumers of this wire op are in the
        // partition, move instead of clone.
        if (llvm::all_of(defOp->getUsers(), [&](Operation *user) {
              return user->getBlock() == partBlock;
            })) {
          defOp->moveBefore(partBlock, b.getInsertionPoint());
        } else {
          b.insert(defOp->clone(map));
          opOper.set(map.lookup(opOper.get()));
        }
      }
    }

    // Move any "free" consumers which we can.
    for (auto *user : llvm::make_early_inc_range(op.getUsers())) {
      // Stop if it's not "free" or already in a partition.
      if (!isWireManipulationOp(user) || getPart(user) ||
          user->getBlock() == partBlock)
        continue;
      // Op must also only have its operands driven (or indirectly driven) by
      // ops in the partition.
      if (!isDrivenByPartOpsOnly(user, map, seen))
        continue;

      // All the conditions are met, move it!
      user->moveBefore(partBlock, partBlock->end());
      // Mark it as being in the block by putting it into the map.
      for (Value result : user->getResults())
        map.map(result, result);
      // Re-map the inputs to results in the block, if they exist.
      for (OpOperand &oper : user->getOpOperands())
        oper.set(map.lookupOrDefault(oper.get()));
    }
  }
}

/// Move tagged ops into separate blocks. Copy any wire ops connecting them as
/// well.
void copyInto(MSFTModuleOp mod, DenseMap<SymbolRefAttr, Block *> &perPartBlocks,
              Block *nonLocalBlock) {
  DenseMap<SymbolRefAttr, SmallVector<Operation *, 8>> perPartTaggedOps;
  SmallVector<Operation *, 16> nonLocalTaggedOps;

  // Bucket the ops by partition tag.
  mod.walk([&](Operation *op) {
    auto partRef = getPart(op);
    if (!partRef)
      return;
    auto partBlockF = perPartBlocks.find(partRef);
    if (partBlockF != perPartBlocks.end())
      perPartTaggedOps[partRef].push_back(op);
    else
      nonLocalTaggedOps.push_back(op);
  });

  // Copy into the appropriate partition block.
  for (auto &partOpQueuePair : perPartTaggedOps) {
    copyIntoPart(partOpQueuePair.second, perPartBlocks[partOpQueuePair.first],
                 false);
  }
  copyIntoPart(nonLocalTaggedOps, nonLocalBlock, true);
}

void PartitionPass::partition(MSFTModuleOp mod) {
  auto modSymbol = SymbolTable::getSymbolName(mod);

  // Construct all the blocks we're going to need.
  Block *nonLocal = mod.addBlock();
  DenseMap<SymbolRefAttr, Block *> perPartBlocks;
  mod.walk([&](DesignPartitionOp part) {
    SymbolRefAttr partRef =
        SymbolRefAttr::get(modSymbol, {SymbolRefAttr::get(part)});
    perPartBlocks[partRef] = mod.addBlock();
  });

  // Sort the tagged ops into ops to hoist (bubble up) and per-partition blocks.
  copyInto(mod, perPartBlocks, nonLocal);

  // Hoist the appropriate ops and erase the partition block.
  if (!nonLocal->empty())
    bubbleUp(mod, nonLocal);
  nonLocal->dropAllReferences();
  nonLocal->dropAllDefinedValueUses();
  mod.getBlocks().remove(nonLocal);

  // Sink all of the "locally-tagged" ops into new partition modules.
  for (auto part :
       llvm::make_early_inc_range(mod.getOps<DesignPartitionOp>())) {
    SymbolRefAttr partRef =
        SymbolRefAttr::get(modSymbol, {SymbolRefAttr::get(part)});
    Block *partBlock = perPartBlocks[partRef];
    partition(part, partBlock);
    part.erase();
  }
}

/// Heuristics to get the entity name.
static StringRef getOpName(Operation *op) {
  StringAttr name;
  if ((name = op->getAttrOfType<StringAttr>("name")) && name.size())
    return name.getValue();
  if (auto innerSym = op->getAttrOfType<hw::InnerSymAttr>("inner_sym"))
    return innerSym.getSymName().getValue();
  return op->getName().getStringRef();
}

/// Try to set the entity name.
/// TODO: this needs to be more complex to deal with renaming symbols.
static void setEntityName(Operation *op, const Twine &name) {
  StringAttr nameAttr = StringAttr::get(op->getContext(), name);
  if (op->hasAttrOfType<StringAttr>("name"))
    op->setAttr("name", nameAttr);
  if (op->hasAttrOfType<hw::InnerSymAttr>("inner_sym"))
    op->setAttr("inner_sym", hw::InnerSymAttr::get(nameAttr));
}

/// Heuristics to get the output name.
static StringRef getResultName(OpResult res, const SymbolCache &syms,
                               std::string &buff) {

  StringRef valName = getValueName(res, syms, buff);
  if (!valName.empty())
    return valName;
  if (res.getOwner()->getNumResults() == 1)
    return {};

  // Fallback. Not ideal.
  buff.clear();
  llvm::raw_string_ostream(buff) << "out" << res.getResultNumber();
  return buff;
}

/// Heuristics to get the input name.
static StringRef getOperandName(OpOperand &oper, const SymbolCache &syms,
                                std::string &buff) {
  Operation *op = oper.getOwner();
  if (auto inst = dyn_cast<InstanceOp>(op)) {
    Operation *modOp = syms.getDefinition(inst.getModuleNameAttr());
    if (modOp) { // If modOp isn't in the cache, it's probably a new module;
      assert(isAnyModule(modOp) && "Instance must point to a module");
      auto mod = cast<hw::HWModuleLike>(modOp);
      return mod.getInputName(oper.getOperandNumber());
    }
  }
  if (auto blockArg = oper.get().dyn_cast<BlockArgument>()) {
    auto mod =
        cast<hw::HWModuleLike>(blockArg.getOwner()->getParent()->getParentOp());
    return mod.getInputName(blockArg.getArgNumber());
  }

  if (oper.getOwner()->getNumOperands() == 1)
    return "in";

  // Fallback. Not ideal.
  buff.clear();
  llvm::raw_string_ostream(buff) << "in" << oper.getOperandNumber();
  return buff;
}

/// Helper to get the circt.globalRef attribute.
static ArrayAttr getGlobalRefs(Operation *op) {
  return op->getAttrOfType<ArrayAttr>(hw::GlobalRefAttr::DialectAttrName);
}

/// Helper to update GlobalRefOps after referenced ops bubble up.
void PartitionPass::bubbleUpGlobalRefs(
    Operation *op, StringAttr parentMod, StringAttr parentName,
    llvm::DenseSet<hw::GlobalRefAttr> &refsMoved) {
  auto globalRefs = getGlobalRefs(op);
  if (!globalRefs)
    return;

  auto innerSym = op->getAttrOfType<hw::InnerSymAttr>("inner_sym");

  for (auto globalRef : globalRefs.getAsRange<hw::GlobalRefAttr>()) {
    // Resolve the GlobalRefOp and get its path.
    auto refSymbol = globalRef.getGlblSym();
    auto globalRefOp = dyn_cast_or_null<hw::GlobalRefOp>(
        topLevelSyms.getDefinition(refSymbol));
    assert(globalRefOp && "symbol must reference a GlobalRefOp");
    auto oldPath = globalRefOp.getNamepath().getValue();
    assert(!oldPath.empty());

    // If the path already points to the target design partition, we are done.
    auto leafModule = oldPath.back().cast<hw::InnerRefAttr>().getModule();
    auto partAttr = op->getAttrOfType<SymbolRefAttr>("targetDesignPartition");
    if (partAttr.getRootReference() == leafModule)
      return;
    assert(oldPath.size() > 1);

    // Find the index of the node in the path that points to the opName. The
    // previous node in the path must point to parentName.
    size_t opIndex = 0;
    for (; opIndex < oldPath.size(); ++opIndex) {
      auto oldNode = oldPath[opIndex].cast<hw::InnerRefAttr>();
      if (oldNode.getModule() == parentMod)
        break;
    }

    assert(0 < opIndex && opIndex < oldPath.size());
    auto parentIndex = opIndex - 1;
    auto parentNode = oldPath[parentIndex].cast<hw::InnerRefAttr>();
    assert(parentNode.getName() == parentName);

    // Split the old path into two chunks: the parent chunk is everything before
    // the node pointing to parentName, and the child chunk is everything after
    // the node pointing to opName.
    auto parentChunk = oldPath.take_front(parentIndex);
    auto childChunk = oldPath.take_back((oldPath.size() - 1) - opIndex);

    // Splice together the nodes that parentName and opName point to.
    auto splicedNode =
        hw::InnerRefAttr::get(parentNode.getModule(), innerSym.getSymName());

    // Construct a new path from the parentChunk, splicedNode, and childChunk.
    SmallVector<Attribute> newPath(parentChunk.begin(), parentChunk.end());
    newPath.push_back(splicedNode);
    newPath.append(childChunk.begin(), childChunk.end());

    // Update the path on the GlobalRefOp.
    auto newPathAttr = ArrayAttr::get(op->getContext(), newPath);
    globalRefOp.setNamepathAttr(newPathAttr);

    refsMoved.insert(globalRef);
  }
}

/// Helper to update GlobalRefops after referenced ops are pushed down.
void PartitionPass::pushDownGlobalRefs(
    Operation *op, DesignPartitionOp partOp,
    llvm::SetVector<Attribute> &newGlobalRefs) {
  auto globalRefs = getGlobalRefs(op);
  if (!globalRefs)
    return;

  for (auto globalRef : globalRefs.getAsRange<hw::GlobalRefAttr>()) {
    // Resolve the GlobalRefOp and get its path.
    auto refSymbol = globalRef.getGlblSym();
    auto globalRefOp = dyn_cast_or_null<hw::GlobalRefOp>(
        topLevelSyms.getDefinition(refSymbol));
    assert(globalRefOp && "symbol must reference a GlobalRefOp");
    auto oldPath = globalRefOp.getNamepath().getValue();
    assert(!oldPath.empty());

    // Get the module containing the partition and the partition's name.
    auto partAttr = op->getAttrOfType<SymbolRefAttr>("targetDesignPartition");
    auto partMod = partAttr.getRootReference();
    auto partName = partAttr.getLeafReference();
    auto partModName = partOp.getVerilogNameAttr();
    assert(partModName);

    // Find the index of the node in the path that points to the innerSym.
    auto innerSym = op->getAttrOfType<hw::InnerSymAttr>("inner_sym");
    size_t opIndex = 0;
    for (; opIndex < oldPath.size(); ++opIndex) {
      auto oldNode = oldPath[opIndex].cast<hw::InnerRefAttr>();
      if (oldNode.getModule() == partMod)
        break;
    }
    assert(opIndex < oldPath.size());

    // If this path already points to the design partition, we are done.
    if (oldPath[opIndex].cast<hw::InnerRefAttr>().getModule() == partModName)
      return;

    // Split the old path into two chunks: the parent chunk is everything before
    // the node pointing to innerSym, and the child chunk is everything after
    // the node pointing to innerSym.
    auto parentChunk = oldPath.take_front(opIndex);
    auto childChunk = oldPath.take_back((oldPath.size() - 1) - opIndex);

    // Create a new node for the partition within the partition's parent module,
    // and a new node for the op within the partition module.
    auto partRef = hw::InnerRefAttr::get(partMod, partName);
    auto leafRef = hw::InnerRefAttr::get(partModName, innerSym.getSymName());

    // Construct a new path from the parentChunk, partRef, leafRef, and
    // childChunk.
    SmallVector<Attribute> newPath(parentChunk.begin(), parentChunk.end());
    newPath.push_back(partRef);
    newPath.push_back(leafRef);
    newPath.append(childChunk.begin(), childChunk.end());

    // Update the path on the GlobalRefOp.
    auto newPathAttr = ArrayAttr::get(op->getContext(), newPath);
    globalRefOp.setNamepathAttr(newPathAttr);

    // Ensure the part instance will have this GlobalRefAttr.
    // global refs if not.
    newGlobalRefs.insert(globalRef);
  }
}

void PartitionPass::bubbleUp(MSFTModuleOp mod, Block *partBlock) {
  auto *ctxt = mod.getContext();
  FunctionType origType = mod.getFunctionType();
  std::string nameBuffer;

  //*************
  //   Figure out all the new ports 'mod' is going to need. The outputs need to
  //   know where they're being driven from, which'll be some of the outputs of
  //   'ops'. Also determine which of the existing ports are no longer used.
  //
  //   Don't do any mutation here, just assemble bookkeeping info.

  // The new input ports for operands not defined in 'partBlock'.
  SmallVector<std::pair<StringAttr, Type>, 64> newInputs;
  // Map the operand value to new input port.
  DenseMap<Value, size_t> oldValueNewResultNum;

  // The new output ports.
  SmallVector<std::pair<StringAttr, Value>, 64> newOutputs;
  // Store the original result value in new port order. Used later on to remap
  // the moved operations to the new block arguments.
  SmallVector<Value, 64> newInputOldValue;

  for (Operation &op : *partBlock) {
    StringRef opName = ::getOpName(&op);
    if (opName.empty())
      opName = op.getName().getIdentifier().getValue();

    // Tagged operation might need new inputs ports to drive its consumers.
    for (OpResult res : op.getOpResults()) {
      // If all the operations will get moved, no new port is necessary.
      if (llvm::all_of(res.getUsers(), [partBlock](Operation *op) {
            return op->getBlock() == partBlock || isa<OutputOp>(op);
          }))
        continue;

      // Create a new inpurt port.
      StringRef name = getResultName(res, topLevelSyms, nameBuffer);
      newInputs.push_back(std::make_pair(
          StringAttr::get(ctxt, opName + (name.empty() ? "" : "." + name)),
          res.getType()));
      newInputOldValue.push_back(res);
    }

    // Tagged operations may need new output ports to drive their operands.
    for (OpOperand &oper : op.getOpOperands()) {
      Value operVal = oper.get();

      // If the value was coming from outside the module, unnecessary.
      if (auto operArg = operVal.dyn_cast<BlockArgument>())
        continue;

      Operation *defOp = operVal.getDefiningOp();
      assert(defOp && "Value must be operation if not block arg");
      // New port unnecessary if source will be moved or there's already a port
      // for that value.
      if (defOp->getBlock() == partBlock || oldValueNewResultNum.count(operVal))
        continue;

      // Create a new output port.
      oldValueNewResultNum[oper.get()] = newOutputs.size();
      StringRef name = getOperandName(oper, topLevelSyms, nameBuffer);
      newOutputs.push_back(std::make_pair(
          StringAttr::get(ctxt, opName + (name.empty() ? "" : "." + name)),
          operVal));
    }
  }

  // Figure out which of the original output ports can be removed.
  llvm::BitVector outputsToRemove(origType.getNumResults() + newOutputs.size());
  DenseMap<size_t, Value> oldResultOldValues;
  Operation *term = mod.getBodyBlock()->getTerminator();
  assert(term && "Invalid IR");
  for (auto outputValIdx : llvm::enumerate(term->getOperands())) {
    Operation *defOp = outputValIdx.value().getDefiningOp();
    if (!defOp || defOp->getBlock() != partBlock)
      continue;
    outputsToRemove.set(outputValIdx.index());
    oldResultOldValues[outputValIdx.index()] = outputValIdx.value();
  }

  // Figure out which of the original input ports will no longer be used and can
  // be removed.
  llvm::BitVector inputsToRemove(origType.getNumInputs() + newInputs.size());
  for (auto blockArg : mod.getBodyBlock()->getArguments()) {
    if (llvm::all_of(blockArg.getUsers(), [&](Operation *op) {
          return op->getBlock() == partBlock;
        }))
      inputsToRemove.set(blockArg.getArgNumber());
  }

  //*************
  //   Add the new ports and re-wire the operands using the new ports. The
  //   `addPorts` method handles adding the correct values to the terminator op.
  SmallVector<BlockArgument> newBlockArgs = mod.addPorts(newInputs, newOutputs);
  for (size_t inputNum = 0, e = newBlockArgs.size(); inputNum < e; ++inputNum)
    for (OpOperand &use : newInputOldValue[inputNum].getUses())
      if (use.getOwner()->getBlock() != partBlock)
        use.set(newBlockArgs[inputNum]);

  //*************
  //   For all of the instantiation sites (for 'mod'):
  //     - Create a new instance with the correct result types.
  //     - Clone in 'ops'.
  //     - Fix up the new operations' operands.
  auto cloneOpsGetOperands = [&](InstanceOp newInst, InstanceOp oldInst,
                                 SmallVectorImpl<Value> &newOperands) {
    OpBuilder b(newInst);
    IRMapping map;

    // Add all of 'mod''s block args to the map in case one of the tagged ops
    // was driven by a block arg. Map to the oldInst operand Value.
    unsigned oldInstNumInputs = oldInst.getNumOperands();
    for (BlockArgument arg : mod.getBodyBlock()->getArguments())
      if (arg.getArgNumber() < oldInstNumInputs)
        map.map(arg, oldInst.getOperand(arg.getArgNumber()));

    // Add all the old values which got moved to output ports to the map.
    size_t origNumResults = origType.getNumResults();
    for (auto valueResultNum : oldValueNewResultNum)
      map.map(valueResultNum.first,
              newInst->getResult(origNumResults + valueResultNum.second));

    // Clone the ops, rename appropriately, and update the global refs.
    llvm::SmallVector<Operation *, 32> newOps;
    llvm::DenseSet<hw::GlobalRefAttr> movedRefs;
    for (Operation &op : *partBlock) {
      Operation *newOp = b.insert(op.clone(map));
      newOps.push_back(newOp);
      setEntityName(newOp, oldInst.getInstanceName() + "." + ::getOpName(&op));
      auto *oldInstMod = oldInst.getReferencedModule();
      assert(oldInstMod);
      auto oldModName = oldInstMod->getAttrOfType<StringAttr>("sym_name");
      bubbleUpGlobalRefs(newOp, oldModName, oldInst.getInstanceNameAttr(),
                         movedRefs);
    }

    // Remove the hoisted global refs from new instance.
    if (ArrayAttr oldInstRefs = oldInst->getAttrOfType<ArrayAttr>(
            hw::GlobalRefAttr::DialectAttrName)) {
      llvm::SmallVector<Attribute> newInstRefs;
      for (Attribute oldRef : oldInstRefs.getValue()) {
        if (hw::GlobalRefAttr ref = oldRef.dyn_cast<hw::GlobalRefAttr>())
          if (movedRefs.contains(ref))
            continue;
        newInstRefs.push_back(oldRef);
      }
      if (newInstRefs.empty())
        newInst->removeAttr(hw::GlobalRefAttr::DialectAttrName);
      else
        newInst->setAttr(hw::GlobalRefAttr::DialectAttrName,
                         ArrayAttr::get(ctxt, newInstRefs));
    }

    // Fix up operands of cloned ops (backedges didn't exist in the map so they
    // didn't get mapped during the initial clone).
    for (Operation *newOp : newOps)
      for (OpOperand &oper : newOp->getOpOperands())
        oper.set(map.lookupOrDefault(oper.get()));

    // Since we're not removing any ports, start with the old operands.
    newOperands.append(oldInst.getOperands().begin(),
                       oldInst.getOperands().end());
    // Gather new operands for the new instance.
    for (Value oldValue : newInputOldValue)
      newOperands.push_back(map.lookup(oldValue));

    // Fix up existing ops which used the old instance's results.
    for (auto oldResultOldValue : oldResultOldValues)
      oldInst.getResult(oldResultOldValue.first)
          .replaceAllUsesWith(map.lookup(oldResultOldValue.second));
  };
  updateInstances(mod, makeSequentialRange(origType.getNumResults()),
                  cloneOpsGetOperands);

  //*************
  //   Lastly, remove the unnecessary ports. Doing this as a separate mutation
  //   makes the previous steps simpler without any practical degradation.
  SmallVector<unsigned> resValues =
      mod.removePorts(inputsToRemove, outputsToRemove);
  updateInstances(mod, resValues,
                  [&](InstanceOp newInst, InstanceOp oldInst,
                      SmallVectorImpl<Value> &newOperands) {
                    for (auto oldOperand :
                         llvm::enumerate(oldInst->getOperands()))
                      if (!inputsToRemove.test(oldOperand.index()))
                        newOperands.push_back(oldOperand.value());
                  });
}

MSFTModuleOp PartitionPass::partition(DesignPartitionOp partOp,
                                      Block *partBlock) {

  auto *ctxt = partOp.getContext();
  auto loc = partOp.getLoc();
  std::string nameBuffer;

  //*************
  //   Determine the partition module's interface. Keep some bookkeeping around.
  SmallVector<hw::PortInfo> inputPorts;
  SmallVector<hw::PortInfo> outputPorts;
  DenseMap<Value, size_t> newInputMap;
  SmallVector<Value, 32> instInputs;
  SmallVector<Value, 32> newOutputs;

  for (Operation &op : *partBlock) {
    StringRef opName = ::getOpName(&op);
    if (opName.empty())
      opName = op.getName().getIdentifier().getValue();

    for (OpOperand &oper : op.getOpOperands()) {
      Value v = oper.get();
      // Don't need a new input if we're consuming a value in the same block.
      if (v.getParentBlock() == partBlock)
        continue;
      auto existingF = newInputMap.find(v);
      if (existingF == newInputMap.end()) {
        // If there's not an existing input, create one.
        auto arg = partBlock->addArgument(v.getType(), loc);
        oper.set(arg);

        newInputMap[v] = inputPorts.size();
        StringRef portName = getValueName(v, topLevelSyms, nameBuffer);

        instInputs.push_back(v);
        inputPorts.push_back(hw::PortInfo{
            {/*name*/ StringAttr::get(
                 ctxt, opName + (portName.empty() ? "" : "." + portName)),
             /*type*/ v.getType(),
             /*direction*/ hw::ModulePort::Direction::Input},
            /*argNum*/ inputPorts.size(),
            /*sym*/ {},
            /*attr*/ {},
            /*location*/ loc});
      } else {
        // There's already an existing port. Just set it.
        oper.set(partBlock->getArgument(existingF->second));
      }
    }

    for (OpResult res : op.getResults()) {
      // If all the consumers of this result are in the same partition, we don't
      // need a new output port.
      if (llvm::all_of(res.getUsers(), [partBlock](Operation *op) {
            return op->getBlock() == partBlock;
          }))
        continue;

      // If not, add one.
      newOutputs.push_back(res);
      StringRef portName = getResultName(res, topLevelSyms, nameBuffer);
      outputPorts.push_back(hw::PortInfo{
          {/*name*/ StringAttr::get(
               ctxt, opName + (portName.empty() ? "" : "." + portName)),
           /*type*/ res.getType(),
           /*direction*/ hw::ModulePort::Direction::Output},
          /*argNum*/ outputPorts.size()});
    }
  }

  //*************
  //   Construct the partition module and replace the design partition op.

  // Build the module.
  hw::ModulePortInfo modPortInfo(inputPorts, outputPorts);
  auto partMod =
      OpBuilder(partOp->getParentOfType<MSFTModuleOp>())
          .create<MSFTModuleOp>(loc, partOp.getVerilogNameAttr(), modPortInfo,
                                ArrayRef<NamedAttribute>{});
  partBlock->moveBefore(partMod.getBodyBlock());
  partMod.getBlocks().back().erase();

  OpBuilder::atBlockEnd(partBlock).create<OutputOp>(partOp.getLoc(),
                                                    newOutputs);

  // Replace partOp with an instantion of the partition.
  SmallVector<Type> instRetTypes(
      llvm::map_range(newOutputs, [](Value v) { return v.getType(); }));
  auto partInst = OpBuilder(partOp).create<InstanceOp>(
      loc, instRetTypes, partOp.getNameAttr(), FlatSymbolRefAttr::get(partMod),
      instInputs);
  moduleInstantiations[partMod].push_back(partInst);

  // And set the outputs properly.
  for (size_t outputNum = 0, e = newOutputs.size(); outputNum < e; ++outputNum)
    for (OpOperand &oper :
         llvm::make_early_inc_range(newOutputs[outputNum].getUses()))
      if (oper.getOwner()->getBlock() != partBlock)
        oper.set(partInst.getResult(outputNum));

  // Push down any global refs to include the partition. Update the
  // partition to include the new set of global refs, and set its inner_sym.
  llvm::SetVector<Attribute> newGlobalRefs;
  for (Operation &op : *partBlock)
    pushDownGlobalRefs(&op, partOp, newGlobalRefs);
  SmallVector<Attribute> newGlobalRefVec(newGlobalRefs.begin(),
                                         newGlobalRefs.end());
  auto newRefsAttr = ArrayAttr::get(partInst->getContext(), newGlobalRefVec);
  partInst->setAttr(hw::GlobalRefAttr::DialectAttrName, newRefsAttr);

  return partMod;
}

std::unique_ptr<Pass> circt::msft::createPartitionPass() {
  return std::make_unique<PartitionPass>();
}
