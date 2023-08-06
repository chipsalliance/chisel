//===- FIRRTLReductions.cpp - Reduction patterns for the FIRRTL dialect ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLReductions.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Reduce/ReductionUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-reductions"

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

namespace detail {
/// A utility doing lazy construction of `SymbolTable`s and `SymbolUserMap`s,
/// which is handy for reductions that need to look up a lot of symbols.
struct SymbolCache {
  SymbolTable &getSymbolTable(Operation *op) {
    return tables.getSymbolTable(op);
  }
  SymbolTable &getNearestSymbolTable(Operation *op) {
    return getSymbolTable(SymbolTable::getNearestSymbolTable(op));
  }

  SymbolUserMap &getSymbolUserMap(Operation *op) {
    auto it = userMaps.find(op);
    if (it != userMaps.end())
      return it->second;
    return userMaps.insert({op, SymbolUserMap(tables, op)}).first->second;
  }
  SymbolUserMap &getNearestSymbolUserMap(Operation *op) {
    return getSymbolUserMap(SymbolTable::getNearestSymbolTable(op));
  }

  void clear() {
    tables = SymbolTableCollection();
    userMaps.clear();
  }

private:
  SymbolTableCollection tables;
  SmallDenseMap<Operation *, SymbolUserMap, 2> userMaps;
};
} // namespace detail

/// Utility to easily get the instantiated firrtl::FModuleOp or an empty
/// optional in case another type of module is instantiated.
static std::optional<firrtl::FModuleOp>
findInstantiatedModule(firrtl::InstanceOp instOp,
                       ::detail::SymbolCache &symbols) {
  auto *tableOp = SymbolTable::getNearestSymbolTable(instOp);
  auto moduleOp = dyn_cast<firrtl::FModuleOp>(
      instOp.getReferencedModule(symbols.getSymbolTable(tableOp))
          .getOperation());
  return moduleOp ? std::optional(moduleOp) : std::nullopt;
}

/// Utility to track the transitive size of modules.
struct ModuleSizeCache {
  void clear() { moduleSizes.clear(); }

  uint64_t getModuleSize(Operation *module, ::detail::SymbolCache &symbols) {
    if (auto it = moduleSizes.find(module); it != moduleSizes.end())
      return it->second;
    uint64_t size = 1;
    module->walk([&](Operation *op) {
      size += 1;
      if (auto instOp = dyn_cast<firrtl::InstanceOp>(op))
        if (auto instModule = findInstantiatedModule(instOp, symbols))
          size += getModuleSize(*instModule, symbols);
    });
    moduleSizes.insert({module, size});
    return size;
  }

private:
  llvm::DenseMap<Operation *, uint64_t> moduleSizes;
};

/// Check that all connections to a value are invalids.
static bool onlyInvalidated(Value arg) {
  return llvm::all_of(arg.getUses(), [](OpOperand &use) {
    auto *op = use.getOwner();
    if (!isa<firrtl::ConnectOp, firrtl::StrictConnectOp>(op))
      return false;
    if (use.getOperandNumber() != 0)
      return false;
    if (!op->getOperand(1).getDefiningOp<firrtl::InvalidValueOp>())
      return false;
    return true;
  });
}

/// A tracker for track NLAs affected by a reduction. Performs the necessary
/// cleanup steps in order to maintain IR validity after the reduction has
/// applied. For example, removing an instance that forms part of an NLA path
/// requires that NLA to be removed as well.
struct NLARemover {
  /// Clear the set of marked NLAs. Call this before attempting a reduction.
  void clear() { nlasToRemove.clear(); }

  /// Remove all marked annotations. Call this after applying a reduction in
  /// order to validate the IR.
  void remove(mlir::ModuleOp module) {
    unsigned numRemoved = 0;
    (void)numRemoved;
    for (Operation &rootOp : *module.getBody()) {
      if (!isa<firrtl::CircuitOp>(&rootOp))
        continue;
      SymbolTable symbolTable(&rootOp);
      for (auto sym : nlasToRemove) {
        if (auto *op = symbolTable.lookup(sym)) {
          ++numRemoved;
          op->erase();
        }
      }
    }
    LLVM_DEBUG({
      unsigned numLost = nlasToRemove.size() - numRemoved;
      if (numRemoved > 0 || numLost > 0) {
        llvm::dbgs() << "Removed " << numRemoved << " NLAs";
        if (numLost > 0)
          llvm::dbgs() << " (" << numLost << " no longer there)";
        llvm::dbgs() << "\n";
      }
    });
  }

  /// Mark all NLAs referenced in the given annotation as to be removed. This
  /// can be an entire array or dictionary of annotations, and the function will
  /// descend into child annotations appropriately.
  void markNLAsInAnnotation(Attribute anno) {
    if (auto dict = dyn_cast<DictionaryAttr>(anno)) {
      if (auto field = dict.getAs<FlatSymbolRefAttr>("circt.nonlocal"))
        nlasToRemove.insert(field.getAttr());
      for (auto namedAttr : dict)
        markNLAsInAnnotation(namedAttr.getValue());
    } else if (auto array = dyn_cast<ArrayAttr>(anno)) {
      for (auto attr : array)
        markNLAsInAnnotation(attr);
    }
  }

  /// Mark all NLAs referenced in an operation. Also traverses all nested
  /// operations. Call this before removing an operation, to mark any associated
  /// NLAs as to be removed as well.
  void markNLAsInOperation(Operation *op) {
    op->walk([&](Operation *op) {
      if (auto annos = op->getAttrOfType<ArrayAttr>("annotations"))
        markNLAsInAnnotation(annos);
    });
  }

  /// The set of NLAs to remove, identified by their symbol.
  llvm::DenseSet<StringAttr> nlasToRemove;
};

//===----------------------------------------------------------------------===//
// Reduction patterns
//===----------------------------------------------------------------------===//

/// A sample reduction pattern that maps `firrtl.module` to `firrtl.extmodule`.
struct FIRRTLModuleExternalizer : public OpReduction<firrtl::FModuleOp> {
  void beforeReduction(mlir::ModuleOp op) override {
    nlaRemover.clear();
    symbols.clear();
    moduleSizes.clear();
  }
  void afterReduction(mlir::ModuleOp op) override { nlaRemover.remove(op); }

  uint64_t match(firrtl::FModuleOp module) override {
    return moduleSizes.getModuleSize(module, symbols);
  }

  LogicalResult rewrite(firrtl::FModuleOp module) override {
    nlaRemover.markNLAsInOperation(module);
    OpBuilder builder(module);
    builder.create<firrtl::FExtModuleOp>(
        module->getLoc(),
        module->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()),
        module.getConventionAttr(), module.getPorts(), StringRef(),
        module.getAnnotationsAttr());
    module->erase();
    return success();
  }

  std::string getName() const override { return "firrtl-module-externalizer"; }

  ::detail::SymbolCache symbols;
  NLARemover nlaRemover;
  ModuleSizeCache moduleSizes;
};

/// Invalidate all the leaf fields of a value with a given flippedness by
/// connecting an invalid value to them. This is useful for ensuring that all
/// output ports of an instance or memory (including those nested in bundles)
/// are properly invalidated.
static void invalidateOutputs(ImplicitLocOpBuilder &builder, Value value,
                              SmallDenseMap<Type, Value, 8> &invalidCache,
                              bool flip = false) {
  auto type = value.getType().dyn_cast<firrtl::FIRRTLType>();
  if (!type)
    return;

  // Descend into bundles by creating subfield ops.
  if (auto bundleType = type.dyn_cast<firrtl::BundleType>()) {
    for (auto element : llvm::enumerate(bundleType.getElements())) {
      auto subfield =
          builder.createOrFold<firrtl::SubfieldOp>(value, element.index());
      invalidateOutputs(builder, subfield, invalidCache,
                        flip ^ element.value().isFlip);
      if (subfield.use_empty())
        subfield.getDefiningOp()->erase();
    }
    return;
  }

  // Descend into vectors by creating subindex ops.
  if (auto vectorType = type.dyn_cast<firrtl::FVectorType>()) {
    for (unsigned i = 0, e = vectorType.getNumElements(); i != e; ++i) {
      auto subindex = builder.createOrFold<firrtl::SubindexOp>(value, i);
      invalidateOutputs(builder, subindex, invalidCache, flip);
      if (subindex.use_empty())
        subindex.getDefiningOp()->erase();
    }
    return;
  }

  // Only drive outputs.
  if (flip)
    return;
  Value invalid = invalidCache.lookup(type);
  if (!invalid) {
    invalid = builder.create<firrtl::InvalidValueOp>(type);
    invalidCache.insert({type, invalid});
  }
  builder.create<firrtl::ConnectOp>(value, invalid);
}

/// Connect a value to every leave of a destination value.
static void connectToLeafs(ImplicitLocOpBuilder &builder, Value dest,
                           Value value) {
  auto type = dest.getType().dyn_cast<firrtl::FIRRTLBaseType>();
  if (!type)
    return;
  if (auto bundleType = type.dyn_cast<firrtl::BundleType>()) {
    for (auto element : llvm::enumerate(bundleType.getElements()))
      connectToLeafs(builder,
                     builder.create<firrtl::SubfieldOp>(dest, element.index()),
                     value);
    return;
  }
  if (auto vectorType = type.dyn_cast<firrtl::FVectorType>()) {
    for (unsigned i = 0, e = vectorType.getNumElements(); i != e; ++i)
      connectToLeafs(builder, builder.create<firrtl::SubindexOp>(dest, i),
                     value);
    return;
  }
  auto valueType = value.getType().dyn_cast<firrtl::FIRRTLBaseType>();
  if (!valueType)
    return;
  auto destWidth = type.getBitWidthOrSentinel();
  auto valueWidth = valueType ? valueType.getBitWidthOrSentinel() : -1;
  if (destWidth >= 0 && valueWidth >= 0 && destWidth < valueWidth)
    value = builder.create<firrtl::HeadPrimOp>(value, destWidth);
  if (!isa<firrtl::UIntType>(type)) {
    if (isa<firrtl::SIntType>(type))
      value = builder.create<firrtl::AsSIntPrimOp>(value);
    else
      return;
  }
  builder.create<firrtl::ConnectOp>(dest, value);
}

/// Reduce all leaf fields of a value through an XOR tree.
static void reduceXor(ImplicitLocOpBuilder &builder, Value &into, Value value) {
  auto type = value.getType().dyn_cast<firrtl::FIRRTLType>();
  if (!type)
    return;
  if (auto bundleType = type.dyn_cast<firrtl::BundleType>()) {
    for (auto element : llvm::enumerate(bundleType.getElements()))
      reduceXor(
          builder, into,
          builder.createOrFold<firrtl::SubfieldOp>(value, element.index()));
    return;
  }
  if (auto vectorType = type.dyn_cast<firrtl::FVectorType>()) {
    for (unsigned i = 0, e = vectorType.getNumElements(); i != e; ++i)
      reduceXor(builder, into,
                builder.createOrFold<firrtl::SubindexOp>(value, i));
    return;
  }
  if (!isa<firrtl::UIntType>(type)) {
    if (isa<firrtl::SIntType>(type))
      value = builder.create<firrtl::AsUIntPrimOp>(value);
    else
      return;
  }
  into = into ? builder.createOrFold<firrtl::XorPrimOp>(into, value) : value;
}

/// A sample reduction pattern that maps `firrtl.instance` to a set of
/// invalidated wires. This often shortcuts a long iterative process of connect
/// invalidation, module externalization, and wire stripping
struct InstanceStubber : public OpReduction<firrtl::InstanceOp> {
  void beforeReduction(mlir::ModuleOp op) override {
    erasedInsts.clear();
    erasedModules.clear();
    symbols.clear();
    nlaRemover.clear();
    moduleSizes.clear();
  }
  void afterReduction(mlir::ModuleOp op) override {
    // Look into deleted modules to find additional instances that are no longer
    // instantiated anywhere.
    SmallVector<Operation *> worklist;
    auto deadInsts = erasedInsts;
    for (auto *op : erasedModules)
      worklist.push_back(op);
    while (!worklist.empty()) {
      auto *op = worklist.pop_back_val();
      auto *tableOp = SymbolTable::getNearestSymbolTable(op);
      op->walk([&](firrtl::InstanceOp instOp) {
        auto moduleOp =
            instOp.getReferencedModule(symbols.getSymbolTable(tableOp));
        deadInsts.insert(instOp);
        if (llvm::all_of(
                symbols.getSymbolUserMap(tableOp).getUsers(moduleOp),
                [&](Operation *user) { return deadInsts.contains(user); })) {
          LLVM_DEBUG(llvm::dbgs() << "- Removing transitively unused module `"
                                  << moduleOp.getModuleName() << "`\n");
          erasedModules.insert(moduleOp);
          worklist.push_back(moduleOp);
        }
      });
    }

    for (auto *op : erasedInsts)
      op->erase();
    for (auto *op : erasedModules)
      op->erase();
    nlaRemover.remove(op);
  }

  uint64_t match(firrtl::InstanceOp instOp) override {
    if (auto fmoduleOp = findInstantiatedModule(instOp, symbols))
      return moduleSizes.getModuleSize(*fmoduleOp, symbols);
    return 0;
  }

  LogicalResult rewrite(firrtl::InstanceOp instOp) override {
    LLVM_DEBUG(llvm::dbgs()
               << "Stubbing instance `" << instOp.getName() << "`\n");
    ImplicitLocOpBuilder builder(instOp.getLoc(), instOp);
    SmallDenseMap<Type, Value, 8> invalidCache;
    for (unsigned i = 0, e = instOp.getNumResults(); i != e; ++i) {
      auto result = instOp.getResult(i);
      auto name = builder.getStringAttr(Twine(instOp.getName()) + "_" +
                                        instOp.getPortNameStr(i));
      auto wire =
          builder
              .create<firrtl::WireOp>(result.getType(), name,
                                      firrtl::NameKindEnum::DroppableName,
                                      instOp.getPortAnnotation(i), StringAttr{})
              .getResult();
      invalidateOutputs(builder, wire, invalidCache,
                        instOp.getPortDirection(i) == firrtl::Direction::In);
      result.replaceAllUsesWith(wire);
    }
    auto *tableOp = SymbolTable::getNearestSymbolTable(instOp);
    auto moduleOp = instOp.getReferencedModule(symbols.getSymbolTable(tableOp));
    nlaRemover.markNLAsInOperation(instOp);
    erasedInsts.insert(instOp);
    if (llvm::all_of(
            symbols.getSymbolUserMap(tableOp).getUsers(moduleOp),
            [&](Operation *user) { return erasedInsts.contains(user); })) {
      LLVM_DEBUG(llvm::dbgs() << "- Removing now unused module `"
                              << moduleOp.getModuleName() << "`\n");
      erasedModules.insert(moduleOp);
    }
    return success();
  }

  std::string getName() const override { return "instance-stubber"; }
  bool acceptSizeIncrease() const override { return true; }

  ::detail::SymbolCache symbols;
  NLARemover nlaRemover;
  llvm::DenseSet<Operation *> erasedInsts;
  llvm::DenseSet<Operation *> erasedModules;
  ModuleSizeCache moduleSizes;
};

/// A sample reduction pattern that maps `firrtl.mem` to a set of invalidated
/// wires.
struct MemoryStubber : public OpReduction<firrtl::MemOp> {
  void beforeReduction(mlir::ModuleOp op) override { nlaRemover.clear(); }
  void afterReduction(mlir::ModuleOp op) override { nlaRemover.remove(op); }
  LogicalResult rewrite(firrtl::MemOp memOp) override {
    LLVM_DEBUG(llvm::dbgs() << "Stubbing memory `" << memOp.getName() << "`\n");
    ImplicitLocOpBuilder builder(memOp.getLoc(), memOp);
    SmallDenseMap<Type, Value, 8> invalidCache;
    Value xorInputs;
    SmallVector<Value> outputs;
    for (unsigned i = 0, e = memOp.getNumResults(); i != e; ++i) {
      auto result = memOp.getResult(i);
      auto name = builder.getStringAttr(Twine(memOp.getName()) + "_" +
                                        memOp.getPortNameStr(i));
      auto wire =
          builder
              .create<firrtl::WireOp>(result.getType(), name,
                                      firrtl::NameKindEnum::DroppableName,
                                      memOp.getPortAnnotation(i), StringAttr{})
              .getResult();
      invalidateOutputs(builder, wire, invalidCache, true);
      result.replaceAllUsesWith(wire);

      // Isolate the input and output data fields of the port.
      Value input, output;
      switch (memOp.getPortKind(i)) {
      case firrtl::MemOp::PortKind::Read:
        output = builder.createOrFold<firrtl::SubfieldOp>(wire, 3);
        break;
      case firrtl::MemOp::PortKind::Write:
        input = builder.createOrFold<firrtl::SubfieldOp>(wire, 3);
        break;
      case firrtl::MemOp::PortKind::ReadWrite:
        input = builder.createOrFold<firrtl::SubfieldOp>(wire, 5);
        output = builder.createOrFold<firrtl::SubfieldOp>(wire, 3);
        break;
      case firrtl::MemOp::PortKind::Debug:
        output = wire;
        break;
      }

      if (!isa<firrtl::RefType>(result.getType())) {
        // Reduce all input ports to a single one through an XOR tree.
        unsigned numFields =
            wire.getType().cast<firrtl::BundleType>().getNumElements();
        for (unsigned i = 0; i != numFields; ++i) {
          if (i != 2 && i != 3 && i != 5)
            reduceXor(builder, xorInputs,
                      builder.createOrFold<firrtl::SubfieldOp>(wire, i));
        }
        if (input)
          reduceXor(builder, xorInputs, input);
      }

      // Track the output port to hook it up to the XORd input later.
      if (output)
        outputs.push_back(output);
    }

    // Hook up the outputs.
    for (auto output : outputs)
      connectToLeafs(builder, output, xorInputs);

    nlaRemover.markNLAsInOperation(memOp);
    memOp->erase();
    return success();
  }
  std::string getName() const override { return "memory-stubber"; }
  bool acceptSizeIncrease() const override { return true; }
  NLARemover nlaRemover;
};

/// Check whether an operation interacts with flows in any way, which can make
/// replacement and operand forwarding harder in some cases.
static bool isFlowSensitiveOp(Operation *op) {
  return isa<firrtl::WireOp, firrtl::RegOp, firrtl::RegResetOp,
             firrtl::InstanceOp, firrtl::SubfieldOp, firrtl::SubindexOp,
             firrtl::SubaccessOp>(op);
}

/// A sample reduction pattern that replaces all uses of an operation with one
/// of its operands. This can help pruning large parts of the expression tree
/// rapidly.
template <unsigned OpNum>
struct FIRRTLOperandForwarder : public Reduction {
  uint64_t match(Operation *op) override {
    if (op->getNumResults() != 1 || OpNum >= op->getNumOperands())
      return 0;
    if (isFlowSensitiveOp(op))
      return 0;
    auto resultTy =
        op->getResult(0).getType().dyn_cast<firrtl::FIRRTLBaseType>();
    auto opTy =
        op->getOperand(OpNum).getType().dyn_cast<firrtl::FIRRTLBaseType>();
    return resultTy && opTy &&
           resultTy.getWidthlessType() == opTy.getWidthlessType() &&
           (resultTy.getBitWidthOrSentinel() == -1) ==
               (opTy.getBitWidthOrSentinel() == -1) &&
           isa<firrtl::UIntType, firrtl::SIntType>(resultTy);
  }
  LogicalResult rewrite(Operation *op) override {
    assert(match(op));
    ImplicitLocOpBuilder builder(op->getLoc(), op);
    auto result = op->getResult(0);
    auto operand = op->getOperand(OpNum);
    auto resultTy = result.getType().cast<firrtl::FIRRTLBaseType>();
    auto operandTy = operand.getType().cast<firrtl::FIRRTLBaseType>();
    auto resultWidth = resultTy.getBitWidthOrSentinel();
    auto operandWidth = operandTy.getBitWidthOrSentinel();
    Value newOp;
    if (resultWidth < operandWidth)
      newOp =
          builder.createOrFold<firrtl::BitsPrimOp>(operand, resultWidth - 1, 0);
    else if (resultWidth > operandWidth)
      newOp = builder.createOrFold<firrtl::PadPrimOp>(operand, resultWidth);
    else
      newOp = operand;
    LLVM_DEBUG(llvm::dbgs() << "Forwarding " << newOp << " in " << *op << "\n");
    result.replaceAllUsesWith(newOp);
    reduce::pruneUnusedOps(op, *this);
    return success();
  }
  std::string getName() const override {
    return ("firrtl-operand" + Twine(OpNum) + "-forwarder").str();
  }
};

/// A sample reduction pattern that replaces FIRRTL operations with a constant
/// zero of their type.
struct FIRRTLConstantifier : public Reduction {
  uint64_t match(Operation *op) override {
    if (op->getNumResults() != 1 || op->getNumOperands() == 0)
      return 0;
    if (isFlowSensitiveOp(op))
      return 0;
    auto type = op->getResult(0).getType().dyn_cast<firrtl::FIRRTLBaseType>();
    return isa_and_nonnull<firrtl::UIntType, firrtl::SIntType>(type);
  }
  LogicalResult rewrite(Operation *op) override {
    assert(match(op));
    OpBuilder builder(op);
    auto type = op->getResult(0).getType().cast<firrtl::FIRRTLBaseType>();
    auto width = type.getBitWidthOrSentinel();
    if (width == -1)
      width = 64;
    auto newOp = builder.create<firrtl::ConstantOp>(
        op->getLoc(), type, APSInt(width, isa<firrtl::UIntType>(type)));
    op->replaceAllUsesWith(newOp);
    reduce::pruneUnusedOps(op, *this);
    return success();
  }
  std::string getName() const override { return "firrtl-constantifier"; }
};

/// A sample reduction pattern that replaces the right-hand-side of
/// `firrtl.connect` and `firrtl.strictconnect` operations with a
/// `firrtl.invalidvalue`. This removes uses from the fanin cone to these
/// connects and creates opportunities for reduction in DCE/CSE.
struct ConnectInvalidator : public Reduction {
  uint64_t match(Operation *op) override {
    if (!isa<firrtl::ConnectOp, firrtl::StrictConnectOp>(op))
      return 0;
    auto type = op->getOperand(1).getType().dyn_cast<firrtl::FIRRTLBaseType>();
    return type && type.isPassive() &&
           !op->getOperand(1).getDefiningOp<firrtl::InvalidValueOp>();
  }
  LogicalResult rewrite(Operation *op) override {
    assert(match(op));
    auto rhs = op->getOperand(1);
    OpBuilder builder(op);
    auto invOp =
        builder.create<firrtl::InvalidValueOp>(rhs.getLoc(), rhs.getType());
    auto *rhsOp = rhs.getDefiningOp();
    op->setOperand(1, invOp);
    if (rhsOp)
      reduce::pruneUnusedOps(rhsOp, *this);
    return success();
  }
  std::string getName() const override { return "connect-invalidator"; }
  bool acceptSizeIncrease() const override { return true; }
};

/// A sample reduction pattern that removes FIRRTL annotations from ports and
/// operations.
struct AnnotationRemover : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override { nlaRemover.clear(); }
  void afterReduction(mlir::ModuleOp op) override { nlaRemover.remove(op); }
  uint64_t match(Operation *op) override {
    return op->hasAttr("annotations") || op->hasAttr("portAnnotations");
  }
  LogicalResult rewrite(Operation *op) override {
    auto emptyArray = ArrayAttr::get(op->getContext(), {});
    if (auto annos = op->getAttr("annotations")) {
      nlaRemover.markNLAsInAnnotation(annos);
      op->setAttr("annotations", emptyArray);
    }
    if (auto annos = op->getAttr("portAnnotations")) {
      nlaRemover.markNLAsInAnnotation(annos);
      auto attr = emptyArray;
      if (isa<firrtl::InstanceOp>(op))
        attr = ArrayAttr::get(
            op->getContext(),
            SmallVector<Attribute>(op->getNumResults(), emptyArray));
      op->setAttr("portAnnotations", attr);
    }
    return success();
  }
  std::string getName() const override { return "annotation-remover"; }
  NLARemover nlaRemover;
};

/// A sample reduction pattern that removes ports from the root `firrtl.module`
/// if the port is not used or just invalidated.
struct RootPortPruner : public OpReduction<firrtl::FModuleOp> {
  uint64_t match(firrtl::FModuleOp module) override {
    auto circuit = module->getParentOfType<firrtl::CircuitOp>();
    if (!circuit)
      return 0;
    return circuit.getNameAttr() == module.getNameAttr();
  }
  LogicalResult rewrite(firrtl::FModuleOp module) override {
    assert(match(module));
    size_t numPorts = module.getNumPorts();
    llvm::BitVector dropPorts(numPorts);
    for (unsigned i = 0; i != numPorts; ++i) {
      if (onlyInvalidated(module.getArgument(i))) {
        dropPorts.set(i);
        for (auto *user :
             llvm::make_early_inc_range(module.getArgument(i).getUsers()))
          user->erase();
      }
    }
    module.erasePorts(dropPorts);
    return success();
  }
  std::string getName() const override { return "root-port-pruner"; }
};

/// A sample reduction pattern that replaces instances of `firrtl.extmodule`
/// with wires.
struct ExtmoduleInstanceRemover : public OpReduction<firrtl::InstanceOp> {
  void beforeReduction(mlir::ModuleOp op) override {
    symbols.clear();
    nlaRemover.clear();
  }
  void afterReduction(mlir::ModuleOp op) override { nlaRemover.remove(op); }

  uint64_t match(firrtl::InstanceOp instOp) override {
    return isa<firrtl::FExtModuleOp>(
        instOp.getReferencedModule(symbols.getNearestSymbolTable(instOp)));
  }
  LogicalResult rewrite(firrtl::InstanceOp instOp) override {
    auto portInfo =
        instOp.getReferencedModule(symbols.getNearestSymbolTable(instOp))
            .getPorts();
    ImplicitLocOpBuilder builder(instOp.getLoc(), instOp);
    SmallVector<Value> replacementWires;
    for (firrtl::PortInfo info : portInfo) {
      auto wire =
          builder
              .create<firrtl::WireOp>(
                  info.type,
                  (Twine(instOp.getName()) + "_" + info.getName()).str())
              .getResult();
      if (info.isOutput()) {
        auto inv = builder.create<firrtl::InvalidValueOp>(info.type);
        builder.create<firrtl::ConnectOp>(wire, inv);
      }
      replacementWires.push_back(wire);
    }
    nlaRemover.markNLAsInOperation(instOp);
    instOp.replaceAllUsesWith(std::move(replacementWires));
    instOp->erase();
    return success();
  }
  std::string getName() const override { return "extmodule-instance-remover"; }
  bool acceptSizeIncrease() const override { return true; }

  ::detail::SymbolCache symbols;
  NLARemover nlaRemover;
};

/// A sample reduction pattern that pushes connected values through wires.
struct ConnectForwarder : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override { opsToErase.clear(); }
  void afterReduction(mlir::ModuleOp op) override {
    for (auto *op : opsToErase)
      op->dropAllReferences();
    for (auto *op : opsToErase)
      op->erase();
  }

  uint64_t match(Operation *op) override {
    if (!isa<firrtl::FConnectLike>(op))
      return 0;
    auto dest = op->getOperand(0);
    auto src = op->getOperand(1);
    auto *destOp = dest.getDefiningOp();
    auto *srcOp = src.getDefiningOp();
    if (dest == src)
      return 0;

    // Ensure that the destination is something we should be able to forward
    // through.
    if (!isa_and_nonnull<firrtl::WireOp>(destOp))
      return 0;

    // Ensure that the destination is connected to only once, and all uses of
    // the connection occur after the definition of the source.
    unsigned numConnects = 0;
    for (auto &use : dest.getUses()) {
      auto *op = use.getOwner();
      if (use.getOperandNumber() == 0 && isa<firrtl::FConnectLike>(op)) {
        if (++numConnects > 1)
          return 0;
        continue;
      }
      if (srcOp && !srcOp->isBeforeInBlock(op))
        return 0;
    }

    return 1;
  }

  LogicalResult rewrite(Operation *op) override {
    auto dest = op->getOperand(0);
    dest.replaceAllUsesWith(op->getOperand(1));
    opsToErase.insert(dest.getDefiningOp());
    opsToErase.insert(op);
    return success();
  }

  std::string getName() const override { return "connect-forwarder"; }

  llvm::DenseSet<Operation *> opsToErase;
};

/// A sample reduction pattern that replaces a single-use wire and register with
/// an operand of the source value of the connection.
template <unsigned OpNum>
struct ConnectSourceOperandForwarder : public Reduction {
  uint64_t match(Operation *op) override {
    if (!isa<firrtl::ConnectOp, firrtl::StrictConnectOp>(op))
      return 0;
    auto dest = op->getOperand(0);
    auto *destOp = dest.getDefiningOp();

    // Ensure that the destination is used only once.
    if (!destOp || !destOp->hasOneUse() ||
        !isa<firrtl::WireOp, firrtl::RegOp, firrtl::RegResetOp>(destOp))
      return 0;

    auto *srcOp = op->getOperand(1).getDefiningOp();
    if (!srcOp || OpNum >= srcOp->getNumOperands())
      return 0;

    auto resultTy = dest.getType().dyn_cast<firrtl::FIRRTLBaseType>();
    auto opTy =
        srcOp->getOperand(OpNum).getType().dyn_cast<firrtl::FIRRTLBaseType>();

    return resultTy && opTy &&
           resultTy.getWidthlessType() == opTy.getWidthlessType() &&
           ((resultTy.getBitWidthOrSentinel() == -1) ==
            (opTy.getBitWidthOrSentinel() == -1)) &&
           isa<firrtl::UIntType, firrtl::SIntType>(resultTy);
  }

  LogicalResult rewrite(Operation *op) override {
    auto *destOp = op->getOperand(0).getDefiningOp();
    auto *srcOp = op->getOperand(1).getDefiningOp();
    auto forwardedOperand = srcOp->getOperand(OpNum);
    ImplicitLocOpBuilder builder(destOp->getLoc(), destOp);
    Value newDest;
    if (auto wire = dyn_cast<firrtl::WireOp>(destOp))
      newDest = builder
                    .create<firrtl::WireOp>(forwardedOperand.getType(),
                                            wire.getName())
                    .getResult();
    else {
      auto regName = destOp->getAttrOfType<StringAttr>("name");
      // We can promote the register into a wire but we wouldn't do here because
      // the error might be caused by the register.
      auto clock = destOp->getOperand(0);
      newDest = builder
                    .create<firrtl::RegOp>(forwardedOperand.getType(), clock,
                                           regName ? regName.str() : "")
                    .getResult();
    }

    // Create new connection between a new wire and the forwarded operand.
    builder.setInsertionPointAfter(op);
    if (isa<firrtl::ConnectOp>(op))
      builder.create<firrtl::ConnectOp>(newDest, forwardedOperand);
    else
      builder.create<firrtl::StrictConnectOp>(newDest, forwardedOperand);

    // Remove the old connection and destination. We don't have to replace them
    // because destination has only one use.
    op->erase();
    destOp->erase();
    reduce::pruneUnusedOps(srcOp, *this);

    return success();
  }
  std::string getName() const override {
    return ("connect-source-operand-" + Twine(OpNum) + "-forwarder").str();
  }
};

/// A sample reduction pattern that tries to remove aggregate wires by replacing
/// all subaccesses with new independent wires. This can disentangle large
/// unused wires that are otherwise difficult to collect due to the subaccesses.
struct DetachSubaccesses : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override { opsToErase.clear(); }
  void afterReduction(mlir::ModuleOp op) override {
    for (auto *op : opsToErase)
      op->dropAllReferences();
    for (auto *op : opsToErase)
      op->erase();
  }
  uint64_t match(Operation *op) override {
    // Only applies to wires and registers that are purely used in subaccess
    // operations.
    return isa<firrtl::WireOp, firrtl::RegOp, firrtl::RegResetOp>(op) &&
           llvm::all_of(op->getUses(), [](auto &use) {
             return use.getOperandNumber() == 0 &&
                    isa<firrtl::SubfieldOp, firrtl::SubindexOp,
                        firrtl::SubaccessOp>(use.getOwner());
           });
  }
  LogicalResult rewrite(Operation *op) override {
    assert(match(op));
    OpBuilder builder(op);
    bool isWire = isa<firrtl::WireOp>(op);
    Value invalidClock;
    if (!isWire)
      invalidClock = builder.create<firrtl::InvalidValueOp>(
          op->getLoc(), firrtl::ClockType::get(op->getContext()));
    for (Operation *user : llvm::make_early_inc_range(op->getUsers())) {
      builder.setInsertionPoint(user);
      auto type = user->getResult(0).getType();
      Operation *replOp;
      if (isWire)
        replOp = builder.create<firrtl::WireOp>(user->getLoc(), type);
      else
        replOp =
            builder.create<firrtl::RegOp>(user->getLoc(), type, invalidClock);
      user->replaceAllUsesWith(replOp);
      opsToErase.insert(user);
    }
    opsToErase.insert(op);
    return success();
  }
  std::string getName() const override { return "detach-subaccesses"; }
  llvm::DenseSet<Operation *> opsToErase;
};

/// This reduction removes symbols on node ops. Name preservation creates a lot
/// of nodes ops with symbols to keep name information but it also prevents
/// normal canonicalizations.
struct NodeSymbolRemover : public OpReduction<firrtl::NodeOp> {

  uint64_t match(firrtl::NodeOp nodeOp) override {
    return nodeOp.getInnerSym() &&
           !nodeOp.getInnerSym()->getSymName().getValue().empty();
  }

  LogicalResult rewrite(firrtl::NodeOp nodeOp) override {
    nodeOp.removeInnerSymAttr();
    return success();
  }

  std::string getName() const override { return "node-symbol-remover"; }
};

/// A sample reduction pattern that eagerly inlines instances.
struct EagerInliner : public OpReduction<firrtl::InstanceOp> {
  void beforeReduction(mlir::ModuleOp op) override {
    symbols.clear();
    nlaRemover.clear();
  }
  void afterReduction(mlir::ModuleOp op) override { nlaRemover.remove(op); }

  uint64_t match(firrtl::InstanceOp instOp) override {
    auto *tableOp = SymbolTable::getNearestSymbolTable(instOp);
    auto moduleOp = instOp.getReferencedModule(symbols.getSymbolTable(tableOp));
    if (!isa<firrtl::FModuleOp>(moduleOp.getOperation()))
      return 0;
    return symbols.getSymbolUserMap(tableOp).getUsers(moduleOp).size() == 1;
  }

  LogicalResult rewrite(firrtl::InstanceOp instOp) override {
    LLVM_DEBUG(llvm::dbgs()
               << "Inlining instance `" << instOp.getName() << "`\n");
    SmallVector<Value> argReplacements;
    ImplicitLocOpBuilder builder(instOp.getLoc(), instOp);
    for (unsigned i = 0, e = instOp.getNumResults(); i != e; ++i) {
      auto result = instOp.getResult(i);
      auto name = builder.getStringAttr(Twine(instOp.getName()) + "_" +
                                        instOp.getPortNameStr(i));
      auto wire =
          builder
              .create<firrtl::WireOp>(result.getType(), name,
                                      firrtl::NameKindEnum::DroppableName,
                                      instOp.getPortAnnotation(i), StringAttr{})
              .getResult();
      result.replaceAllUsesWith(wire);
      argReplacements.push_back(wire);
    }
    auto *tableOp = SymbolTable::getNearestSymbolTable(instOp);
    auto moduleOp = cast<firrtl::FModuleOp>(
        instOp.getReferencedModule(symbols.getSymbolTable(tableOp))
            .getOperation());
    for (auto &op : llvm::make_early_inc_range(*moduleOp.getBodyBlock())) {
      op.remove();
      builder.insert(&op);
      for (auto &operand : op.getOpOperands())
        if (auto blockArg = dyn_cast<BlockArgument>(operand.get()))
          operand.set(argReplacements[blockArg.getArgNumber()]);
    }
    nlaRemover.markNLAsInOperation(instOp);
    instOp->erase();
    moduleOp->erase();
    return success();
  }

  std::string getName() const override { return "eager-inliner"; }
  bool acceptSizeIncrease() const override { return true; }

  ::detail::SymbolCache symbols;
  NLARemover nlaRemover;
};

/// Psuedo-reduction that sanitizes the names of things inside modules.  This is
/// not an actual reduction, but often removes extraneous information that has
/// no bearing on the actual reduction (and would likely be removed before
/// sharing the reduction).  This makes the following changes:
///
///   - All wires are renamed to "wire"
///   - All registers are renamed to "reg"
///   - All nodes are renamed to "node"
///   - All memories are renamed to "mem"
///   - All verification messages and labels are dropped
///
struct ModuleInternalNameSanitizer : public Reduction {
  uint64_t match(Operation *op) override {
    // Only match operations with names.
    return isa<firrtl::WireOp, firrtl::RegOp, firrtl::RegResetOp,
               firrtl::NodeOp, firrtl::MemOp, chirrtl::CombMemOp,
               chirrtl::SeqMemOp, firrtl::AssertOp, firrtl::AssumeOp,
               firrtl::CoverOp>(op);
  }
  LogicalResult rewrite(Operation *op) override {
    TypeSwitch<Operation *, void>(op)
        .Case<firrtl::WireOp>([](auto op) { op.setName("wire"); })
        .Case<firrtl::RegOp, firrtl::RegResetOp>(
            [](auto op) { op.setName("reg"); })
        .Case<firrtl::NodeOp>([](auto op) { op.setName("node"); })
        .Case<firrtl::MemOp, chirrtl::CombMemOp, chirrtl::SeqMemOp>(
            [](auto op) { op.setName("mem"); })
        .Case<firrtl::AssertOp, firrtl::AssumeOp, firrtl::CoverOp>([](auto op) {
          op->setAttr("message", StringAttr::get(op.getContext(), ""));
          op->setAttr("name", StringAttr::get(op.getContext(), ""));
        });
    return success();
  }

  std::string getName() const override {
    return "module-internal-name-sanitizer";
  }

  bool acceptSizeIncrease() const override { return true; }

  bool isOneShot() const override { return true; }
};

/// Psuedo-reduction that sanitizes module, instance, and port names.  This
/// makes the following changes:
///
///     - All modules are given metasyntactic names ("Foo", "Bar", etc.)
///     - All instances are renamed to match the new module name
///     - All module ports are renamed in the following way:
///         - All clocks are reanemd to "clk"
///         - All resets are renamed to "rst"
///         - All references are renamed to "ref"
///         - Anything else is renamed to "port"
///
struct ModuleNameSanitizer : OpReduction<firrtl::CircuitOp> {

  const char *names[48] = {
      "Foo",    "Bar",    "Baz",    "Qux",      "Quux",   "Quuux",  "Quuuux",
      "Quz",    "Corge",  "Grault", "Bazola",   "Ztesch", "Thud",   "Grunt",
      "Bletch", "Fum",    "Fred",   "Jim",      "Sheila", "Barney", "Flarp",
      "Zxc",    "Spqr",   "Wombat", "Shme",     "Bongo",  "Spam",   "Eggs",
      "Snork",  "Zot",    "Blarg",  "Wibble",   "Toto",   "Titi",   "Tata",
      "Tutu",   "Pippo",  "Pluto",  "Paperino", "Aap",    "Noot",   "Mies",
      "Oogle",  "Foogle", "Boogle", "Zork",     "Gork",   "Bork"};

  size_t nameIndex = 0;

  const char *getName() {
    if (nameIndex >= 48)
      nameIndex = 0;
    return names[nameIndex++];
  };

  size_t portNameIndex = 0;

  char getPortName() {
    if (portNameIndex >= 26)
      portNameIndex = 0;
    return 'a' + portNameIndex++;
  }

  void beforeReduction(mlir::ModuleOp op) override { nameIndex = 0; }

  LogicalResult rewrite(firrtl::CircuitOp circuitOp) override {

    firrtl::InstanceGraph iGraph(circuitOp);

    auto *circuitName = getName();
    iGraph.getTopLevelModule().setName(circuitName);
    circuitOp.setName(circuitName);

    for (auto *node : iGraph) {
      auto module = node->getModule();

      bool shouldReplacePorts = false;
      SmallVector<Attribute> newNames;
      if (auto fmodule = dyn_cast<firrtl::FModuleOp>(*module)) {
        portNameIndex = 0;
        // TODO: The namespace should be unnecessary. However, some FIRRTL
        // passes expect that port names are unique.
        circt::Namespace ns;
        auto oldPorts = fmodule.getPorts();
        shouldReplacePorts = !oldPorts.empty();
        for (unsigned i = 0, e = fmodule.getNumPorts(); i != e; ++i) {
          auto port = oldPorts[i];
          auto newName = firrtl::FIRRTLTypeSwitch<Type, StringRef>(port.type)
                             .Case<firrtl::ClockType>(
                                 [&](auto a) { return ns.newName("clk"); })
                             .Case<firrtl::ResetType, firrtl::AsyncResetType>(
                                 [&](auto a) { return ns.newName("rst"); })
                             .Case<firrtl::RefType>(
                                 [&](auto a) { return ns.newName("ref"); })
                             .Default([&](auto a) {
                               return ns.newName(Twine(getPortName()));
                             });
          newNames.push_back(StringAttr::get(circuitOp.getContext(), newName));
        }
        fmodule->setAttr("portNames",
                         ArrayAttr::get(fmodule.getContext(), newNames));
      }

      if (module == iGraph.getTopLevelModule())
        continue;
      auto newName = StringAttr::get(circuitOp.getContext(), getName());
      module.setName(newName);
      for (auto *use : node->uses()) {
        auto instanceOp = dyn_cast<firrtl::InstanceOp>(*use->getInstance());
        instanceOp.setModuleName(newName);
        instanceOp.setName(newName);
        if (shouldReplacePorts)
          instanceOp.setPortNamesAttr(
              ArrayAttr::get(circuitOp.getContext(), newNames));
      }
    }

    circuitOp->dump();

    return success();
  }

  std::string getName() const override { return "module-name-sanitizer"; }

  bool acceptSizeIncrease() const override { return true; }

  bool isOneShot() const override { return true; }
};

//===----------------------------------------------------------------------===//
// Reduction Registration
//===----------------------------------------------------------------------===//

void firrtl::FIRRTLReducePatternDialectInterface::populateReducePatterns(
    circt::ReducePatternSet &patterns) const {
  // Gather a list of reduction patterns that we should try. Ideally these are
  // assigned reasonable benefit indicators (higher benefit patterns are
  // prioritized). For example, things that can knock out entire modules while
  // being cheap should be tried first (and thus have higher benefit), before
  // trying to tweak operands of individual arithmetic ops.
  patterns.add<PassReduction, 30>(
      getContext(), firrtl::createDropNamesPass(PreserveValues::None), false,
      true);
  patterns.add<PassReduction, 29>(getContext(),
                                  firrtl::createLowerCHIRRTLPass(), true, true);
  patterns.add<PassReduction, 28>(getContext(), firrtl::createInferWidthsPass(),
                                  true, true);
  patterns.add<PassReduction, 27>(getContext(), firrtl::createInferResetsPass(),
                                  true, true);
  patterns.add<FIRRTLModuleExternalizer, 26>();
  patterns.add<InstanceStubber, 25>();
  patterns.add<MemoryStubber, 24>();
  patterns.add<EagerInliner, 23>();
  patterns.add<PassReduction, 22>(
      getContext(), firrtl::createLowerFIRRTLTypesPass(), true, true);
  patterns.add<PassReduction, 21>(getContext(), firrtl::createExpandWhensPass(),
                                  true, true);
  patterns.add<PassReduction, 20>(getContext(), firrtl::createInlinerPass());
  patterns.add<PassReduction, 18>(getContext(),
                                  firrtl::createIMConstPropPass());
  patterns.add<PassReduction, 17>(
      getContext(),
      firrtl::createRemoveUnusedPortsPass(/*ignoreDontTouch=*/true));
  patterns.add<NodeSymbolRemover, 15>();
  patterns.add<ConnectForwarder, 14>();
  patterns.add<ConnectInvalidator, 13>();
  patterns.add<FIRRTLConstantifier, 12>();
  patterns.add<FIRRTLOperandForwarder<0>, 11>();
  patterns.add<FIRRTLOperandForwarder<1>, 10>();
  patterns.add<FIRRTLOperandForwarder<2>, 9>();
  patterns.add<DetachSubaccesses, 7>();
  patterns.add<AnnotationRemover, 6>();
  patterns.add<RootPortPruner, 5>();
  patterns.add<ExtmoduleInstanceRemover, 4>();
  patterns.add<ConnectSourceOperandForwarder<0>, 3>();
  patterns.add<ConnectSourceOperandForwarder<1>, 2>();
  patterns.add<ConnectSourceOperandForwarder<2>, 1>();
  patterns.add<ModuleInternalNameSanitizer, 0>();
  patterns.add<ModuleNameSanitizer, 0>();
}

void firrtl::registerReducePatternDialectInterface(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, FIRRTLDialect *dialect) {
    dialect->addInterfaces<FIRRTLReducePatternDialectInterface>();
  });
}
