//===- IMDeadCodeElim.cpp - Intermodule Dead Code Elimination ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMapInfoVariant.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-imdeadcodeelim"

using namespace circt;
using namespace firrtl;

// Return true if this op has side-effects except for alloc and read.
static bool hasUnknownSideEffect(Operation *op) {
  return !(mlir::isMemoryEffectFree(op) ||
           mlir::hasSingleEffect<mlir::MemoryEffects::Allocate>(op) ||
           mlir::hasSingleEffect<mlir::MemoryEffects::Read>(op));
}

/// Return true if this is a wire or a register or a node.
static bool isDeclaration(Operation *op) {
  return isa<WireOp, RegResetOp, RegOp, NodeOp, MemOp>(op);
}

/// Return true if this is a wire or register we're allowed to delete.
static bool isDeletableDeclaration(Operation *op) {
  if (auto name = dyn_cast<FNamableOp>(op))
    if (!name.hasDroppableName())
      return false;
  return !hasDontTouch(op);
}

/// Return true if the annotation is ok to drop when the target is dead.
static bool isWeakReferencingAnnotation(Annotation anno) {
  if (!anno.isClass(omirTrackerAnnoClass))
    return false;

  auto tpe = anno.getMember<StringAttr>("type");
  return tpe &&
         (tpe == "OMReferenceTarget" || tpe == "OMMemberReferenceTarget" ||
          tpe == "OMMemberInstanceTarget");
}

namespace {
struct IMDeadCodeElimPass : public IMDeadCodeElimBase<IMDeadCodeElimPass> {
  void runOnOperation() override;

  void rewriteModuleSignature(FModuleOp module);
  void rewriteModuleBody(FModuleOp module);
  void eraseEmptyModule(FModuleOp module);
  void forwardConstantOutputPort(FModuleOp module);

  /// Return true if the value is known alive.
  bool isKnownAlive(Value value) const {
    assert(value && "null should not be used");
    return liveElements.count(value);
  }

  /// Return true if the value is assumed dead.
  bool isAssumedDead(Value value) const { return !isKnownAlive(value); }
  bool isAssumedDead(Operation *op) const {
    return llvm::none_of(op->getResults(),
                         [&](Value value) { return isKnownAlive(value); });
  }

  /// Return true if the block is alive.
  bool isBlockExecutable(Block *block) const {
    return executableBlocks.count(block);
  }

  void visitUser(Operation *op);
  void visitValue(Value value);

  void visitConnect(FConnectLike connect);
  void visitSubelement(Operation *op);
  void markBlockExecutable(Block *block);
  void markBlockUndeletable(Operation *op) {
    markAlive(op->getParentOfType<FModuleOp>());
  }

  void markDeclaration(Operation *op);
  void markInstanceOp(InstanceOp instanceOp);
  void markUnknownSideEffectOp(Operation *op);
  void visitInstanceOp(InstanceOp instance);
  void visitHierPathOp(hw::HierPathOp hierpath);
  void visitModuleOp(FModuleOp module);

private:
  /// The set of blocks that are known to execute, or are intrinsically alive.
  DenseSet<Block *> executableBlocks;

  InstanceGraph *instanceGraph;

  // The type with which we associate liveness.
  using ElementType =
      std::variant<Value, FModuleOp, InstanceOp, hw::HierPathOp>;

  void markAlive(ElementType element) {
    if (!liveElements.insert(element).second)
      return;
    worklist.push_back(element);
  }

  /// A worklist of values whose liveness recently changed, indicating
  /// the users need to be reprocessed.
  SmallVector<ElementType, 64> worklist;
  llvm::DenseSet<ElementType> liveElements;

  /// A map from instances to hierpaths whose last path is the associated
  /// instance.
  DenseMap<InstanceOp, SmallVector<hw::HierPathOp>> instanceToHierPaths;

  /// Hierpath to its users (=non-local annotation targets).
  DenseMap<hw::HierPathOp, SetVector<ElementType>> hierPathToElements;

  /// A cache for a (inner)symbol lookp.
  circt::hw::InnerRefNamespace *innerRefNamespace;
  mlir::SymbolTable *symbolTable;
};
} // namespace

void IMDeadCodeElimPass::visitInstanceOp(InstanceOp instance) {
  markBlockUndeletable(instance);

  auto module =
      dyn_cast<FModuleOp>(*instanceGraph->getReferencedModule(instance));

  if (!module)
    return;

  // NOTE: Don't call `markAlive(module)` here as liveness of instance doesn't
  // imply the global liveness of the module.

  // Propgate liveness through hierpath.
  for (auto hierPath : instanceToHierPaths[instance])
    markAlive(hierPath);

  // Input ports get alive only when the instance is alive.
  for (auto &blockArg : module.getBody().getArguments()) {
    auto portNo = blockArg.getArgNumber();
    if (module.getPortDirection(portNo) == Direction::In &&
        isKnownAlive(module.getArgument(portNo)))
      markAlive(instance.getResult(portNo));
  }
}

void IMDeadCodeElimPass::visitModuleOp(FModuleOp module) {
  // If the module needs to be alive, so are its instances.
  for (auto *use : instanceGraph->lookup(module)->uses())
    markAlive(cast<InstanceOp>(*use->getInstance()));
}

void IMDeadCodeElimPass::visitHierPathOp(hw::HierPathOp hierPathOp) {
  // If the hierpath is alive, mark all instances on the path alive.
  for (auto path : hierPathOp.getNamepathAttr())
    if (auto innerRef = path.dyn_cast<hw::InnerRefAttr>()) {
      auto *op = innerRefNamespace->lookupOp(innerRef);
      if (auto instance = dyn_cast_or_null<InstanceOp>(op))
        markAlive(instance);
    }

  for (auto elem : hierPathToElements[hierPathOp])
    markAlive(elem);
}

void IMDeadCodeElimPass::markDeclaration(Operation *op) {
  assert(isDeclaration(op) && "only a declaration is expected");
  if (!isDeletableDeclaration(op)) {
    for (auto result : op->getResults())
      markAlive(result);
    markBlockUndeletable(op);
  }
}

void IMDeadCodeElimPass::markUnknownSideEffectOp(Operation *op) {
  // For operations with side effects, pessimistically mark results and
  // operands as alive.
  for (auto result : op->getResults())
    markAlive(result);
  for (auto operand : op->getOperands())
    markAlive(operand);
  markBlockUndeletable(op);
}

void IMDeadCodeElimPass::visitUser(Operation *op) {
  LLVM_DEBUG(llvm::dbgs() << "Visit: " << *op << "\n");
  if (auto connectOp = dyn_cast<FConnectLike>(op))
    return visitConnect(connectOp);
  if (isa<SubfieldOp, SubindexOp, SubaccessOp>(op))
    return visitSubelement(op);
}

void IMDeadCodeElimPass::markInstanceOp(InstanceOp instance) {
  // Get the module being referenced.
  Operation *op = instanceGraph->getReferencedModule(instance);

  // If this is an extmodule, just remember that any inputs and inouts are
  // alive.
  if (!isa<FModuleOp>(op)) {
    auto module = dyn_cast<FModuleLike>(op);
    for (auto resultNo : llvm::seq(0u, instance.getNumResults())) {
      // If this is an output to the extmodule, we can ignore it.
      if (module.getPortDirection(resultNo) == Direction::Out)
        continue;

      // Otherwise this is an input from it or an inout, mark it as alive.
      markAlive(instance.getResult(resultNo));
    }
    markAlive(instance);

    return;
  }

  // Otherwise this is a defined module.
  auto fModule = cast<FModuleOp>(op);
  markBlockExecutable(fModule.getBodyBlock());
}

void IMDeadCodeElimPass::markBlockExecutable(Block *block) {
  if (!executableBlocks.insert(block).second)
    return; // Already executable.

  auto fmodule = cast<FModuleOp>(block->getParentOp());
  if (fmodule.isPublic())
    markAlive(fmodule);

  // Mark ports with don't touch as alive.
  for (auto blockArg : block->getArguments())
    if (hasDontTouch(blockArg)) {
      markAlive(blockArg);
      markAlive(fmodule);
    }

  for (auto &op : *block) {
    if (isDeclaration(&op))
      markDeclaration(&op);
    else if (auto instance = dyn_cast<InstanceOp>(op))
      markInstanceOp(instance);
    else if (isa<FConnectLike>(op))
      // Skip connect op.
      continue;
    else if (hasUnknownSideEffect(&op))
      markUnknownSideEffectOp(&op);

    // TODO: Handle attach etc.
  }
}

void IMDeadCodeElimPass::forwardConstantOutputPort(FModuleOp module) {
  // This tracks constant values of output ports.
  SmallVector<std::pair<unsigned, APSInt>> constantPortIndicesAndValues;
  auto ports = module.getPorts();
  auto *instanceGraphNode = instanceGraph->lookup(module);

  for (const auto &e : llvm::enumerate(ports)) {
    unsigned index = e.index();
    auto port = e.value();
    auto arg = module.getArgument(index);

    // If the port has don't touch, don't propagate the constant value.
    if (!port.isOutput() || hasDontTouch(arg))
      continue;

    // Remember the index and constant value connected to an output port.
    if (auto connect = getSingleConnectUserOf(arg))
      if (auto constant = connect.getSrc().getDefiningOp<ConstantOp>())
        constantPortIndicesAndValues.push_back({index, constant.getValue()});
  }

  // If there is no constant port, abort.
  if (constantPortIndicesAndValues.empty())
    return;

  // Rewrite all uses.
  for (auto *use : instanceGraphNode->uses()) {
    auto instance = cast<InstanceOp>(*use->getInstance());
    ImplicitLocOpBuilder builder(instance.getLoc(), instance);
    for (auto [index, constant] : constantPortIndicesAndValues) {
      auto result = instance.getResult(index);
      assert(ports[index].isOutput() && "must be an output port");

      // Replace the port with the constant.
      result.replaceAllUsesWith(builder.create<ConstantOp>(constant));
    }
  }
}

void IMDeadCodeElimPass::runOnOperation() {
  LLVM_DEBUG(
      llvm::dbgs() << "===----- Inter-module Dead Code Elimination -----==="
                   << "\n");
  auto circuits = getOperation().getOps<CircuitOp>();
  if (circuits.empty())
    return;

  auto circuit = *circuits.begin();

  if (!llvm::hasSingleElement(circuits)) {
    mlir::emitError(circuit.getLoc(),
                    "cannot process multiple circuit operations")
            .attachNote((*std::next(circuits.begin())).getLoc())
        << "second circuit here";
    return signalPassFailure();
  }

  instanceGraph = &getChildAnalysis<InstanceGraph>(circuit);
  symbolTable = &getChildAnalysis<SymbolTable>(circuit);
  auto &istc = getChildAnalysis<hw::InnerSymbolTableCollection>(circuit);

  circt::hw::InnerRefNamespace theInnerRefNamespace{*symbolTable, istc};
  innerRefNamespace = &theInnerRefNamespace;

  // Walk attributes and find unknown uses of inner symbols or hierpaths.
  getOperation().walk([&](Operation *op) {
    if (isa<FModuleOp>(op)) // Port or module annotations are ok to ignore.
      return;

    if (auto hierPath = dyn_cast<hw::HierPathOp>(op)) {
      auto namePath = hierPath.getNamepath().getValue();
      // If the hierpath is public or ill-formed, the verifier should have
      // caught the error. Conservatively mark the symbol as alive.
      if (hierPath.isPublic() || namePath.size() <= 1 ||
          namePath.back().isa<hw::InnerRefAttr>())
        return markAlive(hierPath);

      if (auto instance =
              dyn_cast_or_null<firrtl::InstanceOp>(innerRefNamespace->lookupOp(
                  namePath.drop_back().back().cast<hw::InnerRefAttr>())))
        instanceToHierPaths[instance].push_back(hierPath);
      return;
    }

    // If there is an unknown use of inner sym or hierpath, just mark all of
    // them alive.
    for (NamedAttribute namedAttr : op->getAttrs()) {
      namedAttr.getValue().walk([&](Attribute subAttr) {
        if (auto innerRef = dyn_cast<hw::InnerRefAttr>(subAttr))
          if (auto instance = dyn_cast_or_null<firrtl::InstanceOp>(
                  innerRefNamespace->lookupOp(innerRef)))
            markAlive(instance);

        if (auto flatSymbolRefAttr = dyn_cast<FlatSymbolRefAttr>(subAttr))
          if (auto hierPath = symbolTable->template lookup<hw::HierPathOp>(
                  flatSymbolRefAttr.getAttr()))
            markAlive(hierPath);
      });
    }
  });

  // Create a vector of modules in the post order of instance graph.
  // FIXME: We copy the list of modules into a vector first to avoid iterator
  // invalidation while we mutate the instance graph. See issue 3387.
  SmallVector<FModuleOp, 0> modules(llvm::make_filter_range(
      llvm::map_range(
          llvm::post_order(instanceGraph),
          [](auto *node) { return dyn_cast<FModuleOp>(*node->getModule()); }),
      [](auto module) { return module; }));

  // Forward constant output ports to caller sides so that we can eliminate
  // constant outputs.
  for (auto module : modules)
    forwardConstantOutputPort(module);

  for (auto module : circuit.getBodyBlock()->getOps<FModuleOp>()) {
    // Mark the ports of public modules as alive.
    if (module.isPublic()) {
      markBlockExecutable(module.getBodyBlock());
      for (auto port : module.getBodyBlock()->getArguments())
        markAlive(port);
    }

    // Walk annotations and populate a map from hierpath to attached annotation
    // targets. `portId` is `-1` for module annotations.
    auto visitAnnotation = [&](int portId, Annotation anno) -> bool {
      auto hierPathSym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal");
      hw::HierPathOp hierPathOp;
      if (hierPathSym)
        hierPathOp =
            symbolTable->template lookup<hw::HierPathOp>(hierPathSym.getAttr());

      if (isWeakReferencingAnnotation(anno)) {
        if (hierPathOp && portId >= 0)
          hierPathToElements[hierPathOp].insert(module.getArgument(portId));
        return false;
      }
      if (hierPathOp)
        markAlive(hierPathOp);
      if (portId >= 0)
        markAlive(module.getArgument(portId));
      markAlive(module);
      return false;
    };

    AnnotationSet::removePortAnnotations(module, visitAnnotation);
    AnnotationSet::removeAnnotations(
        module, std::bind(visitAnnotation, -1, std::placeholders::_1));
  }

  // If an element changed liveness then propagate liveness through it.
  while (!worklist.empty()) {
    auto v = worklist.pop_back_val();
    if (auto *value = std::get_if<Value>(&v))
      visitValue(*value);
    else if (auto *instance = std::get_if<InstanceOp>(&v))
      visitInstanceOp(*instance);
    else if (auto *hierpath = std::get_if<hw::HierPathOp>(&v))
      visitHierPathOp(*hierpath);
    else if (auto *module = std::get_if<FModuleOp>(&v))
      visitModuleOp(*module);
  }

  // Rewrite module signatures or delete unreachable modules.
  for (auto module : llvm::make_early_inc_range(
           circuit.getBodyBlock()->getOps<FModuleOp>())) {
    if (isBlockExecutable(module.getBodyBlock()))
      rewriteModuleSignature(module);
    else {
      // If the module is unreachable from the toplevel, just delete it.
      // Note that post-order traversal on the instance graph never visit
      // unreachable modules so it's safe to erase the module even though
      // `modules` seems to be capturing module pointers.
      module.erase();
    }
  }

  // Rewrite module bodies parallelly.
  mlir::parallelForEach(circuit.getContext(),
                        circuit.getBodyBlock()->getOps<FModuleOp>(),
                        [&](auto op) { rewriteModuleBody(op); });

  // Clean up hierpaths.
  for (auto op : llvm::make_early_inc_range(
           circuit.getBodyBlock()->getOps<hw::HierPathOp>()))
    if (!liveElements.count(op))
      op.erase();

  for (auto module : modules)
    eraseEmptyModule(module);

  // Clean up data structures.
  executableBlocks.clear();
  liveElements.clear();
  instanceToHierPaths.clear();
  hierPathToElements.clear();
}

void IMDeadCodeElimPass::visitValue(Value value) {
  assert(isKnownAlive(value) && "only alive values reach here");

  // Propagate liveness through users.
  for (Operation *user : value.getUsers())
    visitUser(user);

  // Requiring an input port propagates the liveness to each instance.
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    auto module = cast<FModuleOp>(blockArg.getParentBlock()->getParentOp());
    auto portDirection = module.getPortDirection(blockArg.getArgNumber());
    // If the port is input, it's necessary to mark corresponding input ports of
    // instances as alive. We don't have to propagate the liveness of output
    // ports.
    if (portDirection == Direction::In) {
      for (auto *instRec : instanceGraph->lookup(module)->uses()) {
        auto instance = cast<InstanceOp>(instRec->getInstance());
        if (liveElements.contains(instance))
          markAlive(instance.getResult(blockArg.getArgNumber()));
      }
    }
    return;
  }

  // Marking an instance port as alive propagates to the corresponding port of
  // the module.
  if (auto instance = value.getDefiningOp<InstanceOp>()) {
    auto instanceResult = value.cast<mlir::OpResult>();
    // Update the src, when it's an instance op.
    auto module =
        dyn_cast<FModuleOp>(*instanceGraph->getReferencedModule(instance));

    // Propagate liveness only when a port is output.
    if (!module || module.getPortDirection(instanceResult.getResultNumber()) ==
                       Direction::In)
      return;

    markAlive(instance);

    BlockArgument modulePortVal =
        module.getArgument(instanceResult.getResultNumber());
    return markAlive(modulePortVal);
  }

  // If a port of a memory is alive, all other ports are.
  if (auto mem = value.getDefiningOp<MemOp>()) {
    for (auto port : mem->getResults())
      markAlive(port);
    return;
  }

  // If op is defined by an operation, mark its operands as alive.
  if (auto op = value.getDefiningOp())
    for (auto operand : op->getOperands())
      markAlive(operand);
}

void IMDeadCodeElimPass::visitConnect(FConnectLike connect) {
  // If the dest is alive, mark the source value as alive.
  if (isKnownAlive(connect.getDest()))
    markAlive(connect.getSrc());
}

void IMDeadCodeElimPass::visitSubelement(Operation *op) {
  if (isKnownAlive(op->getOperand(0)))
    markAlive(op->getResult(0));
}

void IMDeadCodeElimPass::rewriteModuleBody(FModuleOp module) {
  auto *body = module.getBodyBlock();
  assert(isBlockExecutable(body) &&
         "unreachable modules must be already deleted");

  auto removeDeadNonLocalAnnotations = [&](int _, Annotation anno) -> bool {
    auto hierPathSym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal");
    // We only clean up non-local annotations here as local annotations will
    // be deleted afterwards.
    if (!isWeakReferencingAnnotation(anno) || !hierPathSym)
      return false;
    auto hierPathOp =
        symbolTable->template lookup<hw::HierPathOp>(hierPathSym.getAttr());
    return !liveElements.count(hierPathOp);
  };

  AnnotationSet::removePortAnnotations(module, removeDeadNonLocalAnnotations);
  AnnotationSet::removeAnnotations(
      module,
      std::bind(removeDeadNonLocalAnnotations, -1, std::placeholders::_1));

  // Walk the IR bottom-up when deleting operations.
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(*body))) {
    // Connects to values that we found to be dead can be dropped.
    if (auto connect = dyn_cast<FConnectLike>(op)) {
      if (isAssumedDead(connect.getDest())) {
        LLVM_DEBUG(llvm::dbgs() << "DEAD: " << connect << "\n";);
        connect.erase();
        ++numErasedOps;
      }
      continue;
    }

    // Delete dead wires, regs, nodes and alloc/read ops.
    if ((isDeclaration(&op) || !hasUnknownSideEffect(&op)) &&
        isAssumedDead(&op)) {
      LLVM_DEBUG(llvm::dbgs() << "DEAD: " << op << "\n";);
      assert(op.use_empty() && "users should be already removed");
      op.erase();
      ++numErasedOps;
      continue;
    }

    // Remove non-sideeffect op using `isOpTriviallyDead`.
    if (mlir::isOpTriviallyDead(&op)) {
      op.erase();
      ++numErasedOps;
    }
  }
}

void IMDeadCodeElimPass::rewriteModuleSignature(FModuleOp module) {
  assert(isBlockExecutable(module.getBodyBlock()) &&
         "unreachable modules must be already deleted");
  InstanceGraphNode *instanceGraphNode = instanceGraph->lookup(module);
  LLVM_DEBUG(llvm::dbgs() << "Prune ports of module: " << module.getName()
                          << "\n");

  auto replaceInstanceResultWithWire = [&](ImplicitLocOpBuilder &builder,
                                           unsigned index,
                                           InstanceOp instance) {
    auto result = instance.getResult(index);
    if (isAssumedDead(result)) {
      // If the result is dead, replace the result with an unrealized conversion
      // cast which works as a dummy placeholder.
      auto wire = builder
                      .create<mlir::UnrealizedConversionCastOp>(
                          ArrayRef<Type>{result.getType()}, ArrayRef<Value>{})
                      ->getResult(0);
      result.replaceAllUsesWith(wire);
      return;
    }

    // If RefType and live, don't want to leave wire around.
    if (type_isa<RefType>(result.getType())) {
      auto getRefDefine = [](Value result) -> RefDefineOp {
        for (auto *user : result.getUsers()) {
          if (auto rd = dyn_cast<RefDefineOp>(user);
              rd && rd.getDest() == result)
            return rd;
        }
        return {};
      };
      auto rd = getRefDefine(result);
      assert(rd && "input ref port to instance is alive, but no driver?");
      auto source = rd.getSrc();
      auto *srcDefOp = source.getDefiningOp();
      if (srcDefOp && llvm::any_of(result.getUsers(), [&](auto user) {
            return user->getBlock() != source.getParentBlock() ||
                   user->isBeforeInBlock(source.getDefiningOp());
          }))
        llvm::report_fatal_error("unsupported IR with references in IMDCE");
      result.replaceAllUsesWith(source);
      liveElements.erase(result);
      assert(isKnownAlive(source));
      ++numErasedOps;
      rd.erase();
      return;
    }

    Value wire = builder.create<WireOp>(result.getType()).getResult();
    result.replaceAllUsesWith(wire);
    // If a module port is dead but its instance result is alive, the port
    // is used as a temporary wire so make sure that a replaced wire is
    // putted into `liveSet`.
    liveElements.erase(result);
    liveElements.insert(wire);
  };

  // First, delete dead instances.
  for (auto *use : llvm::make_early_inc_range(instanceGraphNode->uses())) {
    auto instance = cast<InstanceOp>(*use->getInstance());
    if (!liveElements.count(instance)) {
      // Replace old instance results with dummy wires.
      ImplicitLocOpBuilder builder(instance.getLoc(), instance);
      for (auto index : llvm::seq(0u, instance.getNumResults()))
        replaceInstanceResultWithWire(builder, index, instance);
      // Make sure that we update the instance graph.
      use->erase();
      instance.erase();
    }
  }

  // Ports of public modules cannot be modified.
  if (module.isPublic())
    return;

  unsigned numOldPorts = module.getNumPorts();
  llvm::BitVector deadPortIndexes(numOldPorts);

  ImplicitLocOpBuilder builder(module.getLoc(), module.getContext());
  builder.setInsertionPointToStart(module.getBodyBlock());
  auto oldPorts = module.getPorts();

  for (auto index : llvm::seq(0u, numOldPorts)) {
    auto argument = module.getArgument(index);
    assert((!hasDontTouch(argument) || isKnownAlive(argument)) &&
           "If the port has don't touch, it should be known alive");

    // If the port has dontTouch, skip.
    if (hasDontTouch(argument))
      continue;

    if (isKnownAlive(argument)) {

      // If an output port is only used internally in the module, then we can
      // remove the port and replace it with a wire.
      if (module.getPortDirection(index) == Direction::In)
        continue;

      // Check if the output port is demanded by any instance.  If not, then it
      // is only demanded internally to the module.
      if (llvm::any_of(instanceGraph->lookup(module)->uses(),
                       [&](InstanceRecord *record) {
                         return isKnownAlive(
                             record->getInstance()->getResult(index));
                       }))
        continue;

      // RefType can't be a wire, especially if it won't be erased.  Skip.
      if (type_isa<RefType>(argument.getType()))
        continue;

      // Ok, this port is used only within its defined module. So we can replace
      // the port with a wire.
      auto wire = builder.create<WireOp>(argument.getType()).getResult();

      // Since `liveSet` contains the port, we have to erase it from the set.
      liveElements.erase(argument);
      liveElements.insert(wire);
      argument.replaceAllUsesWith(wire);
      deadPortIndexes.set(index);
      continue;
    }

    // Replace the port with a dummy wire. This wire should be erased within
    // `rewriteModuleBody`.
    Value wire = builder
                     .create<mlir::UnrealizedConversionCastOp>(
                         ArrayRef<Type>{argument.getType()}, ArrayRef<Value>{})
                     ->getResult(0);

    argument.replaceAllUsesWith(wire);
    assert(isAssumedDead(wire) && "dummy wire must be dead");
    deadPortIndexes.set(index);
  }

  // If there is nothing to remove, abort.
  if (deadPortIndexes.none())
    return;

  // Erase arguments of the old module from liveSet to prevent from creating
  // dangling pointers.
  for (auto arg : module.getArguments())
    liveElements.erase(arg);

  // Delete ports from the module.
  module.erasePorts(deadPortIndexes);

  // Add arguments of the new module to liveSet.
  for (auto arg : module.getArguments())
    liveElements.insert(arg);

  // Rewrite all uses.
  for (auto *use : llvm::make_early_inc_range(instanceGraphNode->uses())) {
    auto instance = cast<InstanceOp>(*use->getInstance());
    ImplicitLocOpBuilder builder(instance.getLoc(), instance);
    // Replace old instance results with dummy wires.
    for (auto index : deadPortIndexes.set_bits())
      replaceInstanceResultWithWire(builder, index, instance);

    // Since we will rewrite instance op, it is necessary to remove old
    // instance results from liveSet.
    for (auto oldResult : instance.getResults())
      liveElements.erase(oldResult);

    // Create a new instance op without dead ports.
    auto newInstance = instance.erasePorts(builder, deadPortIndexes);

    // Mark new results as alive.
    for (auto newResult : newInstance.getResults())
      liveElements.insert(newResult);

    instanceGraph->replaceInstance(instance, newInstance);
    if (liveElements.contains(instance)) {
      liveElements.erase(instance);
      liveElements.insert(newInstance);
    }
    // Remove old one.
    instance.erase();
  }

  numRemovedPorts += deadPortIndexes.count();
}

void IMDeadCodeElimPass::eraseEmptyModule(FModuleOp module) {
  // If the module is not empty, just skip.
  if (!module.getBodyBlock()->empty())
    return;

  // We cannot delete public modules so generate a warning.
  if (module.isPublic()) {
    mlir::emitWarning(module.getLoc())
        << "module `" << module.getName()
        << "` is empty but cannot be removed because the module is public";
    return;
  }

  if (!module.getAnnotations().empty()) {
    module.emitWarning() << "module `" << module.getName()
                         << "` is empty but cannot be removed "
                            "because the module has annotations "
                         << module.getAnnotations();
    return;
  }

  if (!module.getBodyBlock()->args_empty()) {
    auto diag = module.emitWarning()
                << "module `" << module.getName()
                << "` is empty but cannot be removed because the "
                   "module has ports ";
    llvm::interleaveComma(module.getPortNames(), diag);
    diag << " are referenced by name or dontTouched";
    return;
  }

  // Ok, the module is empty. Delete instances unless they have symbols.
  LLVM_DEBUG(llvm::dbgs() << "Erase " << module.getName() << "\n");

  InstanceGraphNode *instanceGraphNode =
      instanceGraph->lookup(module.getModuleNameAttr());

  SmallVector<Location> instancesWithSymbols;
  for (auto *use : llvm::make_early_inc_range(instanceGraphNode->uses())) {
    auto instance = cast<InstanceOp>(use->getInstance());
    if (instance.getInnerSym()) {
      instancesWithSymbols.push_back(instance.getLoc());
      continue;
    }
    use->erase();
    instance.erase();
  }

  // If there is an instance with a symbol, we don't delete the module itself.
  if (!instancesWithSymbols.empty()) {
    auto diag = module.emitWarning()
                << "module `" << module.getName()
                << "` is empty but cannot be removed because an instance is "
                   "referenced by name";
    diag.attachNote(FusedLoc::get(&getContext(), instancesWithSymbols))
        << "these are instances with symbols";
    return;
  }

  instanceGraph->erase(instanceGraphNode);
  module.erase();
  ++numErasedModules;
}

std::unique_ptr<mlir::Pass> circt::firrtl::createIMDeadCodeElimPass() {
  return std::make_unique<IMDeadCodeElimPass>();
}
