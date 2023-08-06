//===- Dedup.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SHA256.h"

#define DEBUG_TYPE "arc-dedup"

using namespace circt;
using namespace arc;
using namespace hw;
using llvm::SmallMapVector;
using llvm::SmallSetVector;

namespace {
struct StructuralHash {
  using Hash = std::array<uint8_t, 32>;
  Hash hash;
  Hash constInvariant; // a hash that ignores constants
};

struct StructuralHasher {
  explicit StructuralHasher(MLIRContext *context) {}

  StructuralHash hash(DefineOp arc) {
    reset();
    update(arc);
    return StructuralHash{state.final(), stateConstInvariant.final()};
  }

private:
  void reset() {
    currentIndex = 0;
    disableConstInvariant = 0;
    indices.clear();
    indicesConstInvariant.clear();
    state.init();
    stateConstInvariant.init();
  }

  void update(const void *pointer) {
    auto *addr = reinterpret_cast<const uint8_t *>(&pointer);
    state.update(ArrayRef<uint8_t>(addr, sizeof pointer));
    if (disableConstInvariant == 0)
      stateConstInvariant.update(ArrayRef<uint8_t>(addr, sizeof pointer));
  }

  void update(size_t value) {
    auto *addr = reinterpret_cast<const uint8_t *>(&value);
    state.update(ArrayRef<uint8_t>(addr, sizeof value));
    if (disableConstInvariant == 0)
      stateConstInvariant.update(ArrayRef<uint8_t>(addr, sizeof value));
  }

  void update(size_t value, size_t valueConstInvariant) {
    state.update(ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(&value),
                                   sizeof value));
    state.update(ArrayRef<uint8_t>(
        reinterpret_cast<const uint8_t *>(&valueConstInvariant),
        sizeof valueConstInvariant));
  }

  void update(TypeID typeID) { update(typeID.getAsOpaquePointer()); }

  void update(Type type) { update(type.getAsOpaquePointer()); }

  void update(Attribute attr) { update(attr.getAsOpaquePointer()); }

  void update(mlir::OperationName name) { update(name.getAsOpaquePointer()); }

  void update(BlockArgument arg) { update(arg.getType()); }

  void update(OpResult result) { update(result.getType()); }

  void update(OpOperand &operand) {
    // We hash the value's index as it apears in the block.
    auto it = indices.find(operand.get());
    auto itCI = indicesConstInvariant.find(operand.get());
    assert(it != indices.end() && itCI != indicesConstInvariant.end() &&
           "op should have been previously hashed");
    update(it->second, itCI->second);
  }

  void update(Block &block) {
    // Assign integer numbers to block arguments and op results. For the const-
    // invariant hash, assign a zero to block args and constant ops, such that
    // they hash as the same.
    for (auto arg : block.getArguments()) {
      indices.insert({arg, currentIndex++});
      indicesConstInvariant.insert({arg, 0});
    }
    for (auto &op : block) {
      for (auto result : op.getResults()) {
        indices.insert({result, currentIndex++});
        if (op.hasTrait<OpTrait::ConstantLike>())
          indicesConstInvariant.insert({result, 0});
        else
          indicesConstInvariant.insert({result, currentIndexConstInvariant++});
      }
    }

    // Hash the block arguments for the const-invariant hash.
    ++disableConstInvariant;
    for (auto arg : block.getArguments())
      update(arg);
    --disableConstInvariant;

    // Hash the operations.
    for (auto &op : block)
      update(&op);
  }

  void update(Operation *op) {
    unsigned skipConstInvariant = op->hasTrait<OpTrait::ConstantLike>();
    disableConstInvariant += skipConstInvariant;

    update(op->getName());

    // Hash the attributes. (Excluded in constant invariant hash.)
    if (!isa<DefineOp>(op)) {
      for (auto namedAttr : op->getAttrDictionary()) {
        auto name = namedAttr.getName();
        auto value = namedAttr.getValue();

        // Hash the interned pointer.
        update(name.getAsOpaquePointer());
        update(value.getAsOpaquePointer());
      }
    }

    // Hash the operands.
    for (auto &operand : op->getOpOperands())
      update(operand);
    // Hash the regions. We need to make sure an empty region doesn't hash
    // the same as no region, so we include the number of regions.
    update(op->getNumRegions());
    for (auto &region : op->getRegions())
      for (auto &block : region.getBlocks())
        update(block);
    // Record any op results.
    for (auto result : op->getResults())
      update(result);

    disableConstInvariant -= skipConstInvariant;
  }

  // Every value is assigned a unique id based on their order of appearance.
  unsigned currentIndex = 0;
  unsigned currentIndexConstInvariant = 0;
  DenseMap<Value, unsigned> indices;
  DenseMap<Value, unsigned> indicesConstInvariant;

  unsigned disableConstInvariant = 0;

  // This is the actual running hash calculation. This is a stateful element
  // that should be reinitialized after each hash is produced.
  llvm::SHA256 state;
  llvm::SHA256 stateConstInvariant;
};
} // namespace

namespace {
struct StructuralEquivalence {
  using OpOperandPair = std::pair<OpOperand *, OpOperand *>;
  explicit StructuralEquivalence(MLIRContext *context) {}

  void check(DefineOp arcA, DefineOp arcB) {
    if (!checkImpl(arcA, arcB)) {
      match = false;
      matchConstInvariant = false;
    }
  }

  SmallSetVector<OpOperandPair, 1> divergences;
  bool match;
  bool matchConstInvariant;

private:
  bool addBlockToWorklist(Block &blockA, Block &blockB) {
    auto *terminatorA = blockA.getTerminator();
    auto *terminatorB = blockB.getTerminator();
    if (!compareOps(terminatorA, terminatorB, OpOperandPair()))
      return false;
    if (!addOpToWorklist(terminatorA, terminatorB))
      return false;
    // TODO: We should probably bail out if there are any operations in the
    // block that aren't in the fan-in of the terminator.
    return true;
  }

  bool addOpToWorklist(Operation *opA, Operation *opB,
                       bool *allOperandsHandled = nullptr) {
    if (opA->getNumOperands() != opB->getNumOperands())
      return false;
    for (auto [operandA, operandB] :
         llvm::zip(opA->getOpOperands(), opB->getOpOperands())) {
      if (!handled.count({&operandA, &operandB})) {
        worklist.emplace_back(&operandA, &operandB);
        if (allOperandsHandled)
          *allOperandsHandled = false;
      }
    }
    return true;
  }

  bool compareOps(Operation *opA, Operation *opB, OpOperandPair values) {
    if (opA->getName() != opB->getName())
      return false;
    if (opA->getAttrDictionary() != opB->getAttrDictionary()) {
      for (auto [namedAttrA, namedAttrB] :
           llvm::zip(opA->getAttrDictionary(), opB->getAttrDictionary())) {
        if (namedAttrA.getName() != namedAttrB.getName())
          return false;
        if (namedAttrA.getValue() == namedAttrB.getValue())
          continue;
        bool mayDiverge = opA->hasTrait<OpTrait::ConstantLike>();
        if (!mayDiverge || !values.first || !values.second)
          return false;
        divergences.insert(values);
        match = false;
        break;
      }
    }
    return true;
  }

  bool checkImpl(DefineOp arcA, DefineOp arcB) {
    worklist.clear();
    divergences.clear();
    match = true;
    matchConstInvariant = true;
    handled.clear();

    if (arcA.getFunctionType().getResults() !=
        arcB.getFunctionType().getResults())
      return false;

    if (!addBlockToWorklist(arcA.getBodyBlock(), arcB.getBodyBlock()))
      return false;

    while (!worklist.empty()) {
      OpOperandPair values = worklist.back();
      if (handled.contains(values)) {
        worklist.pop_back();
        continue;
      }

      auto valueA = values.first->get();
      auto valueB = values.second->get();
      if (valueA.getType() != valueB.getType())
        return false;
      auto *opA = valueA.getDefiningOp();
      auto *opB = valueB.getDefiningOp();

      // Handle the case where one or both values are block arguments.
      if (!opA || !opB) {
        auto argA = valueA.dyn_cast<BlockArgument>();
        auto argB = valueB.dyn_cast<BlockArgument>();
        if (argA && argB) {
          divergences.insert(values);
          if (argA.getArgNumber() != argB.getArgNumber())
            match = false;
          handled.insert(values);
          worklist.pop_back();
          continue;
        }
        auto isConstA = opA && opA->hasTrait<OpTrait::ConstantLike>();
        auto isConstB = opB && opB->hasTrait<OpTrait::ConstantLike>();
        if ((argA && isConstB) || (argB && isConstA)) {
          // One value is a block argument, one is a constant.
          divergences.insert(values);
          match = false;
          handled.insert(values);
          worklist.pop_back();
          continue;
        }
        return false;
      }

      // Go through all operands push the ones we haven't visited yet onto the
      // worklist so they get processed before we continue.
      bool allHandled = true;
      if (!addOpToWorklist(opA, opB, &allHandled))
        return false;
      if (!allHandled)
        continue;
      handled.insert(values);
      worklist.pop_back();

      // Compare the two operations and check that they are equal.
      if (!compareOps(opA, opB, values))
        return false;

      // Descend into subregions of the operation.
      if (opA->getNumRegions() != opB->getNumRegions())
        return false;
      for (auto [regionA, regionB] :
           llvm::zip(opA->getRegions(), opB->getRegions())) {
        if (regionA.getBlocks().size() != regionB.getBlocks().size())
          return false;
        for (auto [blockA, blockB] : llvm::zip(regionA, regionB))
          if (!addBlockToWorklist(blockA, blockB))
            return false;
      }
    }

    return true;
  }

  SmallVector<OpOperandPair, 0> worklist;
  DenseSet<OpOperandPair> handled;
};
} // namespace

static void addCallSiteOperands(
    MutableArrayRef<mlir::CallOpInterface> callSites,
    ArrayRef<std::variant<Operation *, unsigned>> operandMappings) {
  SmallDenseMap<Operation *, Operation *> clonedOps;
  SmallVector<Value> newOperands;
  for (auto &callOp : callSites) {
    OpBuilder builder(callOp);
    newOperands.clear();
    clonedOps.clear();
    for (auto mapping : operandMappings) {
      if (std::holds_alternative<Operation *>(mapping)) {
        auto *op = std::get<Operation *>(mapping);
        auto &newOp = clonedOps[op];
        if (!newOp)
          newOp = builder.clone(*op);
        newOperands.push_back(newOp->getResult(0));
      } else {
        newOperands.push_back(
            callOp.getArgOperands()[std::get<unsigned>(mapping)]);
      }
    }
    callOp.getArgOperandsMutable().assign(newOperands);
  }
}

static bool isOutlinable(OpOperand &operand) {
  auto *op = operand.get().getDefiningOp();
  return !op || op->hasTrait<OpTrait::ConstantLike>();
}

namespace {
struct DedupPass : public DedupBase<DedupPass> {
  void runOnOperation() override;
  void replaceArcWith(DefineOp oldArc, DefineOp newArc);

  /// A mapping from arc names to arc definitions.
  DenseMap<StringAttr, DefineOp> arcByName;
  /// A mapping from arc definitions to call sites.
  DenseMap<DefineOp, SmallVector<mlir::CallOpInterface, 1>> callSites;
};

struct ArcHash {
  DefineOp defineOp;
  StructuralHash hash;
  unsigned order;
  ArcHash(DefineOp defineOp, StructuralHash hash, unsigned order)
      : defineOp(defineOp), hash(hash), order(order) {}
};
} // namespace

void DedupPass::runOnOperation() {
  arcByName.clear();
  callSites.clear();
  SymbolTableCollection symbolTable;

  // Compute the structural hash for each arc definition.
  SmallVector<ArcHash> arcHashes;
  StructuralHasher hasher(&getContext());
  for (auto defineOp : getOperation().getOps<DefineOp>()) {
    arcHashes.emplace_back(defineOp, hasher.hash(defineOp), arcHashes.size());
    arcByName.insert({defineOp.getSymNameAttr(), defineOp});
  }

  // Collect the arc call sites.
  getOperation().walk([&](mlir::CallOpInterface callOp) {
    if (auto defOp =
            dyn_cast_or_null<DefineOp>(callOp.resolveCallable(&symbolTable)))
      callSites[arcByName.lookup(callOp.getCallableForCallee()
                                     .get<mlir::SymbolRefAttr>()
                                     .getLeafReference())]
          .push_back(callOp);
  });

  // Sort the arcs by hash such that arcs with the same hash are next to each
  // other, and sort arcs with the same hash by order in which they appear in
  // the input. This allows us to iterate through the list and check
  // neighbouring arcs for merge opportunities.
  llvm::stable_sort(arcHashes, [](auto a, auto b) {
    if (a.hash.hash < b.hash.hash)
      return true;
    if (a.hash.hash > b.hash.hash)
      return false;
    return a.order < b.order;
  });

  // Perform deduplications that do not require modification of the arc call
  // sites. (No additional ports.)
  LLVM_DEBUG(llvm::dbgs() << "Check for exact merges (" << arcHashes.size()
                          << " arcs)\n");
  StructuralEquivalence equiv(&getContext());
  for (unsigned arcIdx = 0, arcEnd = arcHashes.size(); arcIdx != arcEnd;
       ++arcIdx) {
    auto [defineOp, hash, order] = arcHashes[arcIdx];
    if (!defineOp)
      continue;
    for (unsigned otherIdx = arcIdx + 1; otherIdx != arcEnd; ++otherIdx) {
      auto [otherDefineOp, otherHash, otherOrder] = arcHashes[otherIdx];
      if (hash.hash != otherHash.hash)
        break;
      if (!otherDefineOp)
        continue;
      equiv.check(defineOp, otherDefineOp);
      if (!equiv.match)
        continue;
      LLVM_DEBUG(llvm::dbgs()
                 << "- Merge " << defineOp.getSymNameAttr() << " <- "
                 << otherDefineOp.getSymNameAttr() << "\n");
      replaceArcWith(otherDefineOp, defineOp);
      arcHashes[otherIdx].defineOp = {};
    }
  }

  // The initial pass over the arcs has set the `defineOp` to null for every arc
  // that was already merged. Now sort the list of arcs as follows:
  // - All merged arcs are moved to the back of the list (`!defineOp`)
  // - Sort unmerged arcs by const-invariant hash
  // - Sort arcs with same hash by order in which they appear in the input
  // This allows us to pop the merged arcs off of the back of the list. Then we
  // can iterate through the list and check neighbouring arcs for merge
  // opportunities.
  llvm::stable_sort(arcHashes, [](auto a, auto b) {
    if (!a.defineOp && !b.defineOp)
      return false;
    if (!a.defineOp)
      return false;
    if (!b.defineOp)
      return true;
    if (a.hash.constInvariant < b.hash.constInvariant)
      return true;
    if (a.hash.constInvariant > b.hash.constInvariant)
      return false;
    return a.order < b.order;
  });
  while (!arcHashes.empty() && !arcHashes.back().defineOp)
    arcHashes.pop_back();

  // Perform deduplication of arcs that differ only in constant values.
  LLVM_DEBUG(llvm::dbgs() << "Check for constant-agnostic merges ("
                          << arcHashes.size() << " arcs)\n");
  for (unsigned arcIdx = 0, arcEnd = arcHashes.size(); arcIdx != arcEnd;
       ++arcIdx) {
    auto [defineOp, hash, order] = arcHashes[arcIdx];
    if (!defineOp)
      continue;

    // Perform an initial pass over all other arcs with identical
    // const-invariant hash. Check for equivalence between the current arc
    // (`defineOp`) and the other arc (`otherDefineOp`). In case they match
    // iterate over the list of divergences which holds all non-identical
    // OpOperand pairs in the two arcs. These can come in different forms:
    //
    // - (const, const): Both arcs have the operand set to a constant, but the
    //     constant value differs. We'll want to extract these constants.
    // - (arg, const): The current arc has a block argument where the other has
    //     a constant. No changes needed; when we replace the uses of the other
    //     arc with the current one further done we can use the existing
    //     argument to pass in that constant.
    // - (const, arg): The current arc has a constant where the other has a
    //     block argument. We'll want to extract this constant and replace it
    //     with a block argument. This will allow the other arc to be replaced
    //     with the current one.
    // - (arg, arg): Both arcs have the operand set to a block argument, but
    //     they are different argument numbers. This can happen if for example
    //     one of the arcs uses a single argument in two op operands and the
    //     other arc has two separate arguments for the two op operands. We'll
    //     want to ensure the current arc has two arguments in this case, such
    //     that the two can dedup.
    //
    // Whenever an op operand is involved in such a divergence we add it to the
    // list of operands that must be mapped to a distinct block argument. Later
    // we'll go through this list and add additional block arguments as
    // necessary.
    SmallMapVector<OpOperand *, unsigned, 8> outlineOperands;
    unsigned nextGroupId = 1;
    SmallMapVector<Value,
                   SmallMapVector<Value, SmallSetVector<OpOperand *, 1>, 2>, 2>
        operandMappings;
    SmallVector<StringAttr> candidateNames;

    for (unsigned otherIdx = arcIdx + 1; otherIdx != arcEnd; ++otherIdx) {
      auto [otherDefineOp, otherHash, otherOrder] = arcHashes[otherIdx];
      if (hash.constInvariant != otherHash.constInvariant)
        break;
      if (!otherDefineOp)
        continue;

      equiv.check(defineOp, otherDefineOp);
      if (!equiv.matchConstInvariant)
        continue;
      candidateNames.push_back(otherDefineOp.getSymNameAttr());

      // Iterate over the matching operand pairs ("divergences"), look up the
      // value pair the operands are set to, and then store the current arc's
      // operand in the set that corresponds to this value pair. This builds up
      // `operandMappings` to contain sets of op operands in the current arc
      // that can be routed out to the same block argument. If a block argument
      // of the current arc corresponds to multiple different things in the
      // other arc, this ensures that all unique such combinations get grouped
      // in distinct sets such that we can create an appropriate number of new
      // block args.
      operandMappings.clear();
      for (auto [operand, otherOperand] : equiv.divergences) {
        if (!isOutlinable(*operand) || !isOutlinable(*otherOperand))
          continue;
        operandMappings[operand->get()][otherOperand->get()].insert(operand);
      }

      // Go through the sets of operands that can map to the same block argument
      // for the combination of current and other arc. Assign all operands in
      // each set new unique group IDs. If the operands in the set have multiple
      // IDs, allocate multiple new unique group IDs. This fills the
      // `outlineOperands` map with operands and their corresponding group ID.
      // If we find multiple other arcs that we can potentially combine with the
      // current arc, the operands get distributed into more and more smaller
      // groups. For example, in arc A we can assign operands X and Y to the
      // same block argument, so we assign them the same ID; but in arc B we
      // have to assign X and Y to different block arguments, at which point
      // that same ID we assigned earlier gets reassigned to two new IDs, one
      // for each operand.
      for (auto &[value, mappings] : operandMappings) {
        for (auto &[otherValue, operands] : mappings) {
          SmallDenseMap<unsigned, unsigned> remappedGroupIds;
          for (auto *operand : operands) {
            auto &id = outlineOperands[operand];
            auto &remappedId = remappedGroupIds[id];
            if (remappedId == 0)
              remappedId = nextGroupId++;
            id = remappedId;
          }
        }
      }
    }

    if (outlineOperands.empty())
      continue;
    LLVM_DEBUG({
      llvm::dbgs() << "- Outlining " << outlineOperands.size()
                   << " operands from " << defineOp.getSymNameAttr() << "\n";
      for (auto entry : outlineOperands)
        llvm::dbgs() << "  - Operand #" << entry.first->getOperandNumber()
                     << " of " << *entry.first->getOwner() << "\n";
      for (auto name : candidateNames)
        llvm::dbgs() << "  - Candidate " << name << "\n";
    });

    // Sort the operands to be outlined. The order is already deterministic at
    // this point, but is not really correlated to the existing block argument
    // order since we gathered these operands by traversing the operations
    // depth-first. Establish an order that first honors existing argument order
    // (putting constants at the back), and then considers the order of
    // operations and op operands.
    llvm::stable_sort(outlineOperands, [](auto &a, auto &b) {
      auto argA = a.first->get().template dyn_cast<BlockArgument>();
      auto argB = b.first->get().template dyn_cast<BlockArgument>();
      if (argA && !argB)
        return true;
      if (!argA && argB)
        return false;
      if (argA && argB) {
        if (argA.getArgNumber() < argB.getArgNumber())
          return true;
        if (argA.getArgNumber() > argB.getArgNumber())
          return false;
      }
      auto *opA = a.first->get().getDefiningOp();
      auto *opB = b.first->get().getDefiningOp();
      if (opA == opB)
        return a.first->getOperandNumber() < b.first->getOperandNumber();
      if (opA->getBlock() == opB->getBlock())
        return opA->isBeforeInBlock(opB);
      return false;
    });

    // Build a new set of arc arguments by iterating over the operands that we
    // have determined must be exposed as arguments above. For each operand
    // either reuse its existing block argument (if no other operand in the list
    // has already reused it), or add a new argument for this operand. Also
    // track how each argument must be connected at call sites (outlined
    // constant op or reusing an existing operand).
    unsigned oldArgumentCount = defineOp.getNumArguments();
    SmallDenseMap<unsigned, Value> newArguments; // by group ID
    SmallVector<Type> newInputTypes;
    SmallVector<std::variant<Operation *, unsigned>> newOperands;
    SmallPtrSet<Operation *, 8> outlinedOps;

    for (auto [operand, groupId] : outlineOperands) {
      auto &arg = newArguments[groupId];
      if (!arg) {
        auto value = operand->get();
        arg = defineOp.getBodyBlock().addArgument(value.getType(),
                                                  value.getLoc());
        newInputTypes.push_back(arg.getType());
        if (auto blockArg = value.dyn_cast<BlockArgument>())
          newOperands.push_back(blockArg.getArgNumber());
        else {
          auto *op = value.getDefiningOp();
          newOperands.push_back(op);
          outlinedOps.insert(op);
        }
      }
      operand->set(arg);
    }

    for (auto arg :
         defineOp.getBodyBlock().getArguments().slice(0, oldArgumentCount)) {
      if (!arg.use_empty()) {
        auto d = defineOp.emitError(
                     "dedup failed to replace all argument uses; arc ")
                 << defineOp.getSymNameAttr() << ", argument "
                 << arg.getArgNumber();
        for (auto &use : arg.getUses())
          d.attachNote(use.getOwner()->getLoc())
              << "used in operand " << use.getOperandNumber() << " here";
        return signalPassFailure();
      }
    }

    defineOp.getBodyBlock().eraseArguments(0, oldArgumentCount);
    defineOp.setType(FunctionType::get(
        &getContext(), newInputTypes, defineOp.getFunctionType().getResults()));
    addCallSiteOperands(callSites[defineOp], newOperands);
    for (auto *op : outlinedOps)
      if (op->use_empty())
        op->erase();

    // Perform the actual deduplication with other arcs.
    for (unsigned otherIdx = arcIdx + 1; otherIdx != arcEnd; ++otherIdx) {
      auto [otherDefineOp, otherHash, otherOrder] = arcHashes[otherIdx];
      if (hash.constInvariant != otherHash.constInvariant)
        break;
      if (!otherDefineOp)
        continue;

      // Check for structural equivalence between the two arcs.
      equiv.check(defineOp, otherDefineOp);
      if (!equiv.matchConstInvariant)
        continue;

      // Determine how the other arc's operands map to the arc we're trying to
      // merge into.
      std::variant<Operation *, unsigned> nullOperand = nullptr;
      for (auto &operand : newOperands)
        operand = nullptr;

      bool mappingFailed = false;
      for (auto [operand, otherOperand] : equiv.divergences) {
        auto arg = operand->get().dyn_cast<BlockArgument>();
        if (!arg || !isOutlinable(*otherOperand)) {
          mappingFailed = true;
          break;
        }

        // Determine how the other arc's operand maps to the new connection
        // scheme of the current arc.
        std::variant<Operation *, unsigned> newOperand;
        if (auto otherArg = otherOperand->get().dyn_cast<BlockArgument>())
          newOperand = otherArg.getArgNumber();
        else
          newOperand = otherOperand->get().getDefiningOp();

        // Ensure that there are no conflicting operand assignment.
        auto &newOperandSlot = newOperands[arg.getArgNumber()];
        if (newOperandSlot != nullOperand && newOperandSlot != newOperand) {
          mappingFailed = true;
          break;
        }
        newOperandSlot = newOperand;
      }
      if (mappingFailed) {
        LLVM_DEBUG(llvm::dbgs() << "  - Mapping failed; skipping arc\n");
        continue;
      }
      if (llvm::any_of(newOperands,
                       [&](auto operand) { return operand == nullOperand; })) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  - Not all operands mapped; skipping arc\n");
        continue;
      }

      // Replace all uses of the other arc with the current arc.
      LLVM_DEBUG(llvm::dbgs()
                 << "  - Merged " << defineOp.getSymNameAttr() << " <- "
                 << otherDefineOp.getSymNameAttr() << "\n");
      addCallSiteOperands(callSites[otherDefineOp], newOperands);
      replaceArcWith(otherDefineOp, defineOp);
      arcHashes[otherIdx].defineOp = {};
    }
  }
}

void DedupPass::replaceArcWith(DefineOp oldArc, DefineOp newArc) {
  ++dedupPassNumArcsDeduped;
  auto oldArcOps = oldArc.getOps();
  dedupPassTotalOps += std::distance(oldArcOps.begin(), oldArcOps.end());
  auto &oldUses = callSites[oldArc];
  auto &newUses = callSites[newArc];
  auto newArcName = SymbolRefAttr::get(newArc.getSymNameAttr());
  for (auto callOp : oldUses) {
    callOp.setCalleeFromCallable(newArcName);
    newUses.push_back(callOp);
  }
  callSites.erase(oldArc);
  arcByName.erase(oldArc.getSymNameAttr());
  oldArc->erase();
}

std::unique_ptr<Pass> arc::createDedupPass() {
  return std::make_unique<DedupPass>();
}
