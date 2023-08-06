//===- LegalizeStateUpdate.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-legalize-state-update"

using namespace mlir;
using namespace mlir::dataflow;
using namespace circt;
using namespace arc;
using namespace hw;
using llvm::PointerIntPair;

//===----------------------------------------------------------------------===//
// Data Flow Analysis
//===----------------------------------------------------------------------===//

/// Check if a type is interesting in terms of state accesses.
static bool isTypeInteresting(Type type) { return type.isa<StateType>(); }

/// Check if an operation partakes in state accesses.
static bool isOpInteresting(Operation *op) {
  if (isa<StateReadOp, StateWriteOp>(op))
    return true;
  if (auto callOp = dyn_cast<CallOpInterface>(op))
    return llvm::any_of(callOp.getArgOperands(), [](auto arg) {
      return isTypeInteresting(arg.getType());
    });
  if (auto callableOp = dyn_cast<CallableOpInterface>(op))
    if (auto *region = callableOp.getCallableRegion())
      return llvm::any_of(region->getArguments(), [](auto arg) {
        return isTypeInteresting(arg.getType());
      });
  if (op->getNumRegions() > 0)
    return true;
  return false;
}

namespace {
struct AccessState : public AnalysisState {
  using AnalysisState::AnalysisState;
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AccessState)

  enum AccessType { Read = 0, Write = 1 };
  using Access = PointerIntPair<Value, 1, AccessType>;

  void print(raw_ostream &os) const override {
    if (accesses.empty()) {
      os << "no accesses\n";
      return;
    }
    for (auto access : accesses) {
      os << "- " << (access.getInt() == Read ? "read" : "write") << " "
         << access.getPointer() << "\n";
    }
  }

  ChangeResult join(const AccessState &other) {
    auto result = ChangeResult::NoChange;
    for (auto access : other.accesses)
      if (accesses.insert(access).second)
        result = ChangeResult::Change;
    return result;
  }

  ChangeResult add(Value state, AccessType type) {
    return add(Access(state, type));
  }

  ChangeResult add(Access access) {
    if (accesses.insert(access).second)
      return ChangeResult::Change;
    return ChangeResult::NoChange;
  }

  ChangeResult remove(Value state, AccessType type) {
    return remove(Access(state, type));
  }

  ChangeResult remove(Access access) {
    if (accesses.erase(access))
      return ChangeResult::Change;
    return ChangeResult::NoChange;
  }

  SmallPtrSet<Access, 1> accesses;
};

struct ArgumentAccessState : public AccessState {
  using AccessState::AccessState;
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ArgumentAccessState)
};

struct AccessAnalysis : public DataFlowAnalysis {
  using DataFlowAnalysis::DataFlowAnalysis;

  LogicalResult initialize(Operation *top) override;
  LogicalResult visit(ProgramPoint point) override;

  void visitBlock(Block *block);
  void visitOperation(Operation *op);

  void recordState(Value value) {
    stateOrder.insert({value, stateOrder.size()});
  }

  /// A global order assigned to state values. These allow us to not care about
  /// ordering during the access analysis and only establish a determinstic
  /// order once we insert additional operations later on.
  DenseMap<Value, unsigned> stateOrder;

  /// A symbol table cache.
  SymbolTableCollection symbolTable;
};
} // namespace

LogicalResult AccessAnalysis::initialize(Operation *top) {
  top->walk([&](Block *block) {
    visitBlock(block);
    for (Operation &op : *block)
      if (isOpInteresting(&op))
        visitOperation(&op);
  });
  LLVM_DEBUG(llvm::dbgs() << "Initialized\n");
  return success();
}

LogicalResult AccessAnalysis::visit(ProgramPoint point) {
  if (auto *op = point.dyn_cast<Operation *>()) {
    visitOperation(op);
    return success();
  }
  if (auto *block = point.dyn_cast<Block *>()) {
    visitBlock(block);
    return success();
  }
  return emitError(point.getLoc(), "unknown point kind");
}

void AccessAnalysis::visitBlock(Block *block) {
  // LLVM_DEBUG(llvm::dbgs() << "Visit block " << block << "\n");
  for (auto arg : block->getArguments())
    if (isTypeInteresting(arg.getType()))
      recordState(arg);

  // Aggregate the accesses performed by the operations in this block.
  SmallPtrSet<Value, 4> localState;
  AccessState innerAccesses(block);
  for (Operation &op : *block) {
    if (isa<AllocStateOp>(&op)) {
      localState.insert(op.getResult(0));
      recordState(op.getResult(0));
    }
    if (!isOpInteresting(&op))
      continue;
    innerAccesses.join(*getOrCreateFor<AccessState>(block, &op));
  }

  // Remove any information about locally-defined state which we cannot access
  // outside the current block. This prevents significant blow-up of the access
  // sets, since local state accesses don't get transported to parent ops where
  // they have no meaning.
  for (auto state : localState) {
    innerAccesses.remove(state, AccessState::Read);
    innerAccesses.remove(state, AccessState::Write);
  }

  // Track block argument accesses in a separate analysis state.
  auto *argAccesses = getOrCreate<ArgumentAccessState>(block);
  auto result = ChangeResult::NoChange;
  for (auto arg : block->getArguments()) {
    if (innerAccesses.remove(arg, AccessState::Read) == ChangeResult::Change)
      result |= argAccesses->add(arg, AccessState::Read);
    if (innerAccesses.remove(arg, AccessState::Write) == ChangeResult::Change)
      result |= argAccesses->add(arg, AccessState::Write);
  }
  propagateIfChanged(argAccesses, result);

  // Update the block's access list.
  auto *blockAccesses = getOrCreate<AccessState>(block);
  result = blockAccesses->join(innerAccesses);
  propagateIfChanged(blockAccesses, result);
}

void AccessAnalysis::visitOperation(Operation *op) {
  auto result = ChangeResult::NoChange;
  auto *accesses = getOrCreate<AccessState>(op);

  TypeSwitch<Operation *>(op)
      .Case<StateReadOp>([&](auto readOp) {
        result |= accesses->add(readOp.getState(), AccessState::Read);
      })
      .Case<StateWriteOp>([&](auto writeOp) {
        result |= accesses->add(writeOp.getState(), AccessState::Write);
      })
      .Case<CallableOpInterface>([&](auto callableOp) {
        if (auto *region = callableOp.getCallableRegion()) {
          auto argResult = ChangeResult::NoChange;
          auto *argAccesses = getOrCreate<ArgumentAccessState>(op);
          for (auto &block : *region) {
            argResult |= argAccesses->join(
                *getOrCreateFor<ArgumentAccessState>(op, &block));
          }
          propagateIfChanged(argAccesses, argResult);
        }
      })
      .Case<CallOpInterface>([&](auto callOp) {
        auto calleeOp = dyn_cast_or_null<CallableOpInterface>(
            callOp.resolveCallable(&symbolTable));
        if (!calleeOp)
          return;
        auto operands = callOp.getArgOperands();
        const auto *calleeArgAccesses =
            getOrCreateFor<ArgumentAccessState>(callOp, calleeOp);
        for (auto access : calleeArgAccesses->accesses) {
          auto arg =
              access.getPointer().template dyn_cast_or_null<BlockArgument>();
          if (arg)
            result |=
                accesses->add(operands[arg.getArgNumber()], access.getInt());
        }
      });

  // Don't propagate inner state accesses through models, clock trees, and
  // passthrough ops.
  if (!isa<ModelOp, ClockTreeOp, PassThroughOp>(op)) {
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        const auto *blockAccesses = getOrCreateFor<AccessState>(op, &block);
        result |= accesses->join(*blockAccesses);
      }
    }
  }

  propagateIfChanged(accesses, result);
}

//===----------------------------------------------------------------------===//
// Legalization
//===----------------------------------------------------------------------===//

namespace {
struct Legalizer {
  Legalizer(DataFlowSolver &solver, AccessAnalysis &analysis)
      : solver(solver), analysis(analysis) {}
  LogicalResult run(MutableArrayRef<Region> regions);
  LogicalResult visitBlock(Block *block);

  DataFlowSolver &solver;
  AccessAnalysis &analysis;

  unsigned numLegalizedWrites = 0;
  unsigned numUpdatedReads = 0;

  /// A mapping from pre-existing states to temporary states for read
  /// operations, created during legalization to remove read-after-write
  /// hazards.
  DenseMap<Value, Value> legalizedStates;
};
} // namespace

LogicalResult Legalizer::run(MutableArrayRef<Region> regions) {
  for (auto &region : regions)
    for (auto &block : region)
      if (failed(visitBlock(&block)))
        return failure();
  assert(legalizedStates.empty() && "should be balanced within block");
  return success();
}

LogicalResult Legalizer::visitBlock(Block *block) {
  // In a first reverse pass over the block, find the first write that occurs
  // before the last read of a state, if any.
  SmallPtrSet<Value, 4> readStates;
  DenseMap<Value, Operation *> illegallyWrittenStates;
  for (Operation &op : llvm::reverse(*block)) {
    const auto *accesses = solver.lookupState<AccessState>(&op);
    if (!accesses)
      continue;

    // Determine the states written by this op for which we have already seen a
    // read earlier. These writes need to be legalized.
    SmallVector<Value, 1> affectedStates;
    for (auto access : accesses->accesses)
      if (access.getInt() == AccessState::Write)
        if (readStates.contains(access.getPointer()))
          illegallyWrittenStates[access.getPointer()] = &op;

    // Determine the states read by this op. This comes after handling of the
    // writes, such that a block that contains both reads and writes to a state
    // doesn't mark itself as illegal. Instead, we will descend into that block
    // further down and do a more fine-grained legalization.
    for (auto access : accesses->accesses)
      if (access.getInt() == AccessState::Read)
        readStates.insert(access.getPointer());
  }

  // Create a mapping from operations that create a read-after-write hazard to
  // the states that they modify. Don't consider states that have already been
  // legalized. This is important since we may have already created a temporary
  // in a parent block which we can just reuse.
  DenseMap<Operation *, SmallVector<Value, 1>> illegalWrites;
  for (auto [state, op] : illegallyWrittenStates)
    if (!legalizedStates.count(state))
      illegalWrites[op].push_back(state);

  // In a second forward pass over the block, insert the necessary temporary
  // state to legalize the writes and recur into subblocks while providing the
  // necessary rewrites.
  SmallVector<Value> locallyLegalizedStates;

  auto handleIllegalWrites =
      [&](Operation *op, SmallVector<Value, 1> &states) -> LogicalResult {
    LLVM_DEBUG(llvm::dbgs() << "Visiting illegal " << op->getName() << "\n");

    // Sort the states we need to legalize by a determinstic order established
    // during the access analysis. Without this the exact order in which states
    // were moved into a temporary would be non-deterministic.
    llvm::sort(states, [&](Value a, Value b) {
      return analysis.stateOrder[a] < analysis.stateOrder[b];
    });

    // Legalize each state individually.
    for (auto state : states) {
      LLVM_DEBUG(llvm::dbgs() << "- Legalizing " << state << "\n");

      // HACK: This is ugly, but we need a storage reference to allocate a state
      // into. Ideally we'd materialize this later on, but the current impl of
      // the alloc op requires a storage immediately. So try to find one.
      auto storage = TypeSwitch<Operation *, Value>(state.getDefiningOp())
                         .Case<AllocStateOp, RootInputOp, RootOutputOp>(
                             [&](auto allocOp) { return allocOp.getStorage(); })
                         .Default([](auto) { return Value{}; });
      if (!storage) {
        mlir::emitError(
            state.getLoc(),
            "cannot find storage pointer to allocate temporary into");
        return failure();
      }

      // Allocate a temporary state, read the current value of the state we are
      // legalizing, and write it to the temporary.
      ++numLegalizedWrites;
      ImplicitLocOpBuilder builder(state.getLoc(), op);
      auto tmpState =
          builder.create<AllocStateOp>(state.getType(), storage, nullptr);
      auto stateValue = builder.create<StateReadOp>(state);
      builder.create<StateWriteOp>(tmpState, stateValue, Value{});
      locallyLegalizedStates.push_back(state);
      legalizedStates.insert({state, tmpState});
    }
    return success();
  };

  for (Operation &op : *block) {
    if (isOpInteresting(&op)) {
      if (auto it = illegalWrites.find(&op); it != illegalWrites.end())
        if (failed(handleIllegalWrites(&op, it->second)))
          return failure();
    }
    // BUG: This is insufficient. Actually only reads should have their state
    // updated, since we want writes to still affect the original state. This
    // works for `state_read`, but in the case of a function that both reads and
    // writes a state we only have a single operand to change but we would need
    // one for reads and one for writes instead.
    // HACKY FIX: Assume that there is ever only a single write to a state. In
    // that case it is safe to assume that when an op is marked as writing a
    // state it wants the original state, not the temporary one for reads.
    const auto *accesses = solver.lookupState<AccessState>(&op);
    for (auto &operand : op.getOpOperands()) {
      if (accesses &&
          accesses->accesses.contains({operand.get(), AccessState::Read}) &&
          accesses->accesses.contains({operand.get(), AccessState::Write})) {
        auto d = op.emitWarning("operation reads and writes state; "
                                "legalization may be insufficient");
        d.attachNote()
            << "state update legalization does not properly handle operations "
               "that both read and write states at the same time; runtime data "
               "races between the read and write behavior are possible";
        d.attachNote(operand.get().getLoc()) << "state defined here:";
      }
      if (!accesses ||
          !accesses->accesses.contains({operand.get(), AccessState::Write})) {
        if (auto tmpState = legalizedStates.lookup(operand.get())) {
          operand.set(tmpState);
          ++numUpdatedReads;
        }
      }
    }
    for (auto &region : op.getRegions())
      for (auto &block : region)
        if (failed(visitBlock(&block)))
          return failure();
  }

  // Since we're leaving this block's scope, remove all the locally-legalized
  // states which are no longer accessible outside.
  for (auto state : locallyLegalizedStates)
    legalizedStates.erase(state);
  return success();
}

static LogicalResult getAncestorOpsInCommonDominatorBlock(
    Operation *write, Operation **writeAncestor, Operation *read,
    Operation **readAncestor, DominanceInfo *domInfo) {
  Block *commonDominator =
      domInfo->findNearestCommonDominator(write->getBlock(), read->getBlock());
  if (!commonDominator)
    return write->emitOpError(
        "cannot find a common dominator block with all read operations");

  // Path from writeOp to commmon dominator must only contain IfOps with no
  // return values
  Operation *writeParent = write;
  while (writeParent->getBlock() != commonDominator) {
    if (!isa<scf::IfOp, ClockTreeOp>(writeParent->getParentOp()))
      return write->emitOpError("memory write operations in arbitrarily nested "
                                "regions not supported");
    writeParent = writeParent->getParentOp();
  }
  Operation *readParent = read;
  while (readParent->getBlock() != commonDominator)
    readParent = readParent->getParentOp();

  *writeAncestor = writeParent;
  *readAncestor = readParent;
  return success();
}

static LogicalResult
moveMemoryWritesAfterLastRead(Region &region, const DenseSet<Value> &memories,
                              DominanceInfo *domInfo) {
  // Collect memory values and their reads
  DenseMap<Value, SetVector<Operation *>> readOps;
  auto result = region.walk([&](Operation *op) {
    if (isa<MemoryWriteOp>(op))
      return WalkResult::advance();
    SmallVector<Value> memoriesReadFrom;
    if (auto readOp = dyn_cast<MemoryReadOp>(op)) {
      memoriesReadFrom.push_back(readOp.getMemory());
    } else {
      for (auto operand : op->getOperands())
        if (isa<MemoryType>(operand.getType()))
          memoriesReadFrom.push_back(operand);
    }
    for (auto memVal : memoriesReadFrom) {
      if (!memories.contains(memVal))
        return op->emitOpError("uses memory value not directly defined by a "
                               "arc.alloc_memory operation"),
               WalkResult::interrupt();
      readOps[memVal].insert(op);
    }

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();

  // Collect all writes
  SmallVector<MemoryWriteOp> writes;
  region.walk([&](MemoryWriteOp writeOp) { writes.push_back(writeOp); });

  // Move the writes
  for (auto writeOp : writes) {
    if (!memories.contains(writeOp.getMemory()))
      return writeOp->emitOpError("uses memory value not directly defined by a "
                                  "arc.alloc_memory operation");
    for (auto *readOp : readOps[writeOp.getMemory()]) {
      // (1) If the last read and the write are in the same block, just move the
      // write after the read.
      // (2) If the write is directly in the clock tree region and the last read
      // in some nested region, move the write after the operation with the
      // nested region. (3) If the write is nested in if-statements (arbitrarily
      // deep) without return value, move the whole if operation after the last
      // read or the operation that defines the region if the read is inside a
      // nested region. (4) Number (3) may move more memory operations with the
      // write op, thus messing up the order of previously moved memory writes,
      // we check in a second walk-through if that is the case and just emit an
      // error for now. We could instead move reads in a parent region, split if
      // operations such that the memory write has its own, etc. Alternatively,
      // rewrite this to insert temporaries which is more difficult for memories
      // than simple states because the memory addresses have to be considered
      // (we cannot just copy the whole memory each time).
      Operation *readAncestor, *writeAncestor;
      if (failed(getAncestorOpsInCommonDominatorBlock(
              writeOp, &writeAncestor, readOp, &readAncestor, domInfo)))
        return failure();
      // FIXME: the 'isBeforeInBlock` + 'moveAfter' compination can be
      // computationally very expensive.
      if (writeAncestor->isBeforeInBlock(readAncestor))
        writeAncestor->moveAfter(readAncestor);
    }
  }

  // Double check that all writes happen after all reads to the same memory.
  for (auto writeOp : writes) {
    for (auto *readOp : readOps[writeOp.getMemory()]) {
      Operation *readAncestor, *writeAncestor;
      if (failed(getAncestorOpsInCommonDominatorBlock(
              writeOp, &writeAncestor, readOp, &readAncestor, domInfo)))
        return failure();

      if (writeAncestor->isBeforeInBlock(readAncestor))
        return writeOp
                   ->emitOpError("could not be moved to be after all reads to "
                                 "the same memory")
                   .attachNote(readOp->getLoc())
               << "could not be moved after this read";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LegalizeStateUpdatePass
    : public LegalizeStateUpdateBase<LegalizeStateUpdatePass> {
  LegalizeStateUpdatePass() = default;
  LegalizeStateUpdatePass(const LegalizeStateUpdatePass &pass)
      : LegalizeStateUpdatePass() {}

  void runOnOperation() override;

  Statistic numLegalizedWrites{
      this, "legalized-writes",
      "Writes that required temporary state for later reads"};
  Statistic numUpdatedReads{this, "updated-reads", "Reads that were updated"};
};
} // namespace

void LegalizeStateUpdatePass::runOnOperation() {
  auto module = getOperation();
  auto *domInfo = &getAnalysis<DominanceInfo>();

  for (auto model : module.getOps<ModelOp>()) {
    DenseSet<Value> memories;
    for (auto memOp : model.getOps<AllocMemoryOp>())
      memories.insert(memOp.getResult());
    for (auto ct : model.getOps<ClockTreeOp>())
      if (failed(
              moveMemoryWritesAfterLastRead(ct.getBody(), memories, domInfo)))
        return signalPassFailure();
  }

  DataFlowSolver solver;
  auto &analysis = *solver.load<AccessAnalysis>();
  if (failed(solver.initializeAndRun(module)))
    return signalPassFailure();

  Legalizer legalizer(solver, analysis);
  if (failed(legalizer.run(module->getRegions())))
    return signalPassFailure();
  numLegalizedWrites += legalizer.numLegalizedWrites;
  numUpdatedReads += legalizer.numUpdatedReads;
}

std::unique_ptr<Pass> arc::createLegalizeStateUpdatePass() {
  return std::make_unique<LegalizeStateUpdatePass>();
}
