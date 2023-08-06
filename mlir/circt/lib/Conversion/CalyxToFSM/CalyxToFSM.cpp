//===- CalyxToFSM.cpp - Calyx to FSM conversion pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main Calyx control to FSM Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CalyxToFSM.h"
#include "../PassDetail.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMDialect.h"
#include "circt/Dialect/FSM/FSMGraph.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Support/Namespace.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace calyx;
using namespace fsm;
using namespace sv;

namespace {

class CompileFSMVisitor {
public:
  CompileFSMVisitor(SymbolCache &sc, FSMGraph &graph)
      : graph(graph), sc(sc), ctx(graph.getMachine().getContext()),
        builder(graph.getMachine().getContext()) {
    ns.add(sc);
  }

  /// Lowers the provided 'op' into a new FSM StateOp.
  LogicalResult dispatch(StateOp currentState, Operation *op,
                         StateOp nextState) {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .template Case<SeqOp, EnableOp, IfOp, WhileOp>(
            [&](auto opNode) { return visit(currentState, opNode, nextState); })
        .Default([&](auto) {
          return op->emitError() << "Operation '" << op->getName()
                                 << "' not supported for FSM compilation";
        });
  }

  ArrayRef<Attribute> getCompiledGroups() { return compiledGroups; }

private:
  /// Operation visitors;
  /// Apart from the visited operation, a visitor is provided with two extra
  /// arguments:
  /// currentState:
  ///   This represents a state which the callee has allocated to this visitor;
  ///   the visitor is free to use this state to its liking.
  /// nextState:
  ///   This represent the next state which this visitor eventually must
  ///   transition to.
  LogicalResult visit(StateOp currentState, SeqOp, StateOp nextState);
  LogicalResult visit(StateOp currentState, EnableOp, StateOp nextState);
  LogicalResult visit(StateOp currentState, IfOp, StateOp nextState);
  LogicalResult visit(StateOp currentState, WhileOp, StateOp nextState);

  /// Represents unique state name scopes generated from pushing states onto
  /// the state stack. The guard carries a unique name as well as managing the
  /// lifetime of suffixes on the state stack.
  struct StateScopeGuard {
  public:
    StateScopeGuard(CompileFSMVisitor &visitor, StringRef name,
                    StringRef suffix)
        : visitor(visitor), name(name) {
      visitor.stateStack.push_back(suffix.str());
    }
    ~StateScopeGuard() {
      assert(!visitor.stateStack.empty());
      visitor.stateStack.pop_back();
    }

    StringRef getName() { return name; }

  private:
    CompileFSMVisitor &visitor;
    std::string name;
  };

  /// Generates a new state name based on the current state stack and the
  /// provided suffix. The new suffix is pushed onto the state stack. Returns a
  /// guard object which pops the new suffix upon destruction.
  StateScopeGuard pushStateScope(StringRef suffix) {
    std::string name;
    llvm::raw_string_ostream ss(name);
    llvm::interleave(
        stateStack, ss, [&](const auto &it) { ss << it; }, "_");
    ss << "_" << suffix.str();
    return StateScopeGuard(*this, ns.newName(name), suffix);
  }

  FSMGraph &graph;
  SymbolCache &sc;
  MLIRContext *ctx;
  OpBuilder builder;
  Namespace ns;
  SmallVector<std::string, 4> stateStack;

  /// Maintain the set of compiled groups within this FSM, to pass Calyx
  /// verifiers.
  SmallVector<Attribute, 8> compiledGroups;
};

LogicalResult CompileFSMVisitor::visit(StateOp currentState, IfOp ifOp,
                                       StateOp nextState) {
  auto stateGuard = pushStateScope("if");
  auto loc = ifOp.getLoc();

  // Rename the current state now that we know it's an if header.
  graph.renameState(currentState, stateGuard.getName());

  auto lowerBranch = [&](Value cond, StringRef nextStateSuffix, bool invert,
                         Operation *innerBranchOp) {
    auto branchStateGuard = pushStateScope(nextStateSuffix);
    auto branchStateOp =
        graph.createState(builder, ifOp.getLoc(), branchStateGuard.getName())
            ->getState();

    auto transitionOp = graph
                            .createTransition(builder, ifOp.getLoc(),
                                              currentState, branchStateOp)
                            ->getTransition();
    transitionOp.ensureGuard(builder);
    fsm::ReturnOp returnOp = transitionOp.getGuardReturn();
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(&transitionOp.getGuard().front());
    Value branchTaken = cond;
    if (invert) {
      OpBuilder::InsertionGuard g(builder);
      branchTaken = comb::createOrFoldNot(loc, branchTaken, builder);
    }

    returnOp.setOperand(branchTaken);

    // Recurse into the body of the branch, with an exit state targeting
    // 'nextState'.
    if (failed(dispatch(branchStateOp, innerBranchOp, nextState)))
      return failure();
    return success();
  };

  // Then branch.
  if (failed(lowerBranch(ifOp.getCond(), "then", /*invert=*/false,
                         &ifOp.getThenBody()->front())))
    return failure();

  // Else branch.
  if (ifOp.elseBodyExists() &&
      failed(lowerBranch(ifOp.getCond(), "else", /*invert=*/true,
                         &ifOp.getElseBody()->front())))
    return failure();

  return success();
}

LogicalResult CompileFSMVisitor::visit(StateOp currentState, SeqOp seqOp,
                                       StateOp nextState) {
  Location loc = seqOp.getLoc();
  auto seqStateGuard = pushStateScope("seq");

  // Create a new state for each nested operation within this seqOp.
  auto &seqOps = seqOp.getBodyBlock()->getOperations();
  llvm::SmallVector<std::pair<Operation *, StateOp>> seqStates;

  // Iterate over the operations within the sequence. We do this in reverse
  // order to ensure that we always know the next state.
  StateOp currentOpNextState = nextState;
  int n = seqOps.size() - 1;
  for (auto &op : llvm::reverse(*seqOp.getBodyBlock())) {
    auto subStateGuard = pushStateScope(std::to_string(n--));
    auto thisStateOp =
        graph.createState(builder, op.getLoc(), subStateGuard.getName())
            ->getState();
    seqStates.insert(seqStates.begin(), {&op, thisStateOp});
    sc.addSymbol(thisStateOp);

    // Recurse into the current operation.
    if (failed(dispatch(thisStateOp, &op, currentOpNextState)))
      return failure();

    // This state is now the next state for the following operation.
    currentOpNextState = thisStateOp;
  }

  // Make 'currentState' transition directly the first state in the sequence.
  graph.createTransition(builder, loc, currentState, seqStates.front().second);

  return success();
}

LogicalResult CompileFSMVisitor::visit(StateOp currentState, WhileOp whileOp,
                                       StateOp nextState) {
  OpBuilder::InsertionGuard g(builder);
  auto whileStateGuard = pushStateScope("while");
  auto loc = whileOp.getLoc();

  // The current state is the while header (branch to whileOp or nextState).
  // Rename the current state now that we know it's a while header state.
  StateOp whileHeaderState = currentState;
  graph.renameState(whileHeaderState,
                    (whileStateGuard.getName() + "_header").str());
  sc.addSymbol(whileHeaderState);

  // Dispatch into the while body. The while body will always return to the
  // header.
  auto whileBodyEntryState =
      graph
          .createState(builder, loc,
                       (whileStateGuard.getName() + "_entry").str())
          ->getState();
  sc.addSymbol(whileBodyEntryState);
  Operation *whileBodyOp = &whileOp.getBodyBlock()->front();
  if (failed(dispatch(whileBodyEntryState, whileBodyOp, whileHeaderState)))
    return failure();

  // Create transitions to either the while body or the next state based on the
  // while condition.
  auto bodyTransition =
      graph
          .createTransition(builder, loc, whileHeaderState, whileBodyEntryState)
          ->getTransition();
  auto nextStateTransition =
      graph.createTransition(builder, loc, whileHeaderState, nextState)
          ->getTransition();

  bodyTransition.ensureGuard(builder);
  bodyTransition.getGuardReturn().setOperand(whileOp.getCond());
  nextStateTransition.ensureGuard(builder);
  builder.setInsertionPoint(nextStateTransition.getGuardReturn());
  nextStateTransition.getGuardReturn().setOperand(
      comb::createOrFoldNot(loc, whileOp.getCond(), builder));
  return success();
}

LogicalResult CompileFSMVisitor::visit(StateOp currentState, EnableOp enableOp,
                                       StateOp nextState) {
  assert(currentState &&
         "Expected this enableOp to be nested into some provided state");

  // Rename the current state now that we know it's an enable state.
  auto enableStateGuard = pushStateScope(enableOp.getGroupName());
  graph.renameState(currentState, enableStateGuard.getName());

  // Create a new calyx.enable in the output state referencing the enabled
  // group. We create a new op here as opposed to moving the existing, to make
  // callers iterating over nested ops safer.
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(&currentState.getOutput().front());
  builder.create<calyx::EnableOp>(enableOp.getLoc(), enableOp.getGroupName());

  if (nextState)
    graph.createTransition(builder, enableOp.getLoc(), currentState, nextState);

  // Append this group to the set of compiled groups.
  compiledGroups.push_back(
      SymbolRefAttr::get(builder.getContext(), enableOp.getGroupName()));

  return success();
}

class CalyxToFSMPass : public CalyxToFSMBase<CalyxToFSMPass> {
public:
  void runOnOperation() override;
}; // end anonymous namespace

void CalyxToFSMPass::runOnOperation() {
  ComponentOp component = getOperation();
  OpBuilder builder(&getContext());
  auto ctrlOp = component.getControlOp();
  assert(ctrlOp.getBodyBlock()->getOperations().size() == 1 &&
         "Expected a single top-level operation in the schedule");
  Operation &topLevelCtrlOp = ctrlOp.getBodyBlock()->front();
  builder.setInsertionPoint(&topLevelCtrlOp);

  // Create a side-effect-only FSM (no inputs, no outputs) which will strictly
  // refer to the symbols and SSA values defined in the regions of the
  // ComponentOp. This makes for an intermediate step, which allows for
  // outlining the FSM (materializing FSM I/O) at a later point.
  auto machineName = ("control_" + component.getName()).str();
  auto funcType = FunctionType::get(&getContext(), {}, {});
  auto machine =
      builder.create<MachineOp>(ctrlOp.getLoc(), machineName,
                                /*initialState=*/"fsm_entry", funcType);
  auto graph = FSMGraph(machine);

  SymbolCache sc;
  sc.addDefinitions(machine);

  // Create entry and exit states
  auto entryState =
      graph.createState(builder, ctrlOp.getLoc(), calyxToFSM::sEntryStateName)
          ->getState();
  auto exitState =
      graph.createState(builder, ctrlOp.getLoc(), calyxToFSM::sExitStateName)
          ->getState();

  auto visitor = CompileFSMVisitor(sc, graph);
  if (failed(visitor.dispatch(entryState, &topLevelCtrlOp, exitState))) {
    signalPassFailure();
    return;
  }

  // Remove the top-level calyx control operation that we've now converted to an
  // FSM.
  topLevelCtrlOp.erase();

  // Add the set of compiled groups as an attribute to the fsm.
  machine->setAttr(
      "compiledGroups",
      ArrayAttr::get(builder.getContext(), visitor.getCompiledGroups()));
}

} // namespace

std::unique_ptr<mlir::Pass> circt::createCalyxToFSMPass() {
  return std::make_unique<CalyxToFSMPass>();
}
