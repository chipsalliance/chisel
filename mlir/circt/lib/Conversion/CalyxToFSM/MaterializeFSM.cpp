//===- MaterializeCalyxToFSM.cpp - FSM Materialization Pass -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the FSM materialization pass.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "circt/Conversion/CalyxToFSM.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMGraph.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/STLExtras.h"

using namespace circt;
using namespace calyx;
using namespace mlir;
using namespace fsm;

namespace {

struct MaterializeCalyxToFSMPass
    : public MaterializeCalyxToFSMBase<MaterializeCalyxToFSMPass> {
  void runOnOperation() override;

  /// Assigns the 'fsm.output' operation of the provided 'state' to enabled the
  /// set of provided groups. If 'topLevelDone' is set, also asserts the
  /// top-level done signal.
  void assignStateOutputOperands(OpBuilder &b, StateOp stateOp,
                                 bool topLevelDone = false) {
    SmallVector<Value> outputOperands;
    auto &enabledGroups = stateEnables[stateOp];
    for (StringAttr group : referencedGroups)
      outputOperands.push_back(
          getOrCreateConstant(b, APInt(1, enabledGroups.contains(group))));

    assert(outputOperands.size() == machineOp.getNumArguments() - 1 &&
           "Expected exactly one value for each uniquely referenced group in "
           "this machine");
    outputOperands.push_back(getOrCreateConstant(b, APInt(1, topLevelDone)));
    stateOp.ensureOutput(b);
    auto outputOp = stateOp.getOutputOp();
    outputOp->setOperands(outputOperands);
  }

  /// Extends every `fsm.return` guard in the transitions of this state to also
  /// include the provided set of 'doneGuards'. 'doneGuards' is passed by value
  /// to allow the caller to provide additional done guards apart from group
  /// enable-generated guards.
  void assignStateTransitionGuard(OpBuilder &b, StateOp stateOp,
                                  SmallVector<Value> doneGuards = {}) {
    auto &enabledGroups = stateEnables[stateOp];
    for (auto groupIt : llvm::enumerate(referencedGroups))
      if (enabledGroups.contains(groupIt.value()))
        doneGuards.push_back(machineOp.getArgument(groupIt.index()));

    for (auto transition :
         stateOp.getTransitions().getOps<fsm::TransitionOp>()) {

      if (!transition.hasGuard() && doneGuards.empty())
        continue;
      transition.ensureGuard(b);
      auto guardOp = transition.getGuardReturn();
      llvm::SmallVector<Value> guards;
      llvm::append_range(guards, doneGuards);
      if (guardOp.getNumOperands() != 0)
        guards.push_back(guardOp.getOperand());

      if (guards.empty())
        continue;

      b.setInsertionPoint(guardOp);
      Value guardConjunction;
      if (guards.size() == 1)
        guardConjunction = guards.front();
      else
        guardConjunction =
            b.create<comb::AndOp>(transition.getLoc(), guards, false);
      guardOp.setOperand(guardConjunction);
    }
  }

  Value getOrCreateConstant(OpBuilder &b, APInt value) {
    auto it = constants.find(value);
    if (it != constants.end())
      return it->second;

    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(&machineOp.getBody().front());
    auto constantOp = b.create<hw::ConstantOp>(machineOp.getLoc(), value);
    constants[value] = constantOp;
    return constantOp;
  }

  /// Maintain a set of all groups referenced within this fsm.machine.
  /// Use a SetVector to ensure a deterministic ordering - strong assumptions
  /// are placed on the order of referenced groups wrt. the top-level I/O
  /// created for the group done/go signals.
  SetVector<StringAttr> referencedGroups;

  /// Maintain a relation between states and the groups which they enable.
  DenseMap<fsm::StateOp, DenseSet<StringAttr>> stateEnables;

  /// A handle to the machine under transformation.
  MachineOp machineOp;

  /// Constant cache.
  DenseMap<APInt, Value> constants;

  OpBuilder *b;

  FSMStateNode *entryState;
  FSMStateNode *exitState;

  // Walks the machine and gathers the set of referenced groups and SSA values.
  void walkMachine();

  // Creates the top-level group go/done I/O for the machine.
  void materializeGroupIO();

  // Add attributes to the machine op to indicate which in/out ports are
  // associated with group activations and which are additional inputs to the
  // FSM.
  void assignAttributes();
};

} // end anonymous namespace

void MaterializeCalyxToFSMPass::walkMachine() {
  // Walk the states of the machine and gather the relation between states and
  // the groups which they enable as well as the set of all enabled states.
  for (auto stateOp : machineOp.getOps<fsm::StateOp>()) {
    for (auto enableOp : llvm::make_early_inc_range(
             stateOp.getOutput().getOps<calyx::EnableOp>())) {
      auto groupName = enableOp.getGroupNameAttr().getAttr();
      stateEnables[stateOp].insert(groupName);
      referencedGroups.insert(groupName);
      // Erase the enable op now that we've recorded the information.
      enableOp.erase();
    }
  }
}

void MaterializeCalyxToFSMPass::materializeGroupIO() {
  // Materialize the top-level I/O ports of the fsm.machine. We add an in- and
  // output for every unique group referenced within the machine, as well as an
  // additional in- and output to represent the top-level "go" input and "done"
  // output ports.
  SmallVector<Type> ioTypes = SmallVector<Type>(
      referencedGroups.size() + /*top-level go/done*/ 1, b->getI1Type());
  size_t nGroups = ioTypes.size() - 1;
  machineOp.setType(b->getFunctionType(ioTypes, ioTypes));
  assert(machineOp.getBody().getNumArguments() == 0 &&
         "expected no inputs to the FSM");
  machineOp.getBody().addArguments(
      ioTypes, SmallVector<Location, 4>(ioTypes.size(), b->getUnknownLoc()));

  // Build output assignments and transition guards in every state. We here
  // assume that the ordering of states in referencedGroups is fixed and
  // deterministic, since it is used as an analogue for port I/O ordering.
  for (auto stateOp : machineOp.getOps<fsm::StateOp>()) {
    assignStateOutputOperands(*b, stateOp,
                              /*topLevelDone=*/false);
    assignStateTransitionGuard(*b, stateOp);
  }

  // Assign top-level go guard in the transition state.
  size_t topLevelGoIdx = nGroups;
  assignStateTransitionGuard(*b, entryState->getState(),
                             {machineOp.getArgument(topLevelGoIdx)});

  // Assign top-level done in the exit state.
  assignStateOutputOperands(*b, exitState->getState(),
                            /*topLevelDone=*/true);
}

void MaterializeCalyxToFSMPass::assignAttributes() {
  // sGroupDoneInputs is a mapping from group name to the index of the
  // corresponding done input port.
  llvm::SmallVector<NamedAttribute> groupDoneInputs;
  for (size_t i = 0; i < referencedGroups.size(); ++i)
    groupDoneInputs.push_back({referencedGroups[i], b->getI64IntegerAttr(i)});
  machineOp->setAttr(calyxToFSM::sGroupDoneInputs,
                     b->getDictionaryAttr(groupDoneInputs));

  // sGroupGoOutputs is a mapping from group name to the index of the
  // corresponding go output port.
  llvm::SmallVector<NamedAttribute> groupGoOutputs;
  for (size_t i = 0; i < referencedGroups.size(); ++i)
    groupGoOutputs.push_back({referencedGroups[i], b->getI64IntegerAttr(i)});
  machineOp->setAttr(calyxToFSM::sGroupGoOutputs,
                     b->getDictionaryAttr(groupGoOutputs));

  // Assign top level go/done attributes
  machineOp->setAttr(calyxToFSM::sFSMTopLevelGoIndex,
                     b->getI64IntegerAttr(referencedGroups.size()));
  machineOp->setAttr(calyxToFSM::sFSMTopLevelDoneIndex,
                     b->getI64IntegerAttr(referencedGroups.size()));
}

void MaterializeCalyxToFSMPass::runOnOperation() {
  ComponentOp component = getOperation();
  auto *ctx = &getContext();
  auto builder = OpBuilder(ctx);
  b = &builder;
  auto controlOp = component.getControlOp();
  machineOp =
      dyn_cast_or_null<fsm::MachineOp>(controlOp.getBodyBlock()->front());
  if (!machineOp) {
    controlOp.emitOpError()
        << "expected an 'fsm.machine' operation as the top-level operation "
           "within the control region of this component.";
    signalPassFailure();
    return;
  }

  // Ensure a well-formed FSM.
  auto graph = FSMGraph(machineOp);
  entryState = graph.lookup(b->getStringAttr(calyxToFSM::sEntryStateName));
  exitState = graph.lookup(b->getStringAttr(calyxToFSM::sExitStateName));

  if (!(entryState && exitState)) {
    machineOp.emitOpError()
        << "Expected an '" << calyxToFSM::sEntryStateName << "' and '"
        << calyxToFSM::sExitStateName << "' state to be present in the FSM.";
    signalPassFailure();
    return;
  }

  walkMachine();
  materializeGroupIO();
  assignAttributes();
}

std::unique_ptr<mlir::Pass> circt::createMaterializeCalyxToFSMPass() {
  return std::make_unique<MaterializeCalyxToFSMPass>();
}
