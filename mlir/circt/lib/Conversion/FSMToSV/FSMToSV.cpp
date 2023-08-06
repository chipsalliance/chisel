//===- FSMToSV.cpp - Convert FSM to HW and SV Dialect ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/FSMToSV.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/TypeSwitch.h"

#include <memory>
#include <variant>

using namespace mlir;
using namespace circt;
using namespace fsm;

/// Get the port info of a FSM machine. Clock and reset port are also added.
namespace {
struct ClkRstIdxs {
  size_t clockIdx;
  size_t resetIdx;
};
} // namespace
static ClkRstIdxs getMachinePortInfo(SmallVectorImpl<hw::PortInfo> &ports,
                                     MachineOp machine, OpBuilder &b) {
  // Get the port info of the machine inputs and outputs.
  machine.getHWPortInfo(ports);
  ClkRstIdxs specialPorts;

  // Add clock port.
  hw::PortInfo clock;
  clock.name = b.getStringAttr("clk");
  clock.dir = hw::ModulePort::Direction::Input;
  clock.type = b.getI1Type();
  clock.argNum = machine.getNumArguments();
  ports.push_back(clock);
  specialPorts.clockIdx = clock.argNum;

  // Add reset port.
  hw::PortInfo reset;
  reset.name = b.getStringAttr("rst");
  reset.dir = hw::ModulePort::Direction::Input;
  reset.type = b.getI1Type();
  reset.argNum = machine.getNumArguments() + 1;
  ports.push_back(reset);
  specialPorts.resetIdx = reset.argNum;

  return specialPorts;
}

// Clones constants implicitly captured by the region, into the region.
static void cloneConstantsIntoRegion(Region &region, OpBuilder &builder) {
  // Values implicitly captured by the region.
  llvm::SetVector<Value> captures;
  getUsedValuesDefinedAbove(region, region, captures);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&region.front());

  // Clone ConstantLike operations into the region.
  for (auto &capture : captures) {
    Operation *op = capture.getDefiningOp();
    if (!op || !op->hasTrait<OpTrait::ConstantLike>())
      continue;

    Operation *cloned = builder.clone(*op);
    for (auto [orig, replacement] :
         llvm::zip(op->getResults(), cloned->getResults()))
      replaceAllUsesInRegionWith(orig, replacement, region);
  }
}

namespace {

class StateEncoding {
  // An class for handling state encoding. The class is designed to
  // abstract away how states are selected in case patterns, referred to as
  // values, and used as selection signals for muxes.

public:
  StateEncoding(OpBuilder &b, hw::TypeScopeOp typeScope, MachineOp machine,
                hw::HWModuleOp hwModule);

  // Get the encoded value for a state.
  Value encode(StateOp state);
  // Get the state corresponding to an encoded value.
  StateOp decode(Value value);

  // Returns the type which encodes the state values.
  Type getStateType() { return stateType; }

  // Returns a case pattern which matches the provided state.
  std::unique_ptr<sv::CasePattern> getCasePattern(StateOp state);

protected:
  // Creates a constant value in the module for the given encoded state
  // and records the state value in the mappings. An inner symbol is
  // attached to the wire to avoid it being optimized away.
  // The constant can optionally be assigned behind a sv wire - doing so at this
  // point ensures that constants don't end up behind "_GEN#" wires in the
  // module.
  void setEncoding(StateOp state, Value v, bool wire = false);

  // A mapping between a StateOp and its corresponding encoded value.
  SmallDenseMap<StateOp, Value> stateToValue;

  // A mapping between an encoded value and its corresponding StateOp.
  SmallDenseMap<Value, StateOp> valueToState;

  // A mapping between an encoded value and the source value in the IR.
  SmallDenseMap<Value, Value> valueToSrcValue;

  // A typescope to emit the FSM enum type within.
  hw::TypeScopeOp typeScope;

  // The enum type for the states.
  Type stateType;

  OpBuilder &b;
  MachineOp machine;
  hw::HWModuleOp hwModule;
};

StateEncoding::StateEncoding(OpBuilder &b, hw::TypeScopeOp typeScope,
                             MachineOp machine, hw::HWModuleOp hwModule)
    : typeScope(typeScope), b(b), machine(machine), hwModule(hwModule) {
  Location loc = machine.getLoc();
  llvm::SmallVector<Attribute> stateNames;

  for (auto state : machine.getBody().getOps<StateOp>())
    stateNames.push_back(b.getStringAttr(state.getName()));

  // Create an enum typedef for the states.
  Type rawEnumType =
      hw::EnumType::get(b.getContext(), b.getArrayAttr(stateNames));

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(&typeScope.getBodyRegion().front());
  auto typedeclEnumType = b.create<hw::TypedeclOp>(
      loc, b.getStringAttr(hwModule.getName() + "_state_t"),
      TypeAttr::get(rawEnumType), nullptr);

  stateType = hw::TypeAliasType::get(
      SymbolRefAttr::get(typeScope.getSymNameAttr(),
                         {FlatSymbolRefAttr::get(typedeclEnumType)}),
      rawEnumType);

  // And create enum values for the states
  b.setInsertionPointToStart(&hwModule.getBody().front());
  for (auto state : machine.getBody().getOps<StateOp>()) {
    auto fieldAttr = hw::EnumFieldAttr::get(
        loc, b.getStringAttr(state.getName()), stateType);
    auto enumConstantOp = b.create<hw::EnumConstantOp>(
        loc, fieldAttr.getType().getValue(), fieldAttr);
    setEncoding(state, enumConstantOp,
                /*wire=*/true);
  }
}

// Get the encoded value for a state.
Value StateEncoding::encode(StateOp state) {
  auto it = stateToValue.find(state);
  assert(it != stateToValue.end() && "state not found");
  return it->second;
}
// Get the state corresponding to an encoded value.
StateOp StateEncoding::decode(Value value) {
  auto it = valueToState.find(value);
  assert(it != valueToState.end() && "encoded state not found");
  return it->second;
}

// Returns a case pattern which matches the provided state.
std::unique_ptr<sv::CasePattern> StateEncoding::getCasePattern(StateOp state) {
  // Get the field attribute for the state - fetch it through the encoding.
  auto fieldAttr =
      cast<hw::EnumConstantOp>(valueToSrcValue[encode(state)].getDefiningOp())
          .getFieldAttr();
  return std::make_unique<sv::CaseEnumPattern>(fieldAttr);
}

void StateEncoding::setEncoding(StateOp state, Value v, bool wire) {
  assert(stateToValue.find(state) == stateToValue.end() &&
         "state already encoded");

  Value encodedValue;
  if (wire) {
    auto loc = machine.getLoc();
    auto stateType = getStateType();
    auto stateEncodingWire = b.create<sv::RegOp>(
        loc, stateType, b.getStringAttr("to_" + state.getName()),
        hw::InnerSymAttr::get(state.getNameAttr()));
    b.create<sv::AssignOp>(loc, stateEncodingWire, v);
    encodedValue = b.create<sv::ReadInOutOp>(loc, stateEncodingWire);
  } else
    encodedValue = v;
  stateToValue[state] = encodedValue;
  valueToState[encodedValue] = state;
  valueToSrcValue[encodedValue] = v;
}

class MachineOpConverter {
public:
  MachineOpConverter(OpBuilder &builder, hw::TypeScopeOp typeScope,
                     MachineOp machineOp)
      : machineOp(machineOp), typeScope(typeScope), b(builder) {}

  // Converts the machine op to a hardware module.
  // 1. Creates a HWModuleOp for the machine op, with the same I/O as the FSM +
  // clk/reset ports.
  // 2. Creates a state register + encodings for the states visible in the
  // machine.
  // 3. Iterates over all states in the machine
  //  3.1. Moves all `comb` logic into the body of the HW module
  //  3.2. Records the SSA value(s) associated to the output ports in the state
  //  3.3. iterates of the transitions of the state
  //    3.3.1. Moves all `comb` logic in the transition guard/action regions to
  //            the body of the HW module.
  //    3.3.2. Creates a case pattern for the transition guard
  //  3.4. Creates a next-state value for the state based on the transition
  //  guards.
  // 4. Assigns next-state values for the states in a case statement on the
  // state reg.
  // 5. Assigns the current-state outputs for the states in a case statement
  // on the state reg.
  LogicalResult dispatch();

private:
  struct StateConversionResult {
    // Value of the next state output signal of the converted state.
    Value nextState;
    // Value of the output signals of the converted state.
    llvm::SmallVector<Value> outputs;
  };

  using StateConversionResults = DenseMap<StateOp, StateConversionResult>;

  // Converts a StateOp within this machine, and returns the value corresponding
  // to the next-state output of the op.
  FailureOr<StateConversionResult> convertState(StateOp state);

  // Converts the outgoing transitions of a state and returns the value
  // corresponding to the next-state output of the op.
  // Transitions are priority encoded in the order which they appear in the
  // state transition region.
  FailureOr<Value> convertTransitions(StateOp currentState,
                                      ArrayRef<TransitionOp> transitions);

  // Moves operations from 'block' into module scope, failing if any op were
  // deemed illegal. Returns the final op in the block if the op was a
  // terminator. An optional 'exclude' filer can be provided to dynamically
  // exclude some ops from being moved.
  FailureOr<Operation *>
  moveOps(Block *block,
          llvm::function_ref<bool(Operation *)> exclude = nullptr);

  struct CaseMuxItem;
  using StateCaseMapping =
      llvm::SmallDenseMap<StateOp,
                          std::variant<Value, std::shared_ptr<CaseMuxItem>>>;
  struct CaseMuxItem {
    // The target wire to be assigned.
    sv::RegOp wire;

    // The case select signal to be used.
    Value select;

    // A mapping between a state and an assignment within that state.
    // An assignment can either be a value or a nested CaseMuxItem. The latter
    // case will create nested case statements.
    StateCaseMapping assignmentInState;

    // An optional default value to be assigned before the case statement, if
    // the case is not fully specified for all states.
    std::optional<Value> defaultValue = {};
  };

  // Build an SV-based case mux for the given assignments. Assignments are
  // merged into the same case statement. Caller is expected to ensure that the
  // insertion point is within an `always_...` block.
  void buildStateCaseMux(llvm::MutableArrayRef<CaseMuxItem> assignments);

  // A handle to the state encoder for this machine.
  std::unique_ptr<StateEncoding> encoding;

  // A deterministic ordering of the states in this machine.
  llvm::SmallVector<StateOp> orderedStates;

  // A mapping from a fsm.variable op to its register.
  llvm::SmallDenseMap<VariableOp, seq::CompRegOp> variableToRegister;

  // A mapping from a state to variable updates performed during outgoing state
  // transitions.
  llvm::SmallDenseMap<
      /*currentState*/ StateOp,
      llvm::SmallDenseMap<
          /*targetState*/ StateOp,
          llvm::DenseMap</*targetVariable*/ VariableOp, /*targetValue*/ Value>>>
      stateToVariableUpdates;

  // A handle to the MachineOp being converted.
  MachineOp machineOp;

  // A handle to the HW ModuleOp being created.
  hw::HWModuleOp hwModuleOp;

  // A handle to the state register of the machine.
  seq::CompRegOp stateReg;

  // A typescope to emit the FSM enum type within.
  hw::TypeScopeOp typeScope;

  OpBuilder &b;
};

FailureOr<Operation *>
MachineOpConverter::moveOps(Block *block,
                            llvm::function_ref<bool(Operation *)> exclude) {
  for (auto &op : llvm::make_early_inc_range(*block)) {
    if (!isa<comb::CombDialect, hw::HWDialect, fsm::FSMDialect>(
            op.getDialect()))
      return op.emitOpError()
             << "is unsupported (op from the "
             << op.getDialect()->getNamespace() << " dialect).";

    if (exclude && exclude(&op))
      continue;

    if (op.hasTrait<OpTrait::IsTerminator>())
      return &op;

    op.moveBefore(&hwModuleOp.front(), b.getInsertionPoint());
  }
  return nullptr;
}

void MachineOpConverter::buildStateCaseMux(
    llvm::MutableArrayRef<CaseMuxItem> assignments) {

  // Gather the select signal. All assignments are expected to use the same
  // select signal.
  Value select = assignments.front().select;
  assert(llvm::all_of(
             assignments,
             [&](const CaseMuxItem &item) { return item.select == select; }) &&
         "All assignments must use the same select signal.");

  sv::CaseOp caseMux;
  // Default assignments.
  for (auto &assignment : assignments) {
    if (assignment.defaultValue)
      b.create<sv::BPAssignOp>(assignment.wire.getLoc(), assignment.wire,
                               *assignment.defaultValue);
  }

  // Case assignments.
  caseMux = b.create<sv::CaseOp>(
      machineOp.getLoc(), CaseStmtType::CaseStmt,
      /*sv::ValidationQualifierTypeEnum::ValidationQualifierUnique, */ select,
      /*numCases=*/machineOp.getNumStates() + 1, [&](size_t caseIdx) {
        // Make Verilator happy for sized enums.
        if (caseIdx == machineOp.getNumStates())
          return std::unique_ptr<sv::CasePattern>(
              new sv::CaseDefaultPattern(b.getContext()));
        StateOp state = orderedStates[caseIdx];
        return encoding->getCasePattern(state);
      });

  // Create case assignments.
  for (auto assignment : assignments) {
    OpBuilder::InsertionGuard g(b);
    for (auto [caseInfo, stateOp] :
         llvm::zip(caseMux.getCases(), orderedStates)) {
      auto assignmentInState = assignment.assignmentInState.find(stateOp);
      if (assignmentInState == assignment.assignmentInState.end())
        continue;
      b.setInsertionPointToEnd(caseInfo.block);
      if (auto v = std::get_if<Value>(&assignmentInState->second); v) {
        b.create<sv::BPAssignOp>(machineOp.getLoc(), assignment.wire, *v);
      } else {
        // Nested case statement.
        llvm::SmallVector<CaseMuxItem, 4> nestedAssignments;
        nestedAssignments.push_back(
            *std::get<std::shared_ptr<CaseMuxItem>>(assignmentInState->second));
        buildStateCaseMux(nestedAssignments);
      }
    }
  }
}

LogicalResult MachineOpConverter::dispatch() {
  b.setInsertionPoint(machineOp);
  auto loc = machineOp.getLoc();
  if (machineOp.getNumStates() < 2)
    return machineOp.emitOpError() << "expected at least 2 states.";

  // Clone all referenced constants into the machine body - constants may have
  // been moved to the machine parent due to the lack of IsolationFromAbove.
  cloneConstantsIntoRegion(machineOp.getBody(), b);

  // 1) Get the port info of the machine and create a new HW module for it.
  SmallVector<hw::PortInfo, 16> ports;
  auto clkRstIdxs = getMachinePortInfo(ports, machineOp, b);
  hwModuleOp = b.create<hw::HWModuleOp>(loc, machineOp.getSymNameAttr(), ports);
  b.setInsertionPointToStart(&hwModuleOp.front());

  // Replace all uses of the machine arguments with the arguments of the
  // new created HW module.
  for (auto args :
       llvm::zip(machineOp.getArguments(), hwModuleOp.front().getArguments())) {
    auto machineArg = std::get<0>(args);
    auto hwModuleArg = std::get<1>(args);
    machineArg.replaceAllUsesWith(hwModuleArg);
  }

  auto clock = hwModuleOp.front().getArgument(clkRstIdxs.clockIdx);
  auto reset = hwModuleOp.front().getArgument(clkRstIdxs.resetIdx);

  // 2) Build state and variable registers.
  encoding =
      std::make_unique<StateEncoding>(b, typeScope, machineOp, hwModuleOp);
  auto stateType = encoding->getStateType();

  auto nextStateWire =
      b.create<sv::RegOp>(loc, stateType, b.getStringAttr("state_next"));
  auto nextStateWireRead = b.create<sv::ReadInOutOp>(loc, nextStateWire);
  stateReg = b.create<seq::CompRegOp>(
      loc, stateType, nextStateWireRead, clock, "state_reg", reset,
      /*reset value=*/encoding->encode(machineOp.getInitialStateOp()), nullptr);

  llvm::DenseMap<VariableOp, sv::RegOp> variableNextStateWires;
  for (auto variableOp : machineOp.front().getOps<fsm::VariableOp>()) {
    auto initValueAttr = variableOp.getInitValueAttr().dyn_cast<IntegerAttr>();
    if (!initValueAttr)
      return variableOp.emitOpError() << "expected an integer attribute "
                                         "for the initial value.";
    Type varType = variableOp.getType();
    auto varLoc = variableOp.getLoc();
    auto varNextState = b.create<sv::RegOp>(
        varLoc, varType, b.getStringAttr(variableOp.getName() + "_next"));
    auto varResetVal = b.create<hw::ConstantOp>(varLoc, initValueAttr);
    auto variableReg = b.create<seq::CompRegOp>(
        varLoc, varType, b.create<sv::ReadInOutOp>(varLoc, varNextState), clock,
        b.getStringAttr(variableOp.getName() + "_reg"), reset, varResetVal,
        nullptr);
    variableToRegister[variableOp] = variableReg;
    variableNextStateWires[variableOp] = varNextState;
    // Postpone value replacement until all logic has been created.
    // fsm::UpdateOp's require their target variables to refer to a
    // fsm::VariableOp - if this is not the case, they'll throw an assert.
  }

  // Move any operations at the machine-level scope, excluding state ops, which
  // are handled separately.
  if (failed(moveOps(&machineOp.front(), [](Operation *op) {
        return isa<fsm::StateOp, fsm::VariableOp>(op);
      })))
    return failure();

  // 3) Convert states and record their next-state value assignments.
  StateCaseMapping nextStateFromState;
  StateConversionResults stateConvResults;
  for (auto state : machineOp.getBody().getOps<StateOp>()) {
    auto stateConvRes = convertState(state);
    if (failed(stateConvRes))
      return failure();

    stateConvResults[state] = *stateConvRes;
    orderedStates.push_back(state);
    nextStateFromState[state] = {stateConvRes->nextState};
  }

  // 4/5) Create next-state assignments for each output.
  llvm::SmallVector<CaseMuxItem, 4> outputCaseAssignments;
  for (size_t portIndex = 0; portIndex < machineOp.getNumResults();
       portIndex++) {
    auto outputPort = hwModuleOp.getOutputPort(portIndex);
    auto outputPortType = outputPort.type;
    CaseMuxItem outputAssignment;
    outputAssignment.wire = b.create<sv::RegOp>(
        machineOp.getLoc(), outputPortType,
        b.getStringAttr("output_" + std::to_string(portIndex)));
    outputAssignment.select = stateReg;
    for (auto &state : orderedStates)
      outputAssignment.assignmentInState[state] = {
          stateConvResults[state].outputs[portIndex]};

    outputCaseAssignments.push_back(outputAssignment);
  }

  // Create next-state maps for the FSM variables.
  llvm::DenseMap<VariableOp, CaseMuxItem> variableCaseMuxItems;
  for (auto &[currentState, it] : stateToVariableUpdates) {
    for (auto &[targetState, it2] : it) {
      for (auto &[variableOp, targetValue] : it2) {
        auto caseMuxItemIt = variableCaseMuxItems.find(variableOp);
        if (caseMuxItemIt == variableCaseMuxItems.end()) {
          // First time seeing this variable. Initialize the outer case
          // statement. The outer case has a default assignment to the current
          // value of the variable register.
          variableCaseMuxItems[variableOp];
          caseMuxItemIt = variableCaseMuxItems.find(variableOp);
          assert(variableOp);
          assert(variableNextStateWires.count(variableOp));
          caseMuxItemIt->second.wire = variableNextStateWires[variableOp];
          caseMuxItemIt->second.select = stateReg;
          caseMuxItemIt->second.defaultValue =
              variableToRegister[variableOp].getResult();
        }

        if (!std::get_if<std::shared_ptr<CaseMuxItem>>(
                &caseMuxItemIt->second.assignmentInState[currentState])) {
          // Initialize the inner case statement. This is an inner case within
          // the current state, switching on the next-state value.
          CaseMuxItem innerCaseMuxItem;
          innerCaseMuxItem.wire = caseMuxItemIt->second.wire;
          innerCaseMuxItem.select = nextStateWireRead;
          caseMuxItemIt->second.assignmentInState[currentState] = {
              std::make_shared<CaseMuxItem>(innerCaseMuxItem)};
        }

        // Append to the nested case mux for the variable, with a case select
        // on the next-state signal.
        // Append an assignment in the case that nextState == targetState.
        auto &innerCaseMuxItem = std::get<std::shared_ptr<CaseMuxItem>>(
            caseMuxItemIt->second.assignmentInState[currentState]);
        innerCaseMuxItem->assignmentInState[targetState] = {targetValue};
      }
    }
  }

  // Materialize the case mux.
  llvm::SmallVector<CaseMuxItem, 4> nextStateCaseAssignments;
  nextStateCaseAssignments.push_back(
      CaseMuxItem{nextStateWire, stateReg, nextStateFromState});
  for (auto &[_, caseMuxItem] : variableCaseMuxItems)
    nextStateCaseAssignments.push_back(caseMuxItem);
  nextStateCaseAssignments.append(outputCaseAssignments.begin(),
                                  outputCaseAssignments.end());

  {
    auto alwaysCombOp = b.create<sv::AlwaysCombOp>(loc);
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(alwaysCombOp.getBodyBlock());
    buildStateCaseMux(nextStateCaseAssignments);
  }

  // Replace variable values with their register counterparts.
  for (auto &[variableOp, variableReg] : variableToRegister)
    variableOp.getResult().replaceAllUsesWith(variableReg);

  // Assing output ports.
  llvm::SmallVector<Value> outputPortAssignments;
  for (auto outputAssignment : outputCaseAssignments)
    outputPortAssignments.push_back(
        b.create<sv::ReadInOutOp>(machineOp.getLoc(), outputAssignment.wire));

  // Delete the default created output op and replace it with the output
  // muxes.
  auto *oldOutputOp = hwModuleOp.front().getTerminator();
  b.create<hw::OutputOp>(loc, outputPortAssignments);
  oldOutputOp->erase();

  // Erase the original machine op.
  machineOp.erase();

  return success();
}

FailureOr<Value>
MachineOpConverter::convertTransitions( // NOLINT(misc-no-recursion)
    StateOp currentState, ArrayRef<TransitionOp> transitions) {
  Value nextState;
  if (transitions.empty()) {
    // Base case
    // State: transition to the current state.
    nextState = encoding->encode(currentState);
  } else {
    // Recursive case - transition to a named state.
    auto transition = cast<fsm::TransitionOp>(transitions.front());
    nextState = encoding->encode(transition.getNextStateOp());

    // Action conversion
    if (transition.hasAction()) {
      // Move any ops from the action region to the general scope, excluding
      // variable update ops.
      auto actionMoveOpsRes =
          moveOps(&transition.getAction().front(),
                  [](Operation *op) { return isa<fsm::UpdateOp>(op); });
      if (failed(actionMoveOpsRes))
        return failure();

      // Gather variable updates during the action.
      DenseMap<fsm::VariableOp, Value> variableUpdates;
      for (auto updateOp : transition.getAction().getOps<fsm::UpdateOp>()) {
        VariableOp variableOp = updateOp.getVariableOp();
        variableUpdates[variableOp] = updateOp.getValue();
      }

      stateToVariableUpdates[currentState][transition.getNextStateOp()] =
          variableUpdates;
    }

    // Guard conversion
    if (transition.hasGuard()) {
      // Not always taken; recurse and mux between the targeted next state and
      // the recursion result, selecting based on the provided guard.
      auto guardOpRes = moveOps(&transition.getGuard().front());
      if (failed(guardOpRes))
        return failure();

      auto guardOp = cast<ReturnOp>(*guardOpRes);
      assert(guardOp && "guard should be defined");
      auto guard = guardOp.getOperand();
      auto otherNextState =
          convertTransitions(currentState, transitions.drop_front());
      if (failed(otherNextState))
        return failure();
      comb::MuxOp nextStateMux = b.create<comb::MuxOp>(
          transition.getLoc(), guard, nextState, *otherNextState, false);
      nextState = nextStateMux;
    }
  }

  assert(nextState && "next state should be defined");
  return nextState;
}

FailureOr<MachineOpConverter::StateConversionResult>
MachineOpConverter::convertState(StateOp state) {
  MachineOpConverter::StateConversionResult res;

  // 3.1) Convert the output region by moving the operations into the module
  // scope and gathering the operands of the output op.
  if (!state.getOutput().empty()) {
    auto outputOpRes = moveOps(&state.getOutput().front());
    if (failed(outputOpRes))
      return failure();

    OutputOp outputOp = cast<fsm::OutputOp>(*outputOpRes);
    res.outputs = outputOp.getOperands(); // 3.2
  }

  auto transitions = llvm::SmallVector<TransitionOp>(
      state.getTransitions().getOps<TransitionOp>());
  // 3.3, 3.4) Convert the transitions and record the next-state value
  // derived from the transitions being selected in a priority-encoded manner.
  auto nextStateRes = convertTransitions(state, transitions);
  if (failed(nextStateRes))
    return failure();
  res.nextState = *nextStateRes;
  return res;
}

struct FSMToSVPass : public ConvertFSMToSVBase<FSMToSVPass> {
  void runOnOperation() override;
};

void FSMToSVPass::runOnOperation() {
  auto module = getOperation();
  auto b = OpBuilder(module);
  SmallVector<Operation *, 16> opToErase;

  // Create a typescope shared by all of the FSMs. This typescope will be
  // emitted in a single separate file to avoid polluting each output file with
  // typedefs.
  StringAttr typeScopeFilename = b.getStringAttr("fsm_enum_typedefs.sv");
  b.setInsertionPointToStart(module.getBody());
  auto typeScope = b.create<hw::TypeScopeOp>(
      module.getLoc(), b.getStringAttr("fsm_enum_typedecls"));
  typeScope.getBodyRegion().push_back(new Block());
  typeScope->setAttr(
      "output_file",
      hw::OutputFileAttr::get(typeScopeFilename,
                              /*excludeFromFileList*/ b.getBoolAttr(false),
                              /*includeReplicatedOps*/ b.getBoolAttr(false)));

  // Traverse all machines and convert.
  for (auto machine : llvm::make_early_inc_range(module.getOps<MachineOp>())) {
    MachineOpConverter converter(b, typeScope, machine);

    if (failed(converter.dispatch())) {
      signalPassFailure();
      return;
    }
  }

  // Traverse all machine instances and convert to hw instances.
  llvm::SmallVector<HWInstanceOp> instances;
  module.walk([&](HWInstanceOp instance) { instances.push_back(instance); });
  for (auto instance : instances) {
    auto fsmHWModule =
        module.lookupSymbol<hw::HWModuleOp>(instance.getMachine());
    assert(fsmHWModule &&
           "FSM machine should have been converted to a hw.module");

    b.setInsertionPoint(instance);
    llvm::SmallVector<Value, 4> operands;
    llvm::transform(instance.getOperands(), std::back_inserter(operands),
                    [&](auto operand) { return operand; });
    auto hwInstance = b.create<hw::InstanceOp>(
        instance.getLoc(), fsmHWModule, b.getStringAttr(instance.getName()),
        operands, nullptr);
    instance.replaceAllUsesWith(hwInstance);
    instance.erase();
  }

  if (typeScope.getBodyBlock()->empty()) {
    // If the typescope is empty (no FSMs were converted), erase it.
    typeScope.erase();
  } else {
    // Else, add an include file to the top-level (will include typescope
    // in all files).
    b.setInsertionPointToStart(module.getBody());
    b.create<sv::VerbatimOp>(
        module.getLoc(), "`include \"" + typeScopeFilename.getValue() + "\"");
  }
}

} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::createConvertFSMToSVPass() {
  return std::make_unique<FSMToSVPass>();
}
