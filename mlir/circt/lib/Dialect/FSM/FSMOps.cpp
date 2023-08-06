//===- FSMOps.cpp - Implementation of FSM dialect operations --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FSM/FSMOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace circt;
using namespace fsm;

//===----------------------------------------------------------------------===//
// MachineOp
//===----------------------------------------------------------------------===//

void MachineOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                      StringRef initialStateName, FunctionType type,
                      ArrayRef<NamedAttribute> attrs,
                      ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(MachineOp::getFunctionTypeAttrName(state.name),
                     TypeAttr::get(type));
  state.addAttribute("initialState",
                     StringAttr::get(state.getContext(), initialStateName));
  state.attributes.append(attrs.begin(), attrs.end());
  Region *region = state.addRegion();
  Block *body = new Block();
  region->push_back(body);
  body->addArguments(
      type.getInputs(),
      SmallVector<Location, 4>(type.getNumInputs(), builder.getUnknownLoc()));

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs,
      /*resultAttrs=*/std::nullopt, MachineOp::getArgAttrsAttrName(state.name),
      MachineOp::getResAttrsAttrName(state.name));
}

/// Get the initial state of the machine.
StateOp MachineOp::getInitialStateOp() {
  return dyn_cast_or_null<StateOp>(lookupSymbol(getInitialState()));
}

StringAttr MachineOp::getArgName(size_t i) {
  if (auto args = getArgNames())
    return (*args)[i].cast<StringAttr>();
  else
    return StringAttr::get(getContext(), "in" + std::to_string(i));
}

StringAttr MachineOp::getResName(size_t i) {
  if (auto resNameAttrs = getResNames())
    return (*resNameAttrs)[i].cast<StringAttr>();
  else
    return StringAttr::get(getContext(), "out" + std::to_string(i));
}

/// Get the port information of the machine.
void MachineOp::getHWPortInfo(SmallVectorImpl<hw::PortInfo> &ports) {
  ports.clear();
  auto machineType = getFunctionType();
  for (unsigned i = 0, e = machineType.getNumInputs(); i < e; ++i) {
    hw::PortInfo port;
    port.name = getArgName(i);
    port.dir = circt::hw::ModulePort::Direction::Input;
    port.type = machineType.getInput(i);
    port.argNum = i;
    ports.push_back(port);
  }

  for (unsigned i = 0, e = machineType.getNumResults(); i < e; ++i) {
    hw::PortInfo port;
    port.name = getResName(i);
    port.dir = circt::hw::ModulePort::Direction::Output;
    port.type = machineType.getResult(i);
    port.argNum = i;
    ports.push_back(port);
  }
}

ParseResult MachineOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      MachineOp::getFunctionTypeAttrName(result.name), buildFuncType,
      MachineOp::getArgAttrsAttrName(result.name),
      MachineOp::getResAttrsAttrName(result.name));
}

void MachineOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

static LogicalResult compareTypes(Location loc, TypeRange rangeA,
                                  TypeRange rangeB) {
  if (rangeA.size() != rangeB.size())
    return emitError(loc) << "mismatch in number of types compared ("
                          << rangeA.size() << " != " << rangeB.size() << ")";

  size_t index = 0;
  for (auto zip : llvm::zip(rangeA, rangeB)) {
    auto typeA = std::get<0>(zip);
    auto typeB = std::get<1>(zip);
    if (typeA != typeB)
      return emitError(loc) << "type mismatch at index " << index << " ("
                            << typeA << " != " << typeB << ")";
    ++index;
  }

  return success();
}

LogicalResult MachineOp::verify() {
  // If this function is external there is nothing to do.
  if (isExternal())
    return success();

  // Verify that the argument list of the function and the arg list of the entry
  // block line up.  The trait already verified that the number of arguments is
  // the same between the signature and the block.
  if (failed(compareTypes(getLoc(), getArgumentTypes(),
                          front().getArgumentTypes())))
    return emitOpError(
        "entry block argument types must match the machine input types");

  // Verify that the machine only has one block terminated with OutputOp.
  if (!llvm::hasSingleElement(*this))
    return emitOpError("must only have a single block");

  // Verify that the initial state exists
  if (!getInitialStateOp())
    return emitOpError("initial state '" + getInitialState() +
                       "' was not defined in the machine");

  if (getArgNames() && getArgNames()->size() != getArgumentTypes().size())
    return emitOpError() << "number of machine arguments ("
                         << getArgumentTypes().size()
                         << ") does "
                            "not match the provided number "
                            "of argument names ("
                         << getArgNames()->size() << ")";

  if (getResNames() && getResNames()->size() != getResultTypes().size())
    return emitOpError() << "number of machine results ("
                         << getResultTypes().size()
                         << ") does "
                            "not match the provided number "
                            "of result names ("
                         << getResNames()->size() << ")";

  return success();
}

hw::ModulePortInfo MachineOp::getPortList() {
  SmallVector<hw::PortInfo> inputs, outputs;
  auto argNames = getArgNames();
  auto argTypes = getFunctionType().getInputs();
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    bool isInOut = false;
    auto type = argTypes[i];

    if (auto inout = type.dyn_cast<hw::InOutType>()) {
      isInOut = true;
      type = inout.getElementType();
    }

    auto direction = isInOut ? hw::ModulePort::Direction::InOut
                             : hw::ModulePort::Direction::Input;

    inputs.push_back(
        {{argNames ? (*argNames)[i].cast<StringAttr>()
                   : StringAttr::get(getContext(), Twine("input") + Twine(i)),
          type, direction},
         i,
         {},
         {}});
  }

  auto resultNames = getResNames();
  auto resultTypes = getFunctionType().getResults();
  for (unsigned i = 0, e = resultTypes.size(); i < e; ++i) {
    outputs.push_back(
        {{resultNames
              ? (*resultNames)[i].cast<StringAttr>()
              : StringAttr::get(getContext(), Twine("output") + Twine(i)),
          resultTypes[i], hw::ModulePort::Direction::Output},
         i,
         {},
         {}});
  }
  return hw::ModulePortInfo(inputs, outputs);
}

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

/// Lookup the machine for the symbol.  This returns null on invalid IR.
MachineOp InstanceOp::getMachineOp() {
  auto module = (*this)->getParentOfType<ModuleOp>();
  return module.lookupSymbol<MachineOp>(getMachine());
}

LogicalResult InstanceOp::verify() {
  auto m = getMachineOp();
  if (!m)
    return emitError("cannot find machine definition '") << getMachine() << "'";

  return success();
}

void InstanceOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getInstance(), getSymName());
}

//===----------------------------------------------------------------------===//
// TriggerOp
//===----------------------------------------------------------------------===//

template <typename OpType>
static LogicalResult verifyCallerTypes(OpType op) {
  auto machine = op.getMachineOp();
  if (!machine)
    return op.emitError("cannot find machine definition");

  // Check operand types first.
  if (failed(compareTypes(op.getLoc(), machine.getArgumentTypes(),
                          op.getInputs().getTypes()))) {
    auto diag =
        op.emitOpError("operand types must match the machine input types");
    diag.attachNote(machine->getLoc()) << "original machine declared here";
    return failure();
  }

  // Check result types.
  if (failed(compareTypes(op.getLoc(), machine.getResultTypes(),
                          op.getOutputs().getTypes()))) {
    auto diag =
        op.emitOpError("result types must match the machine output types");
    diag.attachNote(machine->getLoc()) << "original machine declared here";
    return failure();
  }

  return success();
}

/// Lookup the machine for the symbol.  This returns null on invalid IR.
MachineOp TriggerOp::getMachineOp() {
  auto instanceOp = getInstance().getDefiningOp<InstanceOp>();
  if (!instanceOp)
    return nullptr;

  return instanceOp.getMachineOp();
}

LogicalResult TriggerOp::verify() { return verifyCallerTypes(*this); }

//===----------------------------------------------------------------------===//
// HWInstanceOp
//===----------------------------------------------------------------------===//

// HWInstanceLike interface
StringRef HWInstanceOp::getInstanceName() { return getSymName(); }

StringAttr HWInstanceOp::getInstanceNameAttr() { return getSymNameAttr(); }

Operation *HWInstanceOp::getReferencedModule() { return getMachineOp(); }

/// Lookup the machine for the symbol.  This returns null on invalid IR.
MachineOp HWInstanceOp::getMachineOp() {
  auto module = (*this)->getParentOfType<ModuleOp>();
  return module.lookupSymbol<MachineOp>(getMachine());
}

LogicalResult HWInstanceOp::verify() { return verifyCallerTypes(*this); }

hw::ModulePortInfo HWInstanceOp::getPortList() {
  return getMachineOp().getPortList();
}

//===----------------------------------------------------------------------===//
// StateOp
//===----------------------------------------------------------------------===//

void StateOp::build(OpBuilder &builder, OperationState &state,
                    StringRef stateName) {
  OpBuilder::InsertionGuard guard(builder);
  Region *output = state.addRegion();
  output->push_back(new Block());
  builder.setInsertionPointToEnd(&output->back());
  builder.create<fsm::OutputOp>(state.location);
  Region *transitions = state.addRegion();
  transitions->push_back(new Block());
  state.addAttribute("sym_name", builder.getStringAttr(stateName));
}

SetVector<StateOp> StateOp::getNextStates() {
  SmallVector<StateOp> nextStates;
  llvm::transform(
      getTransitions().getOps<TransitionOp>(),
      std::inserter(nextStates, nextStates.begin()),
      [](TransitionOp transition) { return transition.getNextStateOp(); });
  return SetVector<StateOp>(nextStates.begin(), nextStates.end());
}

LogicalResult StateOp::canonicalize(StateOp op, PatternRewriter &rewriter) {
  bool hasAlwaysTakenTransition = false;
  SmallVector<TransitionOp, 4> transitionsToErase;
  // Remove all transitions after an "always-taken" transition.
  for (auto transition : op.getTransitions().getOps<TransitionOp>()) {
    if (!hasAlwaysTakenTransition)
      hasAlwaysTakenTransition = transition.isAlwaysTaken();
    else
      transitionsToErase.push_back(transition);
  }

  for (auto transition : transitionsToErase)
    rewriter.eraseOp(transition);

  return failure(transitionsToErase.empty());
}

LogicalResult StateOp::verify() {
  MachineOp parent = getOperation()->getParentOfType<MachineOp>();

  if (parent.getNumResults() != 0 && (getOutput().empty()))
    return emitOpError("state must have a non-empty output region when the "
                       "machine has results.");

  if (!getOutput().empty()) {
    // Ensure that the output block has a single OutputOp terminator.
    Block *outputBlock = &getOutput().front();
    if (outputBlock->empty() || !isa<fsm::OutputOp>(outputBlock->back()))
      return emitOpError("output block must have a single OutputOp terminator");
  }

  return success();
}

Block *StateOp::ensureOutput(OpBuilder &builder) {
  if (getOutput().empty()) {
    OpBuilder::InsertionGuard g(builder);
    auto *block = new Block();
    getOutput().push_back(block);
    builder.setInsertionPointToStart(block);
    builder.create<fsm::OutputOp>(getLoc());
  }
  return &getOutput().front();
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

LogicalResult OutputOp::verify() {
  if ((*this)->getParentRegion() ==
      &(*this)->getParentOfType<StateOp>().getTransitions()) {
    if (getNumOperands() != 0)
      emitOpError("transitions region must not output any value");
    return success();
  }

  // Verify that the result list of the machine and the operand list of the
  // OutputOp line up.
  auto machine = (*this)->getParentOfType<MachineOp>();
  if (failed(
          compareTypes(getLoc(), machine.getResultTypes(), getOperandTypes())))
    return emitOpError("operand types must match the machine output types");

  return success();
}

//===----------------------------------------------------------------------===//
// TransitionOp
//===----------------------------------------------------------------------===//

void TransitionOp::build(OpBuilder &builder, OperationState &state,
                         StringRef nextState) {
  state.addRegion(); // guard
  state.addRegion(); // action
  state.addAttribute("nextState",
                     FlatSymbolRefAttr::get(builder.getStringAttr(nextState)));
}

void TransitionOp::build(OpBuilder &builder, OperationState &state,
                         StateOp nextState) {
  build(builder, state, nextState.getName());
}

Block *TransitionOp::ensureGuard(OpBuilder &builder) {
  if (getGuard().empty()) {
    OpBuilder::InsertionGuard g(builder);
    auto *block = new Block();
    getGuard().push_back(block);
    builder.setInsertionPointToStart(block);
    builder.create<fsm::ReturnOp>(getLoc());
  }
  return &getGuard().front();
}

Block *TransitionOp::ensureAction(OpBuilder &builder) {
  if (getAction().empty())
    getAction().push_back(new Block());
  return &getAction().front();
}

/// Lookup the next state for the symbol. This returns null on invalid IR.
StateOp TransitionOp::getNextStateOp() {
  auto machineOp = (*this)->getParentOfType<MachineOp>();
  if (!machineOp)
    return nullptr;

  return machineOp.lookupSymbol<StateOp>(getNextState());
}

bool TransitionOp::isAlwaysTaken() {
  if (!hasGuard())
    return true;

  auto guardReturn = getGuardReturn();
  if (guardReturn.getNumOperands() == 0)
    return true;

  if (auto constantOp =
          guardReturn.getOperand().getDefiningOp<mlir::arith::ConstantOp>())
    return constantOp.getValue().cast<BoolAttr>().getValue();

  return false;
}

LogicalResult TransitionOp::canonicalize(TransitionOp op,
                                         PatternRewriter &rewriter) {
  if (op.hasGuard()) {
    auto guardReturn = op.getGuardReturn();
    if (guardReturn.getNumOperands() == 1)
      if (auto constantOp = guardReturn.getOperand()
                                .getDefiningOp<mlir::arith::ConstantOp>()) {
        // Simplify when the guard region returns a constant value.
        if (constantOp.getValue().cast<BoolAttr>().getValue()) {
          // Replace the original return op with a new one without any operands
          // if the constant is TRUE.
          rewriter.setInsertionPoint(guardReturn);
          rewriter.create<fsm::ReturnOp>(guardReturn.getLoc());
          rewriter.eraseOp(guardReturn);
        } else {
          // Erase the whole transition op if the constant is FALSE, because the
          // transition will never be taken.
          rewriter.eraseOp(op);
        }
        return success();
      }
  }

  return failure();
}

LogicalResult TransitionOp::verify() {
  if (!getNextStateOp())
    return emitOpError("cannot find the definition of the next state `")
           << getNextState() << "`";

  // Verify the action region, if present.
  if (hasGuard()) {
    if (getGuard().front().empty() ||
        !isa_and_nonnull<fsm::ReturnOp>(&getGuard().front().back()))
      return emitOpError("guard region must terminate with a ReturnOp");
  }

  // Verify the transition is located in the correct region.
  if ((*this)->getParentRegion() != &getCurrentState().getTransitions())
    return emitOpError("must only be located in the transitions region");

  return success();
}

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

void VariableOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getName());
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

void ReturnOp::setOperand(Value value) {
  if (getOperand())
    getOperation()->setOperand(0, value);
  else
    getOperation()->insertOperands(0, {value});
}

//===----------------------------------------------------------------------===//
// UpdateOp
//===----------------------------------------------------------------------===//

/// Get the targeted variable operation. This returns null on invalid IR.
VariableOp UpdateOp::getVariableOp() {
  return getVariable().getDefiningOp<VariableOp>();
}

LogicalResult UpdateOp::verify() {
  if (!getVariable())
    return emitOpError("destination is not a variable operation");

  if (!(*this)->getParentOfType<TransitionOp>().getAction().isAncestor(
          (*this)->getParentRegion()))
    return emitOpError("must only be located in the action region");

  auto transition = (*this)->getParentOfType<TransitionOp>();
  for (auto otherUpdateOp : transition.getAction().getOps<UpdateOp>()) {
    if (otherUpdateOp == *this)
      continue;
    if (otherUpdateOp.getVariable() == getVariable())
      return otherUpdateOp.emitOpError(
          "multiple updates to the same variable within a single action region "
          "is disallowed");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/FSM/FSM.cpp.inc"
#undef GET_OP_CLASSES

#include "circt/Dialect/FSM/FSMDialect.cpp.inc"
