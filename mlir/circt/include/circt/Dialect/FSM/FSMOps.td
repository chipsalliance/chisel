//===- FSMOps.td - Definition of FSM dialect operations -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FSM_FSMOPS_TD
#define CIRCT_DIALECT_FSM_FSMOPS_TD

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "circt/Dialect/HW/HWOpInterfaces.td"

def HasCustomSSAName : DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>;

def MachineOp : FSMOp<"machine", [
      FunctionOpInterface,
      Symbol, SymbolTable, NoTerminator,
      DeclareOpInterfaceMethods<HWModuleLike>]> {
  let summary = "Define a finite-state machine";
  let description = [{
    `fsm.machine` represents a finite-state machine, including a machine name,
    the type of machine state, and the types of inputs and outputs. This op also
    includes a `$body` region that contains internal variables and states.
  }];

  let arguments = (ins StrAttr:$sym_name, StrAttr:$initialState,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       OptionalAttr<StrArrayAttr>:$argNames,
                       OptionalAttr<StrArrayAttr>:$resNames);
  let regions = (region SizedRegion<1>:$body);

  let builders = [
    OpBuilder<(ins "StringRef":$name, "StringRef":$initialState,
      "FunctionType":$function_type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
      CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>
  ];

  let extraClassDeclaration = [{
    /// Get the initial state of the machine.
    StateOp getInitialStateOp();

    /// Get the port information of the machine.
    void getHWPortInfo(SmallVectorImpl<hw::PortInfo> &);

    /// Returns the number of states in this machhine.
    size_t getNumStates() {
      auto stateOps = getBody().getOps<fsm::StateOp>();
      return std::distance(stateOps.begin(), stateOps.end());
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Verify the type attribute of this function. Returns failure and emits
    /// an error if the attribute is invalid.
    LogicalResult verifyType() {
      auto type = getFunctionTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getFunctionTypeAttrName().getValue() +
                           "' attribute of function type");
      return success();
    }

    /// Return the name of the i'th input to this machine.
    StringAttr getArgName(size_t i);

    /// Return the name of the i'th result of this machine.
    StringAttr getResName(size_t i);

    //===------------------------------------------------------------------===//
    // PortList Methods
    //===------------------------------------------------------------------===//
    ::circt::hw::ModulePortInfo getPortList();
  }];

  let extraClassDefinition = [{
    size_t $cppClass::getNumPorts() {
      auto machineType = getFunctionType();
      return machineType.getNumInputs() + machineType.getNumResults();
    }

    circt::hw::InnerSymAttr $cppClass::getPortSymbolAttr(size_t portIndex) {
      for (NamedAttribute argAttr :
           mlir::function_interface_impl::getArgAttrs(*this, portIndex))
        if (auto sym = argAttr.getValue().dyn_cast<circt::hw::InnerSymAttr>())
          return sym;
      return circt::hw::InnerSymAttr();
    }
  }];

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def InstanceOp : FSMOp<"instance", [Symbol, HasCustomSSAName]> {
  let summary = "Create an instance of a state machine";
  let description = [{
    `fsm.instance` represents an instance of a state machine, including an
    instance name and a symbol reference of the machine.
  }];

  let arguments = (ins StrAttr:$sym_name, FlatSymbolRefAttr:$machine);
  let results = (outs InstanceType:$instance);

  let assemblyFormat = [{ $sym_name $machine attr-dict }];

  let extraClassDeclaration = [{
    /// Lookup the machine for the symbol. This returns null on invalid IR.
    MachineOp getMachineOp();

    //===------------------------------------------------------------------===//
    // PortList Methods
    //===------------------------------------------------------------------===//
    ::circt::hw::ModulePortInfo getPortList();
  }];

  let hasVerifier = 1;
}

def TriggerOp : FSMOp<"trigger", []> {
  let summary = "Trigger an instance";
  let description = [{
    `fsm.trigger` triggers a state machine instance. The inputs and outputs are
    correponding to the inputs and outputs of the referenced machine of the
    instance.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs, InstanceType:$instance);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $instance attr-dict `(` $inputs `)` `:` functional-type($inputs, $outputs)
  }];

  let extraClassDeclaration = [{
    /// Lookup the machine for the symbol. This returns null on invalid IR.
    MachineOp getMachineOp();
  }];

  let hasVerifier = 1;
}

def HWInstanceOp : FSMOp<"hw_instance", [
    Symbol,
    DeclareOpInterfaceMethods<HWInstanceLike>
  ]> {
  let summary = "Create a hardware-style instance of a state machine";
  let description = [{
    `fsm.hw_instance` represents a hardware-style instance of a state machine,
    including an instance name and a symbol reference of the machine. The inputs
    and outputs are correponding to the inputs and outputs of the referenced
    machine.
  }];

  let arguments = (ins StrAttr:$sym_name, FlatSymbolRefAttr:$machine,
                       Variadic<AnyType>:$inputs, I1:$clock, I1:$reset);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $sym_name $machine attr-dict `(` $inputs `)`
    `,` `clock` $clock `,` `reset` $reset `:` functional-type($inputs, $outputs)
  }];

  let extraClassDeclaration = [{
    /// Lookup the machine for the symbol. This returns null on invalid IR.
    MachineOp getMachineOp();

    /// Module name is the same as the machine name.
    StringRef getModuleName() {
      return getMachine();
    }
    FlatSymbolRefAttr getModuleNameAttr() {
      return getMachineAttr();
    }

    //===------------------------------------------------------------------===//
    // PortList Methods
    //===------------------------------------------------------------------===//
    ::circt::hw::ModulePortInfo getPortList();
  }];

  let hasVerifier = 1;
}

def StateOp : FSMOp<"state", [HasParent<"MachineOp">, Symbol, NoTerminator]> {
  let summary = "Define a state of a machine";
  let description = [{
    `fsm.state` represents a state of a state machine. This op includes an
    `$output` region with an `fsm.output` as terminator to define the machine
    outputs under this state. This op also includes a `transitions` region that
    contains all the transitions of this state.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region AnyRegion:$output, AnyRegion:$transitions);

  let hasCanonicalizeMethod = true;

  let assemblyFormat = [{
    $sym_name attr-dict (`output` $output^)? (`transitions` $transitions^)?
  }];

  let extraClassDeclaration = [{
    /// Returns all possible next states from this state.
    llvm::SetVector<StateOp> getNextStates();

    /// Returns the OutputOp of this state.
    OutputOp getOutputOp() {
      auto outputOp = getOutput().getOps<fsm::OutputOp>();
      assert(!outputOp.empty() &&
           "Expected an fsm.output op inside the state output region");
      return *outputOp.begin();
    }

    /// Ensures that this state has an output region with an output operation.
    /// Returns the inner block of the output region.
    Block* ensureOutput(OpBuilder& builder);
  }];

  let builders = [
    OpBuilder<(ins "StringRef":$stateName)>
  ];

  let skipDefaultBuilders = 1;
  let hasVerifier = 1;
}

def OutputOp : FSMOp<"output", [HasParent<"StateOp">, ReturnLike, Terminator]> {
  let summary = "Output values from a state machine";
  let description = [{
    "fsm.output" represents the outputs of a machine under a specific state. The
    types of `$operands` should be consistent with the output types of the state
    machine.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [ OpBuilder<(ins), "build($_builder, $_state, std::nullopt);"> ];

  let assemblyFormat = [{ attr-dict ($operands^ `:` qualified(type($operands)))? }];

  let hasVerifier = 1;
}

def TransitionOp : FSMOp<"transition", [HasParent<"StateOp">, NoTerminator]> {
  let summary = "Define a transition of a state";
  let description = [{
    `fsm.transition` represents a transition of a state with a symbol reference
    of the next state. This op includes an optional `$guard` region with an `fsm.return`
    as terminator that returns a Boolean value indicating the guard condition of
    this transition. This op also includes an optional `$action` region that represents
    the actions to be executed when this transition is taken.
  }];

  let arguments = (ins FlatSymbolRefAttr:$nextState);
  let regions = (region AnyRegion:$guard, AnyRegion:$action);

  let hasCanonicalizeMethod = true;

  let assemblyFormat = [{
    $nextState attr-dict (`guard` $guard^)? (`action` $action^)?
  }];

  let extraClassDeclaration = [{
    /// Lookup the next state for the symbol. This returns null on invalid IR.
    StateOp getNextStateOp();

    /// Get the current state, this should never fail.
    StateOp getCurrentState() {
      return (*this)->getParentOfType<StateOp>();
    }

    /// Get the return operation of the guard region, this should never fail.
    ReturnOp getGuardReturn() {
      assert(!getGuard().empty() && "this transition has no guard");
      return cast<ReturnOp>(getGuard().front().getTerminator());
    }

    /// Return whether this transition has a guard.
    bool hasGuard() {
      return !getGuard().empty();
    }

    /// Return whether this transition has an action.
    bool hasAction() {
      return !getAction().empty();
    }

    /// Returns true if this transition is always taken.
    bool isAlwaysTaken();

    /// Ensures that this transition has a guard region.
    Block* ensureGuard(OpBuilder& builder);

    /// Ensures that this transition has an action region.
    Block* ensureAction(OpBuilder& builder);
  }];

  let builders = [
    OpBuilder<(ins "StringRef":$nextState)>,
    OpBuilder<(ins "fsm::StateOp":$nextState)>
  ];

  let skipDefaultBuilders = 1;
  let hasVerifier = 1;
}

def ReturnOp : FSMOp<"return", [HasParent<"TransitionOp">, ReturnLike,
      Terminator]> {
  let summary = "Return values from a region";
  let description = [{
    "fsm.return" marks the end of a region of `fsm.transition` and return
    values if the parent region is a `$guard` region.
  }];

  let extraClassDeclaration = [{
    // Assigns the guard operand to the provided value.
    void setOperand(Value operand);
  }];

  let arguments = (ins Optional<I1>:$operand);
  let builders = [ OpBuilder<(ins), "build($_builder, $_state, Value());"> ];
  let assemblyFormat = [{ attr-dict ($operand^)? }];
}

def VariableOp : FSMOp<"variable", [HasParent<"MachineOp">, HasCustomSSAName,
      FirstAttrDerivedResultType]> {
  let summary = "Declare a variable in a state machine";
  let description = [{
    `fsm.variable` represents an internal variable in a state machine with an
    initialization value.
  }];

  let arguments = (ins AnyAttr:$initValue, StrAttr:$name);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{ $name attr-dict `:` qualified(type($result)) }];
}

def UpdateOp : FSMOp<"update", [HasParent<"TransitionOp">, SameTypeOperands]> {
  let summary = "Update a variable in a state machine";
  let description = [{
    `fsm.update` updates the `$variable` with the `$value`. The definition op of
    `$variable` should be an `fsm.variable`. This op should *only* appear in the
    `action` region of a transtion.
  }];

  let arguments = (ins AnyType:$variable, AnyType:$value);

  let assemblyFormat = [{ attr-dict $variable `,` $value `:` qualified(type($value)) }];

  let extraClassDeclaration = [{
    /// Get the targeted variable operation. This returns null on invalid IR.
    VariableOp getVariableOp();
  }];

  let hasVerifier = 1;
}

#endif // CIRCT_DIALECT_FSM_FSMOPS_TD
