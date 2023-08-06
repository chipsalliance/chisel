// RUN: circt-opt %s --split-input-file --verify-diagnostics

// expected-error @+1 {{entry block must have 3 arguments to match function signature}}
"systemc.module"() ({
  ^bb0(%arg0: !systemc.out<i4>, %arg1: !systemc.in<i32>, %arg2: !systemc.out<i4>, %arg3: !systemc.inout<i8>):
  }) {function_type = (!systemc.out<i4>, !systemc.in<i32>, !systemc.out<i4>) -> (), portNames = ["port0", "port1", "port2", "port3"], sym_name = "verifierTest"} : () -> ()

// -----

// expected-error @+1 {{incorrect number of port names}}
"systemc.module"() ({
  ^bb0(%arg0: !systemc.out<i4>, %arg1: !systemc.in<i32>, %arg2: !systemc.out<i4>, %arg3: !systemc.inout<i8>):
  }) {function_type = (!systemc.out<i4>, !systemc.in<i32>, !systemc.out<i4>, !systemc.inout<i8>) -> (), portNames = ["port0", "port1", "port2"], sym_name = "verifierTest"} : () -> ()

// -----

// expected-error @+1 {{incorrect number of function results (always has to be 0)}}
"systemc.module"() ({
  ^bb0(%arg0: !systemc.out<i4>, %arg1: !systemc.in<i32>, %arg2: !systemc.out<i4>, %arg3: !systemc.inout<i8>):
  }) {function_type = (!systemc.out<i4>, !systemc.in<i32>, !systemc.out<i4>, !systemc.inout<i8>) -> (i1), portNames = ["port0", "port1", "port2", "port3"], sym_name = "verifierTest"} : () -> ()

// -----

// expected-error @+1 {{port name must not be empty}}
"systemc.module"() ({
  ^bb0(%arg0: !systemc.out<i4>, %arg1: !systemc.in<i32>, %arg2: !systemc.out<i4>, %arg3: !systemc.inout<i8>):
  }) {function_type = (!systemc.out<i4>, !systemc.in<i32>, !systemc.out<i4>, !systemc.inout<i8>) -> (), portNames = ["port0", "port1", "port2", ""], sym_name = "verifierTest"} : () -> ()

// -----

// expected-note @+1 {{in module '@verifierTest'}}
"systemc.module"() ({
  // expected-error @+2 {{redefines name 'port2'}}
  // expected-note @+1 {{'port2' first defined here}}
  ^bb0(%arg0: !systemc.out<i4>, %arg1: !systemc.in<i32>, %arg2: !systemc.out<i4>, %arg3: !systemc.inout<i8>):
  }) {function_type = (!systemc.out<i4>, !systemc.in<i32>, !systemc.out<i4>, !systemc.inout<i8>) -> (), portNames = ["port0", "port1", "port2", "port2"], sym_name = "verifierTest"} : () -> ()

// -----

"systemc.module"() ({
  // expected-error @+1 {{module port must be of type 'sc_in', 'sc_out', or 'sc_inout'}}
  ^bb0(%arg0: i4):
  }) {function_type = (i4) -> (), portNames = ["port0"], sym_name = "verifierTest"} : () -> ()

// -----

// expected-error @+1 {{failed to satisfy constraint: string array attribute}}
"systemc.module"() ({
  ^bb0(%arg0: !systemc.in<i4>):
  }) {function_type = (!systemc.in<i4>) -> (), portNames = [i32], sym_name = "verifierTest"} : () -> ()

// -----

// expected-error @+1 {{attribute 'portNames' occurs more than once in the attribute list}}
systemc.module @noExplicitPortNamesAttr () attributes {portNames=["p1"]} {}

// -----

// expected-note @+1 {{in module '@signalNameConflict'}}
systemc.module @signalNameConflict () {
  // expected-note @+1 {{'signal0' first defined here}}
  %0 = "systemc.signal"() {name = "signal0"} : () -> !systemc.signal<i32>
  // expected-error @+1 {{redefines name 'signal0'}}
  %1 = "systemc.signal"() {name = "signal0"} : () -> !systemc.signal<i32>
}

// -----

// expected-note @+2 {{in module '@signalNameConflictWithArg'}}
// expected-note @+1 {{'in' first defined here}}
systemc.module @signalNameConflictWithArg (%in: !systemc.in<i32>) {
  // expected-error @+1 {{redefines name 'in'}}
  %0 = "systemc.signal"() {name = "in"} : () -> !systemc.signal<i32>
}

// -----

systemc.module @signalNameNotEmpty () {
  // expected-error @+1 {{'name' attribute must not be empty}}
  %0 = "systemc.signal"() {name = ""} : () -> !systemc.signal<i32>
}

// -----

systemc.module @moduleDoesNotAccessNameBeforeExistanceVerified () {
  // expected-error @+1 {{requires attribute 'name'}}
  %0 = "systemc.signal"() {} : () -> !systemc.signal<i32>
}

// -----

systemc.module @signalMustBeDirectChildOfModule () {
  systemc.ctor {
    // expected-error @+1 {{expects parent op 'systemc.module'}}
    %signal = systemc.signal : !systemc.signal<i32>
  }
}

// -----

systemc.module @ctorNoBlockArguments () {
  // expected-error @+1 {{op must not have any arguments}} 
  "systemc.ctor"() ({
    ^bb0(%arg0: i32):
    }) : () -> ()
}

// -----

systemc.module @destructorNoBlockArguments () {
  // expected-error @+1 {{op must not have any arguments}} 
  "systemc.cpp.destructor"() ({
    ^bb0(%arg0: i32):
    }) : () -> ()
}

// -----

systemc.module @funcNoBlockArguments () {
  // expected-error @+1 {{op must not have any arguments}} 
  %0 = "systemc.func"() ({
    ^bb0(%arg0: i32):
    }) {name="funcname"}: () -> (() -> ())
}

// -----

systemc.module @funcNoBlockArguments () {
  // expected-error @+1 {{result #0 must be FunctionType with no inputs and results, but got '(i32) -> ()'}}
  %0 = "systemc.func"() ({
    ^bb0():
    }) {name="funcname"}: () -> ((i32) -> ())
}

// -----

// expected-note @+1 {{in module '@signalFuncNameConflict'}}
systemc.module @signalFuncNameConflict () {
  // expected-note @+1 {{'name' first defined here}}
  %0 = "systemc.signal"() {name="name"} : () -> !systemc.signal<i32>
  // expected-error @+1 {{redefines name 'name'}}
  %1 = "systemc.func"() ({
    ^bb0:
    }) {name="name"}: () -> (() -> ())
}

// -----

systemc.module @cannotReadFromOutPort (%port0: !systemc.out<i32>) {
  // expected-error @+1 {{operand #0 must be a SystemC sc_in<T> type or a SystemC sc_inout<T> type or a SystemC sc_signal<T> type, but got '!systemc.out<i32>'}}
  %0 = systemc.signal.read %port0 : !systemc.out<i32>
}

// -----

systemc.module @inferredTypeDoesNotMatch (%port0: !systemc.in<i32>) {
  // expected-error @below {{failed to infer returned types}}
  // expected-error @+1 {{op inferred type(s) 'i32' are incompatible with return type(s) of operation 'i4'}}
  %0 = "systemc.signal.read"(%port0) : (!systemc.in<i32>) -> i4
}

// -----

systemc.module @cannotWriteToInputPort (%port0: !systemc.in<i32>) {
  %0 = hw.constant 0 : i32
  // expected-error @+1 {{'dest' must be a SystemC sc_out<T> type or a SystemC sc_inout<T> type or a SystemC sc_signal<T> type, but got '!systemc.in<i32>'}}
  systemc.signal.write %port0, %0 : !systemc.in<i32>
}

// -----

systemc.module @invalidSignalOpReturnType () {
  // expected-error @+1 {{result #0 must be a SystemC sc_signal<T> type, but got 'i32'}}
  %signal0 = systemc.signal : i32
}

// -----

systemc.module @m1() {}

// expected-note @+1 {{in module '@instanceDeclNameConflict'}}
systemc.module @instanceDeclNameConflict () {
  // expected-note @+1 {{'name' first defined here}}
  %0 = "systemc.signal"() {name="name"} : () -> !systemc.signal<i32>
  // expected-error @+1 {{redefines name 'name'}}
  %1 = "systemc.instance.decl"() {name="name", moduleName=@m1} : () -> !systemc.module<m1()>
}

// -----

systemc.module @m1() {}

systemc.module @instanceDeclNameNotEmpty () {
  // expected-error @+1 {{'name' attribute must not be empty}}
  %0 = "systemc.instance.decl"() {name = "", moduleName=@m1} : () -> !systemc.module<m1()>
}

// -----

systemc.module @submodule (%in0: !systemc.in<i32>) {}

systemc.module @instanceDeclMustBeDirectChildOfModule () {
  systemc.ctor {
    // expected-error @+1 {{expects parent op 'systemc.module'}}
    %instance = systemc.instance.decl @submodule : !systemc.module<submodule(in0: !systemc.in<i32>)>
  }
}

// -----

systemc.module @submodule (%in0: !systemc.in<i32>) {}

systemc.module @bindPortMustBeDirectChildOfCtor (%input0: !systemc.in<i32>) {
  %instance = systemc.instance.decl @submodule : !systemc.module<submodule(in0: !systemc.in<i32>)>
  // expected-error @+1 {{expects parent op 'systemc.ctor'}}
  systemc.instance.bind_port %instance["in0"] to %input0 : !systemc.module<submodule(in0: !systemc.in<i32>)>, !systemc.in<i32>
}

// -----

// expected-note @+1 {{module declared here}}
systemc.module @adder (%summand_a: !systemc.in<i32>, %summand_b: !systemc.in<i32>, %sum: !systemc.out<i32>) {}

systemc.module @instanceDeclTypeMismatch () {
  // expected-error @+1 {{port type #1 must be '!systemc.in<i32>', but got '!systemc.in<i1>'}}
  %moduleInstance = systemc.instance.decl @adder : !systemc.module<adder(summand_a: !systemc.in<i32>, summand_b: !systemc.in<i1>, sum: !systemc.out<i32>)>
}

// -----

// expected-note @+1 {{module declared here}}
systemc.module @adder (%summand_a: !systemc.in<i32>, %summand_b: !systemc.in<i32>, %sum: !systemc.out<i32>) {}

systemc.module @instanceDeclPortNameMismatch () {
  // expected-error @+1 {{port name #1 must be "summand_b", but got "summand"}}
  %moduleInstance = systemc.instance.decl @adder : !systemc.module<adder(summand_a: !systemc.in<i32>, summand: !systemc.in<i32>, sum: !systemc.out<i32>)>
}

// -----

// expected-note @+1 {{module declared here}}
systemc.module @adder (%summand_a: !systemc.in<i32>, %summand_b: !systemc.in<i32>, %sum: !systemc.out<i32>) {}

systemc.module @instanceDeclPortNumMismatch () {
  // expected-error @+1 {{has a wrong number of ports; expected 3 but got 2}}
  %moduleInstance = systemc.instance.decl @adder : !systemc.module<adder(summand_a: !systemc.in<i32>, summand_b: !systemc.in<i32>)>
}

// -----

systemc.module @instanceDeclNonExistentModule () {
  // expected-error @+1 {{cannot find module definition 'adder'}}
  %moduleInstance = systemc.instance.decl @adder : !systemc.module<adder(summand_a: !systemc.in<i32>, summand_b: !systemc.in<i32>)>
}

// -----

// expected-note @+1 {{module declared here}}
hw.module @adder () -> () {}

systemc.module @instanceDeclDoesNotReferenceSystemCModule () {
  // expected-error @+1 {{symbol reference 'adder' isn't a systemc module}}
  %moduleInstance = systemc.instance.decl @adder : !systemc.module<adder()>
}

// -----

// expected-note @+1 {{module declared here}}
systemc.module @adder (%summand_a: !systemc.in<i32>, %summand_b: !systemc.in<i32>, %sum: !systemc.out<i32>) {}

systemc.module @instanceDeclPortNumMismatch () {
  // expected-error @+1 {{module names must match; expected 'adder' but got 'wrongname'}}
  %moduleInstance = systemc.instance.decl @adder : !systemc.module<wrongname(summand_a: !systemc.in<i32>, summand_b: !systemc.in<i32>)>
}

// -----

systemc.module @submodule (%in0: !systemc.in<i32>) {}

systemc.module @invalidPortName (%input0: !systemc.in<i32>) {
  %instance = systemc.instance.decl @submodule : !systemc.module<submodule(in0: !systemc.in<i32>)>
  systemc.ctor {
    // expected-error @+1 {{port name "out" not found in module}}
    systemc.instance.bind_port %instance["out"] to %input0 : !systemc.module<submodule(in0: !systemc.in<i32>)>, !systemc.in<i32>
  }
}

// -----

systemc.module @submodule (%in0: !systemc.in<i32>) {}

systemc.module @directionOfPortAndChannelMismatch (%output0: !systemc.out<i32>) {
  %instance = systemc.instance.decl @submodule : !systemc.module<submodule(in0: !systemc.in<i32>)>
  systemc.ctor {
    // expected-error @+1 {{'!systemc.in<i32>' port cannot be bound to '!systemc.out<i32>' channel due to port direction mismatch}}
    systemc.instance.bind_port %instance["in0"] to %output0 : !systemc.module<submodule(in0: !systemc.in<i32>)>, !systemc.out<i32>
  }
}

// -----

systemc.module @submodule (%out0: !systemc.out<i32>) {}

systemc.module @directionOfPortAndChannelMismatch (%input0: !systemc.in<i32>) {
  %instance = systemc.instance.decl @submodule : !systemc.module<submodule(out0: !systemc.out<i32>)>
  systemc.ctor {
    // expected-error @+1 {{'!systemc.out<i32>' port cannot be bound to '!systemc.in<i32>' channel due to port direction mismatch}}
    systemc.instance.bind_port %instance["out0"] to %input0 : !systemc.module<submodule(out0: !systemc.out<i32>)>, !systemc.in<i32>
  }
}

// -----

systemc.module @submodule (%out0: !systemc.out<i32>) {}

systemc.module @baseTypeMismatch () {
  %signal = systemc.signal : !systemc.signal<i8>
  %instance = systemc.instance.decl @submodule : !systemc.module<submodule(out0: !systemc.out<i32>)>
  systemc.ctor {
    // expected-error @+1 {{'!systemc.out<i32>' port cannot be bound to '!systemc.signal<i8>' channel due to base type mismatch}}
    systemc.instance.bind_port %instance["out0"] to %signal : !systemc.module<submodule(out0: !systemc.out<i32>)>, !systemc.signal<i8>
  }
}

// -----

systemc.module @submodule (%out0: !systemc.out<i32>) {}

systemc.module @baseTypeMismatch (%out0: !systemc.out<i32>) {
  %instance = systemc.instance.decl @submodule : !systemc.module<submodule(out0: !systemc.out<i32>)>
  systemc.ctor {
    // expected-error @+1 {{expected a list of exactly 2 types, but got 1}}
    systemc.instance.bind_port %instance["out0"] to %out0 : !systemc.module<submodule(out0: !systemc.out<i32>)>
  }
}

// -----

systemc.module @submodule (%out0: !systemc.out<i32>) {}

systemc.module @baseTypeMismatch (%out0: !systemc.out<i32>) {
  %instance = systemc.instance.decl @submodule : !systemc.module<submodule(out0: !systemc.out<i32>)>
  systemc.ctor {
    // expected-error @+1 {{port #1 does not exist, there are only 1 ports}}
    "systemc.instance.bind_port"(%instance, %out0) {portId = 1 : index} : (!systemc.module<submodule(out0: !systemc.out<i32>)>, !systemc.out<i32>) -> ()
  }
}

// -----

systemc.module @assignOperandTypeMismatch () {
  %var = systemc.cpp.variable : i32
  systemc.ctor {
    %0 = hw.constant 0 : i8
    // expected-error @+1 {{requires all operands to have the same type}}
    "systemc.cpp.assign"(%var, %0) : (i32, i8) -> ()
  }
}

// -----

systemc.module @variableOperandTypeMismatch () {
  systemc.ctor {
    %0 = hw.constant 0 : i8
    // expected-error @+1 {{'init' and 'variable' must have the same type, but got 'i8' and 'i32'}}
    %1 = "systemc.cpp.variable"(%0) {name = "var"} : (i8) -> i32
  }
}

// -----

// expected-note @+1 {{in module '@variableNameCollision'}}
systemc.module @variableNameCollision () {
  systemc.ctor {
    // expected-note @+1 {{'var' first defined here}}
    %0 = "systemc.cpp.variable"() {name = "var"} : () -> i32
    // expected-error @+1 {{redefines name 'var'}}
    %1 = "systemc.cpp.variable"() {name = "var"} : () -> i32
  }
}

// -----

// expected-error @+1 {{unknown type `value_base` in dialect `systemc`}}
func.func @invalidType (%arg0: !systemc.value_base>) {}

// -----

// Check that the verifySymbolUses function calls the instance impl library

systemc.module @submodule () { }

hw.module @verilatedCannotReferenceNonHWModule() {
  // expected-error @+1 {{symbol reference 'submodule' isn't a module}}
  systemc.interop.verilated "verilated" @submodule () -> ()
}

// -----

systemc.module @sensitivityNotInCtor() {
  // expected-error @+1 {{expects parent op 'systemc.ctor'}}
  systemc.sensitive
}

// -----

systemc.module @sensitivityNoChannelType() {
  systemc.ctor {
    %var = systemc.cpp.variable : i1
    // expected-error @+1 {{operand #0 must be a SystemC sc_in<T> type or a SystemC sc_inout<T> type or a SystemC sc_out<T> type or a SystemC sc_signal<T> type, but got 'i1'}}
    systemc.sensitive %var : i1
  }
}
