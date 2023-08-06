// RUN: circt-opt %s -export-verilog -verify-diagnostics -o %t.mlir | FileCheck %s --strict-whitespace

// CHECK: module namechange(
// CHECK: input  [3:0] casex_0,
// CHECK: output [3:0] if_0
// CHECK: );
hw.module @namechange(%casex: i4) -> (if: i4) {
  // CHECK: assign if_0 = casex_0;
  hw.output %casex : i4
}

hw.module.extern @module_with_bool<bparam: i1>() -> ()

// CHECK-LABEL: module parametersNameConflict
// CHECK-NEXT:    #(parameter [41:0] p2 = 42'd17,
// CHECK-NEXT:      parameter [0:0]  wire_0) (
// CHECK-NEXT:    input [7:0] p1
// CHECK-NEXT: );
hw.module @parametersNameConflict<p2: i42 = 17, wire: i1>(%p1: i8) {
  %myWire = sv.wire : !hw.inout<i1>

  // CHECK: `ifdef SOMEMACRO
  sv.ifdef "SOMEMACRO" {
    // CHECK: localparam local_0 = wire_0;
    %local = sv.localparam { value = #hw.param.decl.ref<"wire">: i1 } : i1

    // CHECK: assign myWire = wire_0;
    %0 = hw.param.value i1 = #hw.param.decl.ref<"wire">
    sv.assign %myWire, %0: i1
  }

  // "wire" param getting updated should update in this instance.

  // CHECK: module_with_bool #(
  // CHECK:  .bparam(wire_0)
  // CHECK: ) inst ();
  hw.instance "inst" @module_with_bool<bparam: i1 = #hw.param.decl.ref<"wire">>() -> ()

  // CHECK: module_with_bool #(
  // CHECK:  .bparam(wire ^ 1)
  // CHECK: ) inst2 ();
  hw.instance "inst2" @module_with_bool<bparam: i1 = #hw.param.expr.xor<#hw.param.verbatim<"wire">, true>>() -> ()
}

// CHECK-LABEL: module useParametersNameConflict(
hw.module @useParametersNameConflict(%xxx: i8) {
  // CHECK: parametersNameConflict #(
  // CHECK:  .p2(42'd27),
  // CHECK:  .wire_0(0)
  // CHECK: ) inst (
  // CHECK:  .p1 (xxx)
  // CHECK: );
  hw.instance "inst" @parametersNameConflict<p2: i42 = 27, wire: i1 = 0>(p1: %xxx: i8) -> ()

  // CHECK: `ifdef SOMEMACRO
  sv.ifdef "SOMEMACRO" {
    // CHECK: reg [3:0] xxx_0;
    %0 = sv.reg name "xxx" : !hw.inout<i4>
  }
}

// https://github.com/llvm/circt/issues/681
// Rename keywords used in variable/module names
// CHECK-LABEL: module inout_0(
// CHECK:         input  inout_0,
// CHECK:         output output_0
// CHECK:       );
hw.module @inout(%inout: i1) -> (output: i1) {
// CHECK:       assign output_0 = inout_0;
  hw.output %inout : i1
}

// CHECK-LABEL: module inout_inst(
hw.module @inout_inst(%a: i1) {
  // CHECK: inout_0 foo (
  // CHECK:   .inout_0  (a),
  // CHECK:   .output_0 (/* unused */)
  // CHECK: );
  %0 = hw.instance "foo" @inout (inout: %a: i1) -> (output: i1)
}

// https://github.com/llvm/circt/issues/681
// Rename keywords used in variable/module names
// CHECK-LABEL: module reg_0(
// CHECK-NEXT:    input  inout_0,
// CHECK-NEXT:    output output_0
// CHECK-NEXT:  );
hw.module @reg(%inout: i1) -> (output: i1) {
  // CHECK: assign output_0 = inout_0;
  hw.output %inout : i1
}

// https://github.com/llvm/circt/issues/525
// CHECK-LABEL: module issue525(
// CHECK-NEXT:    input  [1:0] struct_0,
// CHECK-NEXT:                 else_0,
// CHECK-NEXT:    output [1:0] casex_0
// CHECK-NEXT:  );
hw.module @issue525(%struct: i2, %else: i2) -> (casex: i2) {
  // CHECK: assign casex_0 = struct_0 + else_0;
  %2 = comb.add %struct, %else : i2
  hw.output %2 : i2
}

hw.module @B(%a: i1) -> () {
}

// CHECK-LABEL: module TestDupInstanceName(
hw.module @TestDupInstanceName(%a: i1) {
  // CHECK: B name (
  hw.instance "name" @B(a: %a: i1) -> ()
  // CHECK: B name_0 (
  hw.instance "name" @B(a: %a: i1) -> ()
}

// CHECK-LABEL: module TestEmptyInstanceName(
hw.module @TestEmptyInstanceName(%a: i1) {
  // CHECK: B _GEN (
  hw.instance "" @B(a: %a: i1) -> ()
  // CHECK: B _GEN_0 (
  hw.instance "" @B(a: %a: i1) -> ()
}

// CHECK-LABEL: module TestInstanceNameValueConflict(
hw.module @TestInstanceNameValueConflict(%a: i1) {
  // CHECK:  wire name;
  %name = sv.wire : !hw.inout<i1>
  // CHECK:  wire output_0;
  %output = sv.wire : !hw.inout<i1>
  // CHECK:  reg  input_0;
  %input = sv.reg : !hw.inout<i1>
  // CHECK: B name_0 (
  hw.instance "name" @B(a: %a: i1) -> ()
}

// https://github.com/llvm/circt/issues/855
// CHECK-LABEL: module nameless_reg(
hw.module @nameless_reg(%a: i1) -> () {
  // CHECK: reg [3:0] _GEN;
  %661 = sv.reg : !hw.inout<i4>
}

// CHECK-LABEL: module verif_renames(
hw.module @verif_renames(%cond: i1) {
  // CHECK: initial
  sv.initial {
    // CHECK:   assert_0: assert(cond);
    sv.assert %cond, immediate label "assert"
  }
}

// CHECK-LABEL: module verbatim_renames(
hw.module @verbatim_renames(%a: i1 {hw.exportPort = #hw<innerSym@asym>}) {
  // CHECK: // VERB Module : reg_0 inout_0
  // CHECK: wire wire_0;
  sv.verbatim "// VERB Module : {{0}} {{1}}" {symbols = [@reg, @inout]}

  // Make sure symbol references to wires and instances get renamed correctly.
  %wire = sv.wire sym @wire1 : !hw.inout<i1>

  // CHECK: inout_0 module_0 (
  %0 = hw.instance "module" sym @struct @inout (inout: %a: i1) -> (output: i1)

  // CHECK: // VERB Instance : module_0 wire_0 a
  sv.verbatim "// VERB Instance : {{0}} {{1}} {{2}}" {symbols = [#hw.innerNameRef<@verbatim_renames::@struct>, #hw.innerNameRef<@verbatim_renames::@wire1>, #hw.innerNameRef<@verbatim_renames::@asym>]}
}

// CHECK-LABEL: interface output_0;
sv.interface @output {
  // CHECK-NEXT: logic input_0;
  sv.interface.signal @input : i1
  // CHECK-NEXT: logic wire_0;
  sv.interface.signal @wire : i1
  // CHECK-NEXT: modport always_0(input input_0, output wire_0);
  sv.interface.modport @always (input @input, output @wire)
}

// Renaming the above interface declarations needs to rename their use in the
// following types.
// CHECK-LABEL: module InterfaceAsInstance();
hw.module @InterfaceAsInstance () {
  // CHECK: output_0 myOutput();
  %myOutput = sv.interface.instance : !sv.interface<@output>
}
