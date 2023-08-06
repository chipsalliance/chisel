// RUN: circt-opt --convert-hw-to-systemc --verify-diagnostics %s | FileCheck %s

// CHECK: emitc.include <"systemc.h">

// CHECK-LABEL: systemc.module @emptyModule ()
hw.module @emptyModule () -> () {}

// CHECK-LABEL: systemc.module @onlyInputs (%a: !systemc.in<!systemc.uint<32>>, %b: !systemc.in<!systemc.biguint<256>>, %c: !systemc.in<!systemc.bv<1024>>, %d: !systemc.in<i1>)
hw.module @onlyInputs (%a: i32, %b: i256, %c: i1024, %d: i1) -> () {}

// CHECK-LABEL: systemc.module @onlyOutputs (%sum: !systemc.out<!systemc.uint<32>>)
hw.module @onlyOutputs () -> (sum: i32) {
  // CHECK-NEXT: systemc.ctor {
  // CHECK-NEXT:   systemc.method %innerLogic
  // CHECK-NEXT: }
  // CHECK-NEXT: %innerLogic = systemc.func  {
  // CHECK-NEXT:   %c0_i32 = hw.constant 0 : i32
  // CHECK-NEXT:   [[CAST:%.+]] = systemc.convert %c0_i32 : (i32) -> !systemc.uint<32>
  // CHECK-NEXT:   systemc.signal.write %sum, [[CAST]] : !systemc.out<!systemc.uint<32>>
  // CHECK-NEXT: }
  %0 = hw.constant 0 : i32
  hw.output %0 : i32
}

// CHECK-LABEL: systemc.module @adder (%a: !systemc.in<!systemc.uint<32>>, %b: !systemc.in<!systemc.uint<32>>, %sum: !systemc.out<!systemc.uint<32>>)
hw.module @adder (%a: i32, %b: i32) -> (sum: i32) {
  // CHECK-NEXT: systemc.ctor {
  // CHECK-NEXT:   systemc.method %innerLogic
  // CHECK-NEXT:   systemc.sensitive %a, %b : !systemc.in<!systemc.uint<32>>, !systemc.in<!systemc.uint<32>>
  // CHECK-NEXT: }
  // CHECK-NEXT: %innerLogic = systemc.func  {
  // CHECK-NEXT:   [[A:%.+]] = systemc.signal.read %a : !systemc.in<!systemc.uint<32>>
  // CHECK-NEXT:   [[AC:%.+]] = systemc.convert [[A]] : (!systemc.uint<32>) -> i32
  // CHECK-NEXT:   [[B:%.+]] = systemc.signal.read %b : !systemc.in<!systemc.uint<32>>
  // CHECK-NEXT:   [[BC:%.+]] = systemc.convert [[B]] : (!systemc.uint<32>) -> i32
  // CHECK-NEXT:   [[RES:%.*]] = comb.add [[AC]], [[BC]] : i32
  // CHECK-NEXT:   [[RESC:%.+]] = systemc.convert [[RES]] : (i32) -> !systemc.uint<32>
  // CHECK-NEXT:   systemc.signal.write %sum, [[RESC]] : !systemc.out<!systemc.uint<32>>
  // CHECK-NEXT: }
  %0 = comb.add %a, %b : i32
  hw.output %0 : i32
// CHECK-NEXT: }
}

// CHECK-LABEL: systemc.module private @moduleVisibility
hw.module private @moduleVisibility () -> () {}

// CHECK-LABEL: systemc.module @argAttrs (%port0: !systemc.in<!systemc.uint<32>> {hw.attrname = "sometext"}, %port1: !systemc.in<!systemc.uint<32>>, %out0: !systemc.out<!systemc.uint<32>>)
hw.module @argAttrs (%port0: i32 {hw.attrname = "sometext"}, %port1: i32) -> (out0: i32) {
  %0 = hw.constant 0 : i32
  hw.output %0 : i32
}

// CHECK-LABEL: systemc.module @resultAttrs (%port0: !systemc.in<!systemc.uint<32>>, %out0: !systemc.out<!systemc.uint<32>> {hw.attrname = "sometext"})
hw.module @resultAttrs (%port0: i32) -> (out0: i32 {hw.attrname = "sometext"}) {
  %0 = hw.constant 0 : i32
  hw.output %0 : i32
}

// CHECK-LABEL: systemc.module @submodule
hw.module @submodule (%in0: i16, %in1: i32) -> (out0: i16, out1: i32, out2: i64) {
  %0 = hw.constant 0 : i64
  hw.output %in0, %in1, %0 : i16, i32, i64
}

// CHECK-LABEL:  systemc.module @instanceLowering (%port0: !systemc.in<!systemc.uint<32>>, %out0: !systemc.out<!systemc.uint<16>>, %out1: !systemc.out<!systemc.uint<32>>, %out2: !systemc.out<!systemc.uint<64>>) {
hw.module @instanceLowering (%port0: i32) -> (out0: i16, out1: i32, out2: i64) {
// CHECK-NEXT:    %inst1 = systemc.instance.decl  @submodule : !systemc.module<submodule(in0: !systemc.in<!systemc.uint<16>>, in1: !systemc.in<!systemc.uint<32>>, out0: !systemc.out<!systemc.uint<16>>, out1: !systemc.out<!systemc.uint<32>>, out2: !systemc.out<!systemc.uint<64>>)>
// CHECK-NEXT:    %inst1_in0 = systemc.signal  : !systemc.signal<!systemc.uint<16>>
// CHECK-NEXT:    %inst1_in1 = systemc.signal  : !systemc.signal<!systemc.uint<32>>
// CHECK-NEXT:    %inst1_out0 = systemc.signal  : !systemc.signal<!systemc.uint<16>>
// CHECK-NEXT:    %inst1_out1 = systemc.signal  : !systemc.signal<!systemc.uint<32>>
// CHECK-NEXT:    %inst1_out2 = systemc.signal  : !systemc.signal<!systemc.uint<64>>
// CHECK-NEXT:    %inst2 = systemc.instance.decl  @submodule : !systemc.module<submodule(in0: !systemc.in<!systemc.uint<16>>, in1: !systemc.in<!systemc.uint<32>>, out0: !systemc.out<!systemc.uint<16>>, out1: !systemc.out<!systemc.uint<32>>, out2: !systemc.out<!systemc.uint<64>>)>
// CHECK-NEXT:    %inst2_out0 = systemc.signal  : !systemc.signal<!systemc.uint<16>>
// CHECK-NEXT:    %inst2_out1 = systemc.signal  : !systemc.signal<!systemc.uint<32>>
// CHECK-NEXT:    %inst2_out2 = systemc.signal  : !systemc.signal<!systemc.uint<64>>
// CHECK-NEXT:    systemc.ctor {
// CHECK-NEXT:      systemc.method [[UPDATEFUNC:%.+]]
// CHECK-NEXT:      systemc.sensitive %port0 : !systemc.in<!systemc.uint<32>>
// CHECK-NEXT:      systemc.instance.bind_port %inst1["in0"] to %inst1_in0 : !systemc.module<submodule(in0: !systemc.in<!systemc.uint<16>>, in1: !systemc.in<!systemc.uint<32>>, out0: !systemc.out<!systemc.uint<16>>, out1: !systemc.out<!systemc.uint<32>>, out2: !systemc.out<!systemc.uint<64>>)>, !systemc.signal<!systemc.uint<16>>
// CHECK-NEXT:      systemc.instance.bind_port %inst1["in1"] to %inst1_in1 : !systemc.module<submodule(in0: !systemc.in<!systemc.uint<16>>, in1: !systemc.in<!systemc.uint<32>>, out0: !systemc.out<!systemc.uint<16>>, out1: !systemc.out<!systemc.uint<32>>, out2: !systemc.out<!systemc.uint<64>>)>, !systemc.signal<!systemc.uint<32>>
// CHECK-NEXT:      systemc.instance.bind_port %inst1["out0"] to %inst1_out0 : !systemc.module<submodule(in0: !systemc.in<!systemc.uint<16>>, in1: !systemc.in<!systemc.uint<32>>, out0: !systemc.out<!systemc.uint<16>>, out1: !systemc.out<!systemc.uint<32>>, out2: !systemc.out<!systemc.uint<64>>)>, !systemc.signal<!systemc.uint<16>>
// CHECK-NEXT:      systemc.instance.bind_port %inst1["out1"] to %inst1_out1 : !systemc.module<submodule(in0: !systemc.in<!systemc.uint<16>>, in1: !systemc.in<!systemc.uint<32>>, out0: !systemc.out<!systemc.uint<16>>, out1: !systemc.out<!systemc.uint<32>>, out2: !systemc.out<!systemc.uint<64>>)>, !systemc.signal<!systemc.uint<32>>
// CHECK-NEXT:      systemc.instance.bind_port %inst1["out2"] to %inst1_out2 : !systemc.module<submodule(in0: !systemc.in<!systemc.uint<16>>, in1: !systemc.in<!systemc.uint<32>>, out0: !systemc.out<!systemc.uint<16>>, out1: !systemc.out<!systemc.uint<32>>, out2: !systemc.out<!systemc.uint<64>>)>, !systemc.signal<!systemc.uint<64>>
// CHECK-NEXT:      systemc.instance.bind_port %inst2["in0"] to %inst1_out0 : !systemc.module<submodule(in0: !systemc.in<!systemc.uint<16>>, in1: !systemc.in<!systemc.uint<32>>, out0: !systemc.out<!systemc.uint<16>>, out1: !systemc.out<!systemc.uint<32>>, out2: !systemc.out<!systemc.uint<64>>)>, !systemc.signal<!systemc.uint<16>>
// CHECK-NEXT:      systemc.instance.bind_port %inst2["in1"] to %inst1_out1 : !systemc.module<submodule(in0: !systemc.in<!systemc.uint<16>>, in1: !systemc.in<!systemc.uint<32>>, out0: !systemc.out<!systemc.uint<16>>, out1: !systemc.out<!systemc.uint<32>>, out2: !systemc.out<!systemc.uint<64>>)>, !systemc.signal<!systemc.uint<32>>
// CHECK-NEXT:      systemc.instance.bind_port %inst2["out0"] to %inst2_out0 : !systemc.module<submodule(in0: !systemc.in<!systemc.uint<16>>, in1: !systemc.in<!systemc.uint<32>>, out0: !systemc.out<!systemc.uint<16>>, out1: !systemc.out<!systemc.uint<32>>, out2: !systemc.out<!systemc.uint<64>>)>, !systemc.signal<!systemc.uint<16>>
// CHECK-NEXT:      systemc.instance.bind_port %inst2["out1"] to %inst2_out1 : !systemc.module<submodule(in0: !systemc.in<!systemc.uint<16>>, in1: !systemc.in<!systemc.uint<32>>, out0: !systemc.out<!systemc.uint<16>>, out1: !systemc.out<!systemc.uint<32>>, out2: !systemc.out<!systemc.uint<64>>)>, !systemc.signal<!systemc.uint<32>>
// CHECK-NEXT:      systemc.instance.bind_port %inst2["out2"] to %inst2_out2 : !systemc.module<submodule(in0: !systemc.in<!systemc.uint<16>>, in1: !systemc.in<!systemc.uint<32>>, out0: !systemc.out<!systemc.uint<16>>, out1: !systemc.out<!systemc.uint<32>>, out2: !systemc.out<!systemc.uint<64>>)>, !systemc.signal<!systemc.uint<64>>
// CHECK-NEXT:    }
// CHECK-NEXT:    [[UPDATEFUNC]] = systemc.func  {
// CHECK-NEXT:      [[V0:%.+]] = systemc.signal.read %port0 : !systemc.in<!systemc.uint<32>>
// CHECK-NEXT:      [[V1:%.+]] = systemc.convert [[V0]] : (!systemc.uint<32>) -> i32
// CHECK-NEXT:      [[V2:%.+]] = systemc.convert [[V1]] : (i32) -> !systemc.uint<32>
// CHECK-NEXT:      %c0_i16 = hw.constant 0 : i16
// CHECK-NEXT:      [[V3:%.+]] = systemc.convert %c0_i16 : (i16) -> !systemc.uint<16>
// CHECK-NEXT:      systemc.signal.write %inst1_in0, [[V3]] : !systemc.signal<!systemc.uint<16>>
// CHECK-NEXT:      systemc.signal.write %inst1_in1, [[V2]] : !systemc.signal<!systemc.uint<32>>
// CHECK-NEXT:      systemc.signal.read %inst1_out0 : !systemc.signal<!systemc.uint<16>>
// CHECK-NEXT:      systemc.signal.read %inst1_out1 : !systemc.signal<!systemc.uint<32>>
// CHECK-NEXT:      [[V6:%.+]] = systemc.signal.read %inst1_out2 : !systemc.signal<!systemc.uint<64>>
// CHECK-NEXT:      [[V7:%.+]] = systemc.signal.read %inst2_out0 : !systemc.signal<!systemc.uint<16>>
// CHECK-NEXT:      [[V8:%.+]] = systemc.signal.read %inst2_out1 : !systemc.signal<!systemc.uint<32>>
// CHECK-NEXT:      systemc.signal.read %inst2_out2 : !systemc.signal<!systemc.uint<64>>
// CHECK-NEXT:      [[V10:%.+]] = systemc.convert [[V7]] : (!systemc.uint<16>) -> !systemc.uint<16>
// CHECK-NEXT:      systemc.signal.write %out0, [[V10]] : !systemc.out<!systemc.uint<16>>
// CHECK-NEXT:      [[V11:%.+]] = systemc.convert [[V8]] : (!systemc.uint<32>) -> !systemc.uint<32>
// CHECK-NEXT:      systemc.signal.write %out1, [[V11]] : !systemc.out<!systemc.uint<32>>
// CHECK-NEXT:      [[V12:%.+]] = systemc.convert [[V6]] : (!systemc.uint<64>) -> !systemc.uint<64>
// CHECK-NEXT:      systemc.signal.write %out2, [[V12]] : !systemc.out<!systemc.uint<64>>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
  %0 = hw.constant 0 : i16
  %inst1.out0, %inst1.out1, %inst1.out2 = hw.instance "inst1" @submodule (in0: %0: i16, in1: %port0: i32) -> (out0: i16, out1: i32, out2: i64)
  %inst2.out0, %inst2.out1, %inst2.out2 = hw.instance "inst2" @submodule (in0: %inst1.out0: i16, in1: %inst1.out1: i32) -> (out0: i16, out1: i32, out2: i64)
  hw.output %inst2.out0, %inst2.out1, %inst1.out2 : i16, i32, i64
}

// CHECK-LABEL:  systemc.module @instanceLowering2
hw.module @instanceLowering2 () -> () {
// CHECK-NEXT: %inst1 = systemc.instance.decl @emptyModule : !systemc.module<emptyModule()>
// CHECK-NEXT: systemc.ctor {
// CHECK-NEXT:   systemc.method %
// CHECK-NEXT: }
// CHECK-NEXT: systemc.func {
// CHECK-NEXT: }
  hw.instance "inst1" @emptyModule () -> ()
// CHECK-NEXT: }
}

// CHECK-LABEL: systemc.module @systemCTypes (%p0: !systemc.in<!systemc.int<32>>, %p1: !systemc.in<!systemc.int_base>, %p2: !systemc.in<!systemc.uint<32>>, %p3: !systemc.in<!systemc.uint_base>, %p4: !systemc.in<!systemc.bigint<32>>, %p5: !systemc.in<!systemc.signed>, %p6: !systemc.in<!systemc.biguint<32>>, %p7: !systemc.in<!systemc.unsigned>, %p8: !systemc.in<!systemc.bv<32>>, %p9: !systemc.in<!systemc.bv_base>, %p10: !systemc.in<!systemc.lv<32>>, %p11: !systemc.in<!systemc.lv_base>, %p12: !systemc.in<!systemc.logic>)
hw.module @systemCTypes (%p0: !systemc.int<32>, %p1: !systemc.int_base,
                         %p2: !systemc.uint<32>, %p3: !systemc.uint_base,
                         %p4: !systemc.bigint<32>, %p5: !systemc.signed,
                         %p6: !systemc.biguint<32>, %p7: !systemc.unsigned,
                         %p8: !systemc.bv<32>, %p9: !systemc.bv_base,
                         %p10: !systemc.lv<32>, %p11: !systemc.lv_base,
                         %p12: !systemc.logic) -> () {}
