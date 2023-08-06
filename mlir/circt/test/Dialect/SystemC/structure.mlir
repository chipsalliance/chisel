// RUN: circt-opt %s | FileCheck %s

// CHECK: systemc.module @adder (%summand_a: !systemc.in<!systemc.uint<32>>, %summand_b: !systemc.in<!systemc.uint<32>>, %sum: !systemc.out<!systemc.uint<32>>) {
systemc.module @adder (%summand_a: !systemc.in<!systemc.uint<32>>, %summand_b: !systemc.in<!systemc.uint<32>>, %sum: !systemc.out<!systemc.uint<32>>) {
  //CHECK-NEXT: systemc.ctor {
  systemc.ctor {
    // CHECK-NEXT: systemc.method %addFunc
    systemc.method %addFunc
    // CHECK-NEXT: systemc.thread %addFunc
    systemc.thread %addFunc
  //CHECK-NEXT: }
  }
  // CHECK-NEXT: %addFunc = systemc.func {
  %addFunc = systemc.func {
    // CHECK-NEXT: [[SA:%.+]] = systemc.signal.read %summand_a : !systemc.in<!systemc.uint<32>>
    %0 = systemc.signal.read %summand_a : !systemc.in<!systemc.uint<32>>
    // CHECK-NEXT: [[SB:%.+]] = systemc.signal.read %summand_b : !systemc.in<!systemc.uint<32>>
    %1 = systemc.signal.read %summand_b : !systemc.in<!systemc.uint<32>>
    // CHECK-NEXT: [[SAC:%.+]] = systemc.convert [[SA]] : (!systemc.uint<32>) -> i32
    %2 = systemc.convert %0 : (!systemc.uint<32>) -> i32
    // CHECK-NEXT: [[SBC:%.+]] = systemc.convert [[SB]] : (!systemc.uint<32>) -> i32
    %3 = systemc.convert %1 : (!systemc.uint<32>) -> i32
    // CHECK-NEXT: [[ADD:%.*]] = comb.add [[SAC]], [[SBC]] : i32
    %4 = comb.add %2, %3 : i32
    // CHECK-NEXT: [[RES:%.*]] = systemc.convert [[ADD]] : (i32) -> !systemc.uint<32>
    %5 = systemc.convert %4 : (i32) -> !systemc.uint<32>
    // CHECK-NEXT: systemc.signal.write %sum, [[RES]] : !systemc.out<!systemc.uint<32>>
    systemc.signal.write %sum, %5 : !systemc.out<!systemc.uint<32>>
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: systemc.cpp.destructor {
  systemc.cpp.destructor {
    // CHECK-NEXT: [[PTR:%.+]] = systemc.cpp.new(%summand_a, %summand_b) : (!systemc.in<!systemc.uint<32>>, !systemc.in<!systemc.uint<32>>) -> !emitc.ptr<!emitc.opaque<"someclass">>
    %0 = systemc.cpp.new(%summand_a, %summand_b) : (!systemc.in<!systemc.uint<32>>, !systemc.in<!systemc.uint<32>>) -> !emitc.ptr<!emitc.opaque<"someclass">>
    // CHECK-NEXT: systemc.cpp.delete [[PTR]] : !emitc.ptr<!emitc.opaque<"someclass">>
    systemc.cpp.delete %0 : !emitc.ptr<!emitc.opaque<"someclass">>
    // CHECK-NEXT: systemc.cpp.new() : () -> !emitc.ptr<!emitc.opaque<"someclass">>
    %1 = systemc.cpp.new() : () -> !emitc.ptr<!emitc.opaque<"someclass">>
  // CHECK-NEXT: }
  }
// CHECK-NEXT: }
}

// CHECK-LABEL: systemc.module @mixedPorts (%port0: !systemc.out<!systemc.int<4>>, %port1: !systemc.in<!systemc.uint<32>>, %port2: !systemc.out<!systemc.biguint<256>>, %port3: !systemc.inout<!systemc.bigint<512>>, %port4: !systemc.out<!systemc.bv<1024>>, %port5: !systemc.out<!systemc.bv_base>, %port6: !systemc.out<!systemc.lv<1024>>, %port7: !systemc.out<!systemc.lv_base>, %port8: !systemc.out<!systemc.int_base>, %port9: !systemc.out<!systemc.uint_base>, %port10: !systemc.out<!systemc.signed>, %port11: !systemc.out<!systemc.unsigned>, %port12: !systemc.out<!systemc.logic>)
systemc.module @mixedPorts (%port0: !systemc.out<!systemc.int<4>>,
                            %port1: !systemc.in<!systemc.uint<32>>,
                            %port2: !systemc.out<!systemc.biguint<256>>,
                            %port3: !systemc.inout<!systemc.bigint<512>>,
                            %port4: !systemc.out<!systemc.bv<1024>>,
                            %port5: !systemc.out<!systemc.bv_base>,
                            %port6: !systemc.out<!systemc.lv<1024>>,
                            %port7: !systemc.out<!systemc.lv_base>,
                            %port8: !systemc.out<!systemc.int_base>,
                            %port9: !systemc.out<!systemc.uint_base>,
                            %port10: !systemc.out<!systemc.signed>,
                            %port11: !systemc.out<!systemc.unsigned>,
                            %port12: !systemc.out<!systemc.logic>) {}

// CHECK-LABEL: systemc.module @signals
systemc.module @signals () {
  // CHECK-NEXT: %signal0 = systemc.signal : !systemc.signal<!systemc.uint<32>>
  %signal0 = systemc.signal : !systemc.signal<!systemc.uint<32>>
  // CHECK-NEXT: %signal1 = systemc.signal : !systemc.signal<i1>
  %signal1 = systemc.signal : !systemc.signal<i1>
  // CHECK-NEXT: %signal2 = systemc.signal : !systemc.signal<i1>
  %ignoredName1 = systemc.signal name "signal2" : !systemc.signal<i1>

  // CHECK-NEXT: %namedSignal = systemc.signal named : !systemc.signal<i1>
  %namedSignal = systemc.signal named : !systemc.signal<i1>

  // CHECK-NEXT: systemc.func
  %funchandle = systemc.func {
    // CHECK-NEXT: [[READ:%.*]] = systemc.signal.read %signal0 : !systemc.signal<!systemc.uint<32>>
    %0 = systemc.signal.read %signal0 : !systemc.signal<!systemc.uint<32>>
    // CHECK-NEXT: systemc.signal.write %signal0, [[READ:%.*]] : !systemc.signal<!systemc.uint<32>>
    systemc.signal.write %signal0, %0 : !systemc.signal<!systemc.uint<32>>
  }
}

// CHECK-LABEL: systemc.module @readWriteInOutPorts
systemc.module @readWriteInOutPorts (%port0 : !systemc.inout<!systemc.uint<32>>, %port1 : !systemc.inout<!systemc.uint<32>>) {
  // CHECK-NEXT: systemc.func
  %funchandle = systemc.func {
    // CHECK-NEXT: [[READ:%.*]] = systemc.signal.read %port0 : !systemc.inout<!systemc.uint<32>>
    %0 = systemc.signal.read %port0 : !systemc.inout<!systemc.uint<32>>
    // CHECK-NEXT: systemc.signal.write %port1, [[READ:%.*]] : !systemc.inout<!systemc.uint<32>>
    systemc.signal.write %port1, %0 : !systemc.inout<!systemc.uint<32>>
  }
}

// CHECK-LABEL: systemc.module private @moduleVisibility
systemc.module private @moduleVisibility () {}

// CHECK-LABEL: systemc.module @argAttrs (%port0: !systemc.in<!systemc.uint<32>> {hw.attrname = "sometext"}, %port1: !systemc.in<!systemc.uint<32>>, %out0: !systemc.out<!systemc.uint<32>>)
systemc.module @argAttrs (%port0: !systemc.in<!systemc.uint<32>> {hw.attrname = "sometext"}, %port1: !systemc.in<!systemc.uint<32>>, %out0: !systemc.out<!systemc.uint<32>>) {}

// CHECK-LABEL: systemc.module @resultAttrs (%port0: !systemc.in<!systemc.uint<32>>, %out0: !systemc.out<!systemc.uint<32>> {hw.attrname = "sometext"})
systemc.module @resultAttrs (%port0: !systemc.in<!systemc.uint<32>>, %out0: !systemc.out<!systemc.uint<32>> {hw.attrname = "sometext"}) {}

// CHECK-LABEL: systemc.module @instanceDecl
systemc.module @instanceDecl (%input0: !systemc.in<!systemc.uint<32>>) {
  // CHECK-NEXT: %moduleInstance0 = systemc.instance.decl @adder : !systemc.module<adder(summand_a: !systemc.in<!systemc.uint<32>>, summand_b: !systemc.in<!systemc.uint<32>>, sum: !systemc.out<!systemc.uint<32>>)>
  %moduleInstance0 = systemc.instance.decl @adder : !systemc.module<adder(summand_a: !systemc.in<!systemc.uint<32>>, summand_b: !systemc.in<!systemc.uint<32>>, sum: !systemc.out<!systemc.uint<32>>)>
  // CHECK-NEXT: %moduleInstance1 = systemc.instance.decl @moduleVisibility : !systemc.module<moduleVisibility()>
  %moduleInstance1 = systemc.instance.decl @moduleVisibility : !systemc.module<moduleVisibility()>
  // CHECK-NEXT: %moduleInstance2 = systemc.instance.decl @moduleVisibility : !systemc.module<moduleVisibility()>
  %ignoredName = systemc.instance.decl name "moduleInstance2" @moduleVisibility : !systemc.module<moduleVisibility()>
  // CHECK-NEXT: systemc.ctor
  systemc.ctor {
    // CHECK-NEXT: systemc.instance.bind_port %moduleInstance0["summand_a"] to %input0 : !systemc.module<adder(summand_a: !systemc.in<!systemc.uint<32>>, summand_b: !systemc.in<!systemc.uint<32>>, sum: !systemc.out<!systemc.uint<32>>)>, !systemc.in<!systemc.uint<32>>
    systemc.instance.bind_port %moduleInstance0["summand_a"] to %input0 : !systemc.module<adder(summand_a: !systemc.in<!systemc.uint<32>>, summand_b: !systemc.in<!systemc.uint<32>>, sum: !systemc.out<!systemc.uint<32>>)>, !systemc.in<!systemc.uint<32>>
  }
}

// CHECK-LABEL: systemc.module @attributes
systemc.module @attributes () {
  // CHECK-NEXT: systemc.ctor attributes {systemc.someattr = 0 : i64} {
  systemc.ctor attributes {systemc.someattr = 0 : i64} {}
  // CHECK-NEXT: }
  // CHECK-NEXT: %func = systemc.func attributes {systemc.someattr = 0 : i64} {
  %func = systemc.func attributes {systemc.someattr = 0 : i64} {}
  // CHECK-NEXT: }
  // CHECK-NEXT: systemc.cpp.destructor attributes {systemc.someattr = 0 : i64} {
  systemc.cpp.destructor attributes {systemc.someattr = 0 : i64} {}
  // CHECK-NEXT: }
}

// CHECK-LABEL: systemc.module @variableAndAssign
systemc.module @variableAndAssign () {
  // CHECK-NEXT: %varname = systemc.cpp.variable : i32
  %varname = systemc.cpp.variable : i32
  // CHECK-NEXT: systemc.ctor {
  systemc.ctor {
    // CHECK-NEXT: %c42_i32 = hw.constant 42 : i32
    %c42_i32 = hw.constant 42 : i32
    // CHECK-NEXT: systemc.cpp.assign %varname = %c42_i32 : i32
    systemc.cpp.assign %varname = %c42_i32 : i32
    // CHECK-NEXT: %varwithinit = systemc.cpp.variable %varname : i32
    %varwithinit = systemc.cpp.variable %varname : i32
    // CHECK-NEXT: systemc.cpp.assign %varwithinit = %varname : i32
    systemc.cpp.assign %varwithinit = %varname : i32
  }
}

// CHECK-LABEL: @member_access
systemc.module @member_access () {
  %member = systemc.cpp.variable : !emitc.opaque<"std::vector<int>">
  %ptrmember = systemc.cpp.variable : !emitc.ptr<!emitc.opaque<"std::vector<int>">>
  systemc.ctor {
    // CHECK: %{{.+}} = systemc.cpp.member_access %member dot "clear" : (!emitc.opaque<"std::vector<int>">) -> (() -> ())
    %0 = systemc.cpp.member_access %member dot "clear" : (!emitc.opaque<"std::vector<int>">) -> (() -> ())
    // CHECK-NEXT: %{{.+}} = systemc.cpp.member_access %ptrmember arrow "clear" : (!emitc.ptr<!emitc.opaque<"std::vector<int>">>) -> (() -> ())
    %1 = systemc.cpp.member_access %ptrmember arrow "clear" : (!emitc.ptr<!emitc.opaque<"std::vector<int>">>) -> (() -> ())
  }
}

hw.module.extern @inst() -> ()
hw.module.extern @instWithArgs(%a: i32, %b: i8) -> (c: i32, d: i8)

// CHECK-LABEL: @verilatorInteropInHwModule
hw.module @verilatorInteropInHwModule(%arg0: i32, %arg1: i8) -> () {
  // CHECK-NEXT: %verilated.c, %verilated.d = systemc.interop.verilated "verilated" @instWithArgs (a: %arg0: i32, b: %arg1: i8) -> (c: i32, d: i8)
  %verilated.c, %verilated.d = systemc.interop.verilated "verilated" @instWithArgs (a: %arg0: i32, b: %arg1: i8) -> (c: i32, d: i8)
}

// CHECK-LABEL: @verilatorInteropInSystemCModule
systemc.module @verilatorInteropInSystemCModule() {
  // CHECK-NEXT: systemc.interop.verilated "verilated" @inst () -> ()
  systemc.interop.verilated "verilated" @inst () -> ()
}

// CHECK-LABEL: @sensitivity
systemc.module @sensitivity (%in: !systemc.in<i1>, %out: !systemc.out<i1>, %inout: !systemc.inout<i1>) {
  %signal = systemc.signal : !systemc.signal<i1>
  systemc.ctor {
    // CHECK: systemc.sensitive %in, %out, %inout, %signal : !systemc.in<i1>, !systemc.out<i1>, !systemc.inout<i1>, !systemc.signal<i1>
    systemc.sensitive %in, %out, %inout, %signal : !systemc.in<i1>, !systemc.out<i1>, !systemc.inout<i1>, !systemc.signal<i1>
    // CHECK-NEXT: systemc.sensitive
    systemc.sensitive
  }
}

// CHECK-LABEL: systemc.module @funcNames
systemc.module @funcNames () {
  // CHECK: %func1 = systemc.func
  // CHECK: %func2 = systemc.func
  %func1 = systemc.func {}
  %ignoredName = systemc.func name "func2" {}
}
