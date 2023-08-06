// RUN: circt-translate %s --export-systemc | FileCheck %s

// CHECK-LABEL: // stdout.h
// CHECK-NEXT: #ifndef STDOUT_H
// CHECK-NEXT: #define STDOUT_H

// CHECK: #include <systemc.h>
// CHECK: #include "nosystemheader"

emitc.include <"systemc.h">
emitc.include "nosystemheader"

// CHECK-EMPTY:
// CHECK-LABEL: SC_MODULE(submodule) {
systemc.module @submodule (%in0: !systemc.in<!systemc.uint<32>>, %in1: !systemc.in<!systemc.uint<32>>, %out0: !systemc.out<!systemc.uint<32>>) {
// CHECK-NEXT: sc_in<sc_uint<32>> in0;
// CHECK-NEXT: sc_in<sc_uint<32>> in1;
// CHECK-NEXT: sc_out<sc_uint<32>> out0;
// CHECK-NEXT: };
}

// CHECK-EMPTY:
// CHECK-LABEL: SC_MODULE(basic) {
systemc.module @basic (%port0: !systemc.in<i1>, %port1: !systemc.inout<!systemc.uint<64>>, %port2: !systemc.out<i64>, %port3: !systemc.out<!systemc.bv<1024>>, %port4: !systemc.out<i1>) {
  // CHECK-NEXT: sc_in<bool> port0;
  // CHECK-NEXT: sc_inout<sc_uint<64>> port1;
  // CHECK-NEXT: sc_out<uint64_t> port2;
  // CHECK-NEXT: sc_out<sc_bv<1024>> port3;
  // CHECK-NEXT: sc_out<bool> port4;
  // CHECK-NEXT: sc_signal<sc_uint<64>> sig;
  %sig = systemc.signal : !systemc.signal<!systemc.uint<64>>
  // CHECK-NEXT: sc_signal<sc_uint<32>> channel;
  %channel = systemc.signal : !systemc.signal<!systemc.uint<32>>
  // CHECK-NEXT: submodule submoduleInstance;
  %submoduleInstance = systemc.instance.decl @submodule : !systemc.module<submodule(in0: !systemc.in<!systemc.uint<32>>, in1: !systemc.in<!systemc.uint<32>>, out0: !systemc.out<!systemc.uint<32>>)>
  // CHECK-NEXT: uint32_t testvar;
  %testvar = systemc.cpp.variable : i32
  // CHECK-NEXT: uint32_t testvarwithinit = 42;
  %c42_i32 = hw.constant 42 : i32
  %testvarwithinit = systemc.cpp.variable %c42_i32 : i32
  // CHECK-EMPTY: 
  // CHECK-NEXT: SC_CTOR(basic) {
  systemc.ctor {
    // CHECK-NEXT: submoduleInstance.in0(channel);
    systemc.instance.bind_port %submoduleInstance["in0"] to %channel : !systemc.module<submodule(in0: !systemc.in<!systemc.uint<32>>, in1: !systemc.in<!systemc.uint<32>>, out0: !systemc.out<!systemc.uint<32>>)>, !systemc.signal<!systemc.uint<32>>
    // CHECK-NEXT: SC_METHOD(add);
    systemc.method %add
    // CHECK-NEXT: SC_THREAD(add);
    systemc.thread %add
  // CHECK-NEXT: }
  }
  // CHECK-EMPTY:
  // CHECK-NEXT: void add() {
  %add = systemc.func {
    // CHECK-NEXT: sig.write(port1.read());
    %0 = systemc.signal.read %port1 : !systemc.inout<!systemc.uint<64>>
    systemc.signal.write %sig, %0 : !systemc.signal<!systemc.uint<64>>
    // CHECK-NEXT: port2.write(42);
    %1 = hw.constant 42 : i64
    systemc.signal.write %port2, %1 : !systemc.out<i64>
    // CHECK-NEXT: port4.write(true);
    %2 = hw.constant 1 : i1
    systemc.signal.write %port4, %2 : !systemc.out<i1>
    // CHECK-NEXT: port4.write(false);
    %3 = hw.constant 0 : i1
    systemc.signal.write %port4, %3 : !systemc.out<i1>
    // CHECK-NEXT: testvar = 42;
    systemc.cpp.assign %testvar = %c42_i32 : i32
    // CHECK-NEXT: testvarwithinit = testvar;
    systemc.cpp.assign %testvarwithinit = %testvar : i32
  // CHECK-NEXT: }
  }
// CHECK-NEXT: };
}

// CHECK-LABEL: namedSignal
systemc.module @namedSignal() {
  // CHECK-NEXT: sc_signal<bool> SC_NAMED(sigName);
  %sigName = systemc.signal named : !systemc.signal<i1>
}

// CHECK-LABEL: SC_MODULE(nativeCTypes)
// CHECK-NEXT: sc_in<bool> port0;
// CHECK-NEXT: sc_in<uint8_t> port1;
// CHECK-NEXT: sc_in<uint16_t> port2;
// CHECK-NEXT: sc_in<uint32_t> port3;
// CHECK-NEXT: sc_in<uint64_t> port4;
// CHECK-NEXT: sc_in<bool> port5;
// CHECK-NEXT: sc_in<int8_t> port6;
// CHECK-NEXT: sc_in<int16_t> port7;
// CHECK-NEXT: sc_in<int32_t> port8;
// CHECK-NEXT: sc_in<int64_t> port9;
// CHECK-NEXT: sc_in<bool> port10;
// CHECK-NEXT: sc_in<uint8_t> port11;
// CHECK-NEXT: sc_in<uint16_t> port12;
// CHECK-NEXT: sc_in<uint32_t> port13;
// CHECK-NEXT: sc_in<uint64_t> port14;
systemc.module @nativeCTypes (%port0: !systemc.in<i1>,
                              %port1: !systemc.in<i8>,
                              %port2: !systemc.in<i16>,
                              %port3: !systemc.in<i32>,
                              %port4: !systemc.in<i64>,
                              %port5: !systemc.in<si1>,
                              %port6: !systemc.in<si8>,
                              %port7: !systemc.in<si16>,
                              %port8: !systemc.in<si32>,
                              %port9: !systemc.in<si64>,
                              %port10: !systemc.in<ui1>,
                              %port11: !systemc.in<ui8>,
                              %port12: !systemc.in<ui16>,
                              %port13: !systemc.in<ui32>,
                              %port14: !systemc.in<ui64>) {}

// CHECK-LABEL: SC_MODULE(systemCTypes)
// CHECK-NEXT: sc_in<sc_int_base> p0;
// CHECK-NEXT: sc_in<sc_int<32>> p1;
// CHECK-NEXT: sc_in<sc_uint_base> p2;
// CHECK-NEXT: sc_in<sc_uint<32>> p3;
// CHECK-NEXT: sc_in<sc_signed> p4;
// CHECK-NEXT: sc_in<sc_bigint<256>> p5;
// CHECK-NEXT: sc_in<sc_unsigned> p6;
// CHECK-NEXT: sc_in<sc_biguint<256>> p7;
// CHECK-NEXT: sc_in<sc_bv_base> p8;
// CHECK-NEXT: sc_in<sc_bv<1024>> p9;
// CHECK-NEXT: sc_in<sc_lv_base> p10
// CHECK-NEXT: sc_in<sc_lv<1024>> p11
// CHECK-NEXT: sc_in<sc_logic> p12;
systemc.module @systemCTypes (%p0: !systemc.in<!systemc.int_base>,
                              %p1: !systemc.in<!systemc.int<32>>,
                              %p2: !systemc.in<!systemc.uint_base>,
                              %p3: !systemc.in<!systemc.uint<32>>,
                              %p4: !systemc.in<!systemc.signed>,
                              %p5: !systemc.in<!systemc.bigint<256>>,
                              %p6: !systemc.in<!systemc.unsigned>,
                              %p7: !systemc.in<!systemc.biguint<256>>,
                              %p8: !systemc.in<!systemc.bv_base>,
                              %p9: !systemc.in<!systemc.bv<1024>>,
                              %p10: !systemc.in<!systemc.lv_base>,
                              %p11: !systemc.in<!systemc.lv<1024>>,
                              %p12: !systemc.in<!systemc.logic>) {}

// CHECK-LABEL: SC_MODULE(emitcEmission)
systemc.module @emitcEmission () {
  // CHECK: SC_CTOR
  systemc.ctor {
    // Test: emitc.constant
    // CHECK-NEXT: int five = 5;
    %0 = "emitc.constant"() {value = #emitc.opaque<"5"> : !emitc.opaque<"int">} : () -> !emitc.opaque<"int">
    %five = systemc.cpp.variable %0 : !emitc.opaque<"int">

    // Test: emitc.apply "&" without having to emit parentheses
    // CHECK-NEXT: int* fiveptr = &five;
    %1 = emitc.apply "&"(%five) : (!emitc.opaque<"int">) -> !emitc.ptr<!emitc.opaque<"int">>
    %fiveptr = systemc.cpp.variable %1: !emitc.ptr<!emitc.opaque<"int">>

    // Test: emitc.apply "&" with parentheses to conform to the precedence rules
    // TODO: add this test-case once we have support for an inlinable operation that has lower precedence

    // Test: emitc.apply "*" without having to emit parentheses
    // CHECK-NEXT: int fivederef = *fiveptr;
    %2 = emitc.apply "*"(%fiveptr) : (!emitc.ptr<!emitc.opaque<"int">>) -> !emitc.opaque<"int">
    %fivederef = systemc.cpp.variable %2: !emitc.opaque<"int">

    // Test: emitc.apply "*" with parentheses to conform to the precedence rules
    // TODO: add this test-case once we have support for an inlinable operation that has lower precedence

    // Test: emit.call without a result is emitted as a statement, having operands and attribute arguments
    // CHECK-NEXT: printf("result: %d, %d\0A", five, 6);
    emitc.call "printf" (%five) {args=["result: %d, %d\n", 0 : index, 6 : i32]} : (!emitc.opaque<"int">) -> ()

    // Test: emit.call without a result is emitted as a statement and having attribute arguments only
    // CHECK-NEXT: printf("result: %d\0A", 6);
    emitc.call "printf" () {args=["result: %d\n", 6 : i32]} : () -> ()

    // Test: emit.call without a result is emitted as a statement, no operands, no attribute arguments
    // CHECK-NEXT: printf();
    emitc.call "printf" () : () -> ()

    // Test: emitc.call with a result is inlined properly, having operands only
    // CHECK-NEXT: void* v0 = malloc(4);
    %3 = hw.constant 4 : i32
    %4 = emitc.call "malloc" (%3) : (i32) -> !emitc.ptr<!emitc.opaque<"void">>
    %v0 = systemc.cpp.variable %4 : !emitc.ptr<!emitc.opaque<"void">>

    // Test: emitc.call with a result is inlined properly, attribute arguments only
    // CHECK-NEXT: void* v1 = malloc(4);
    %5 = emitc.call "malloc" () {args=[4 : i32]}: () -> !emitc.ptr<!emitc.opaque<"void">>
    %v1 = systemc.cpp.variable %5 : !emitc.ptr<!emitc.opaque<"void">>

    // Test: emitc.call properly inserts parentheses when an argument has COMMA precedence
    // TODO: no operation having COMMA precedence supported yet

    // Test: emit.cast adds parentheses around the operand when it has higher precedence than the operand
    // TODO: no applicable operation having lower precedence than CAST supported yet

    // Test: emit.cast does not add parentheses around the operand when it has lower precedence than the operand
    // CHECK-NEXT: int* v2 = (int*) malloc(4);
    %6 = emitc.call "malloc" () {args=[4 : i32]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %7 = emitc.cast %6 : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<!emitc.opaque<"int">>
    %v2 = systemc.cpp.variable %7 : !emitc.ptr<!emitc.opaque<"int">>

    // Test: index -> size_t
    // CHECK-NEXT: size_t idx;
    %idx = systemc.cpp.variable : index
  }
}

// CHECK-LABEL: SC_MODULE(CppEmission)
systemc.module @CppEmission () {
  // CHECK-EMPTY:

  // Test: systemc.cpp.destructor
  // CHECK-NEXT: ~CppEmission() override {
  systemc.cpp.destructor {
    // Test: systemc.cpp.new w/o arguments
    // CHECK-NEXT: submodule* v0 = new submodule;
    %0 = systemc.cpp.new() : () -> !emitc.ptr<!emitc.opaque<"submodule">>
    %v0 = systemc.cpp.variable %0 : !emitc.ptr<!emitc.opaque<"submodule">>

    // Test: systemc.cpp.new with arguments, w/o parens due to precedence
    // CHECK-NEXT: std::tuple<uint32_t, uint32_t>* v1 = new std::tuple<uint32_t, uint32_t>(1, 2);
    %3 = hw.constant 1 : i32
    %4 = hw.constant 2 : i32
    %5 = systemc.cpp.new(%3, %4) : (i32, i32) -> !emitc.ptr<!emitc.opaque<"std::tuple<uint32_t, uint32_t>">>
    %v1 = systemc.cpp.variable %5 : !emitc.ptr<!emitc.opaque<"std::tuple<uint32_t, uint32_t>">>

    // Test: systemc.cpp.new w arguments, with parens due to precedence
    // TODO: there is currently no appropriate inlinable operation implemented with lower precedence

    // Test: systemc.cpp.delete w/o parens due to precedence
    // CHECK-NEXT: delete v0;
    systemc.cpp.delete %v0 : !emitc.ptr<!emitc.opaque<"submodule">>

    // Test: systemc.cpp.delete with parens due to precedence
    // TODO: there is currently no appropriate inlinable operation implemented with lower precedence

  // CHECK-NEXT: }
  }
}

// CHECK-LABEL: SC_MODULE(MemberAccess)
systemc.module @MemberAccess () {
  // CHECK-NEXT: std::pair<int, int> member;
  // CHECK-NEXT: int result;
  %member = systemc.cpp.variable : !emitc.opaque<"std::pair<int, int>">
  %c5 = "emitc.constant"() {value = #emitc.opaque<"5"> : !emitc.opaque<"int">} : () -> !emitc.opaque<"int">
  %result = systemc.cpp.variable : !emitc.opaque<"int">
  // CHECK-EMPTY:
  // CHECK-NEXT: SC_CTOR(MemberAccess) {
  systemc.ctor {
    // CHECK-NEXT: result = member.first;
    %0 = systemc.cpp.member_access %member dot "first" : (!emitc.opaque<"std::pair<int, int>">) -> !emitc.opaque<"int">
    systemc.cpp.assign %result = %0 : !emitc.opaque<"int">
    // CHECK-NEXT: result = (new std::pair<int, int>(5, 5))->second;
    %1 = systemc.cpp.new (%c5, %c5) : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.ptr<!emitc.opaque<"std::pair<int, int>">>
    %2 = systemc.cpp.member_access %1 arrow "second" : (!emitc.ptr<!emitc.opaque<"std::pair<int, int>">>) -> !emitc.opaque<"int">
    systemc.cpp.assign %result = %2 : !emitc.opaque<"int">
  }
}

// CHECK-LABEL: sensitivities
systemc.module @sensitivities(%in: !systemc.in<i1>, %inout: !systemc.inout<i1>) {
  systemc.ctor {
    // CHECK: sensitive << in << inout;
    systemc.sensitive %in, %inout : !systemc.in<i1>, !systemc.inout<i1>
    // CHECK-NEXT: }
    systemc.sensitive
  }
}

// CHECK: #endif // STDOUT_H
