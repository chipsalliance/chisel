// REQUIRES: clang-tidy, systemc
// RUN: circt-translate %s --export-systemc > %t.cpp
// RUN: clang-tidy --extra-arg=-frtti %t.cpp --

emitc.include <"systemc.h">
emitc.include <"tuple">

systemc.module @submodule (%in0: !systemc.in<!systemc.uint<32>>, %in1: !systemc.in<!systemc.uint<32>>, %out0: !systemc.out<!systemc.uint<32>>) {}

systemc.module @module (%port0: !systemc.in<i1>, %port1: !systemc.inout<!systemc.uint<64>>, %port2: !systemc.out<i64>, %port3: !systemc.out<!systemc.bv<1024>>, %port4: !systemc.out<i1>) {
  %submoduleInstance = systemc.instance.decl @submodule : !systemc.module<submodule(in0: !systemc.in<!systemc.uint<32>>, in1: !systemc.in<!systemc.uint<32>>, out0: !systemc.out<!systemc.uint<32>>)>
  %sig = systemc.signal : !systemc.signal<!systemc.uint<64>>
  %channel = systemc.signal : !systemc.signal<!systemc.uint<32>>
  %testvar = systemc.cpp.variable : i32
  %c42_i32 = hw.constant 42 : i32
  %testvarwithinit = systemc.cpp.variable %c42_i32 : i32
  systemc.ctor {
    systemc.instance.bind_port %submoduleInstance["in0"] to %channel : !systemc.module<submodule(in0: !systemc.in<!systemc.uint<32>>, in1: !systemc.in<!systemc.uint<32>>, out0: !systemc.out<!systemc.uint<32>>)>, !systemc.signal<!systemc.uint<32>>
    systemc.method %add
    systemc.thread %add
  }
  %add = systemc.func {
    %0 = systemc.signal.read %port1 : !systemc.inout<!systemc.uint<64>>
    systemc.signal.write %sig, %0 : !systemc.signal<!systemc.uint<64>>
    %1 = hw.constant 42 : i64
    systemc.signal.write %port2, %1 : !systemc.out<i64>
    %2 = hw.constant 1 : i1
    systemc.signal.write %port4, %2 : !systemc.out<i1>
    %3 = hw.constant 0 : i1
    systemc.signal.write %port4, %3 : !systemc.out<i1>
    systemc.cpp.assign %testvar = %c42_i32 : i32
    systemc.cpp.assign %testvarwithinit = %testvar : i32
  }
}

systemc.module @namedSignal() {
  %sigName = systemc.signal named : !systemc.signal<i1>
}

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

systemc.module @emitcEmission () {
  systemc.ctor {
    %0 = "emitc.constant"() {value = #emitc.opaque<"5"> : !emitc.opaque<"int">} : () -> !emitc.opaque<"int">
    %five = systemc.cpp.variable %0 : !emitc.opaque<"int">
    %1 = emitc.apply "&"(%five) : (!emitc.opaque<"int">) -> !emitc.ptr<!emitc.opaque<"int">>
    %2 = emitc.apply "*"(%1) : (!emitc.ptr<!emitc.opaque<"int">>) -> !emitc.opaque<"int">
    %3 = emitc.cast %2: !emitc.opaque<"int"> to !emitc.opaque<"long">
    emitc.call "printf" (%3) {args=["result: %ld\n", 0 : index]} : (!emitc.opaque<"long">) -> ()

    %idx = systemc.cpp.variable : index

    %4 = hw.constant 4 : i32
    %5 = emitc.call "malloc" (%4) : (i32) -> !emitc.ptr<!emitc.opaque<"void">>
    %6 = emitc.cast %5 : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<!emitc.opaque<"int">>
    %somePtr = systemc.cpp.variable %6 : !emitc.ptr<!emitc.opaque<"int">>
  }
}

systemc.module @CppEmission () {
  systemc.cpp.destructor {
    %0 = systemc.cpp.new() : () -> !emitc.ptr<!emitc.opaque<"submodule">>
    systemc.cpp.delete %0 : !emitc.ptr<!emitc.opaque<"submodule">>

    %1 = systemc.cpp.new() : () -> !emitc.ptr<!emitc.opaque<"int[5]">>
    %2 = emitc.cast %1 : !emitc.ptr<!emitc.opaque<"int[5]">> to !emitc.ptr<!emitc.opaque<"int">>
    %arr = systemc.cpp.variable %2 : !emitc.ptr<!emitc.opaque<"int">>
    systemc.cpp.delete %arr : !emitc.ptr<!emitc.opaque<"int">>

    %3 = hw.constant 1 : i32
    %4 = hw.constant 2 : i32
    %5 = systemc.cpp.new(%3, %4) : (i32, i32) -> !emitc.ptr<!emitc.opaque<"std::tuple<uint32_t, uint32_t>">>
    %tup = systemc.cpp.variable %5 : !emitc.ptr<!emitc.opaque<"std::tuple<uint32_t, uint32_t>">>
    systemc.cpp.delete %tup : !emitc.ptr<!emitc.opaque<"std::tuple<uint32_t, uint32_t>">>
  }
}

systemc.module @MemberAccess () {
  %member = systemc.cpp.variable : !emitc.opaque<"std::pair<int, int>">
  %c5 = "emitc.constant"() {value = #emitc.opaque<"5"> : !emitc.opaque<"int">} : () -> !emitc.opaque<"int">
  %result = systemc.cpp.variable : !emitc.opaque<"int">
  systemc.ctor {
    %0 = systemc.cpp.member_access %member dot "first" : (!emitc.opaque<"std::pair<int, int>">) -> !emitc.opaque<"int">
    systemc.cpp.assign %result = %0 : !emitc.opaque<"int">
    %1 = systemc.cpp.new (%c5, %c5) : (!emitc.opaque<"int">, !emitc.opaque<"int">) -> !emitc.ptr<!emitc.opaque<"std::pair<int, int>">>
    %2 = systemc.cpp.member_access %1 arrow "second" : (!emitc.ptr<!emitc.opaque<"std::pair<int, int>">>) -> !emitc.opaque<"int">
    systemc.cpp.assign %result = %2 : !emitc.opaque<"int">
  }
}

systemc.module @Sensitive (%in: !systemc.in<i1>, %inout: !systemc.inout<i1>) {
  systemc.ctor {
    systemc.method %update
    systemc.sensitive %in, %inout : !systemc.in<i1>, !systemc.inout<i1>
  }
  %update = systemc.func {}
}
