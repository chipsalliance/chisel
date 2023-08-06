// REQUIRES: verilator
// RUN: circt-opt %s -export-verilog -verify-diagnostics -o %t2.mlir > %t1.sv
// RUN: verilator --lint-only --top-module A %t1.sv
// RUN: verilator --lint-only --top-module AB %t1.sv
// RUN: verilator --lint-only --top-module shl %t1.sv
// RUN: verilator --lint-only -Wno-WIDTH --top-module TESTSIMPLE %t1.sv
// RUN: verilator --lint-only --top-module casts %t1.sv
// RUN: verilator --lint-only --top-module exprInlineTestIssue439 %t1.sv
// RUN: verilator --lint-only --top-module StructDecls %t1.sv

hw.module @B(%a: i1) -> (b: i1, c: i1) {
  %0 = comb.or %a, %a : i1
  %1 = comb.and %a, %a : i1
  hw.output %0, %1 : i1, i1
}

hw.module @A(%d: i1, %e: i1) -> (f: i1) {
  %1 = comb.mux %d, %d, %e : i1
  hw.output %1 : i1
}

hw.module @AAA(%d: i1, %e: i1) -> (f: i1) {
  %z = hw.constant 0 : i1
  hw.output %z : i1
}

hw.module @AB(%w: i1, %x: i1) -> (y: i1, z: i1) {
  %w2 = hw.instance "a1" @AAA(d: %w: i1, e: %w1: i1) -> (f: i1)
  %w1, %y = hw.instance "b1" @B(a: %w2: i1) -> (b: i1, c: i1)
  hw.output %y, %x : i1, i1
}

hw.module @shl(%a: i1) -> (b: i1) {
  %0 = comb.shl %a, %a : i1
  hw.output %0 : i1
}

hw.module @TESTSIMPLE(%a: i4, %b: i4, %cond: i1, %array: !hw.array<10xi4>,
                        %uarray: !hw.uarray<16xi8>) -> (
  r0: i4, r1: i4, r2: i4, r3: i4,
  r4: i4, r5: i4, r6: i4, r7: i4,
  r8: i4, r9: i4, r10: i4, r11: i4,
  r12: i4, r13: i1,
  r14: i1, r15: i1, r16: i1, r17: i1,
  r18: i1, r19: i1, r20: i1, r21: i1,
  r22: i1, r23: i1,
  r24: i12, r25: i2, r27: i4, r28: i4,
  r29: !hw.array<3xi4>
  ) {

  %0 = comb.add %a, %b : i4
  %1 = comb.sub %a, %b : i4
  %2 = comb.mul %a, %b : i4
  %3 = comb.divu %a, %b : i4
  %4 = comb.divs %a, %b : i4
  %5 = comb.modu %a, %b : i4
  %6 = comb.mods %a, %b : i4
  %7 = comb.shl %a, %b : i4
  %8 = comb.shru %a, %b : i4
  %9 = comb.shrs %a, %b : i4
  %10 = comb.or %a, %b : i4
  %11 = comb.and %a, %b : i4
  %12 = comb.xor %a, %b : i4
  %13 = comb.icmp eq %a, %b : i4
  %14 = comb.icmp ne %a, %b : i4
  %15 = comb.icmp slt %a, %b : i4
  %16 = comb.icmp sle %a, %b : i4
  %17 = comb.icmp sgt %a, %b : i4
  %18 = comb.icmp sge %a, %b : i4
  %19 = comb.icmp ult %a, %b : i4
  %20 = comb.icmp ule %a, %b : i4
  %21 = comb.icmp ugt %a, %b : i4
  %22 = comb.icmp uge %a, %b : i4
  %23 = comb.parity %a : i4
  %24 = comb.concat %a, %a, %b : i4, i4, i4
  %25 = comb.extract %a from 1 : (i4) -> i2
  %27 = comb.mux %cond, %a, %b : i4

  %allone = hw.constant 15 : i4
  %28 = comb.xor %a, %allone : i4

  %one = hw.constant 1 : i4
  %aPlusOne = comb.add %a, %one : i4
  sv.verbatim "/* verilator lint_off WIDTH */"
  %29 = hw.array_slice %array[%aPlusOne]: (!hw.array<10xi4>) -> !hw.array<3xi4>


  hw.output %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %27, %28, %29:
    i4,i4, i4,i4,i4,i4,i4, i4,i4,i4,i4,i4,
    i4,i1,i1,i1,i1, i1,i1,i1,i1,i1, i1,i1,
    i12,i2, i4, i4, !hw.array<3xi4>
}

hw.module @exprInlineTestIssue439(%clk: i1) {
  %c = hw.constant 0 : i32

  sv.always posedge %clk {
    %e = comb.extract %c from 0 : (i32) -> i16
    %f = comb.add %e, %e : i16
    %fd = hw.constant 0x80000002 : i32
    sv.fwrite %fd, "%d"(%f) : i16
  }
}

hw.module @casts(%in1: i64) -> (r1: !hw.array<5xi8>) {
  %bits = hw.bitcast %in1 : (i64) -> !hw.array<64xi1>
  %idx = hw.constant 10 : i6
  %midBits = hw.array_slice %bits[%idx] : (!hw.array<64xi1>) -> !hw.array<40xi1>
  %r1 = hw.bitcast %midBits : (!hw.array<40xi1>) -> !hw.array<5xi8>
  hw.output %r1 : !hw.array<5xi8>
}

hw.module @StructDecls() {
  %reg1 = sv.reg : !hw.inout<struct<a: i1, b: i1>>
  %reg2 = sv.reg : !hw.inout<array<8xstruct<a: i1, b: i1>>>
}

hw.module @UniformArrayCreate() -> (arr: !hw.array<5xi8>) {
  %c0_i8 = hw.constant 0 : i8
  %arr = hw.array_create %c0_i8, %c0_i8, %c0_i8, %c0_i8, %c0_i8 : i8
  hw.output %arr : !hw.array<5xi8>
}
