// RUN: circt-opt -pass-pipeline='builtin.module(calyx.component(calyx-gicm))' %s | FileCheck %s

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %add.left, %add.right, %add.out = calyx.std_add @add : i8, i8, i8
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i8, i1, i1, i1, i8, i1

    %c9_i8 = hw.constant 9 : i8
    %true = hw.constant 1 : i1
    %c8_i6 = hw.constant 8 : i8

    calyx.wires {
      // CHECK: %[[ICMP2_GROUP2:.*]] = comb.icmp slt %r.out, %c8_i8 : i8
      // CHECK: %[[ICMP1_GROUP2:.*]] = comb.icmp slt %[[ADD_GROUP2:.*]], %c8_i8 : i8
      // CHECK: %[[ADD_GROUP2]] = comb.add %r.out, %c8_i8 : i8
      // CHECK: %[[ICMP_GROUP1:.*]] = comb.icmp slt %r.out, %c8_i8 : i8
      // CHECK-LABEL: calyx.group @Group1
      calyx.group @Group1 {
        // CHECK: calyx.assign %add.left = %[[ICMP_GROUP1]] ? %c8_i8 : i8
        %0 = comb.icmp slt %r.out, %c8_i6 : i8
        calyx.assign %add.left = %0 ? %c8_i6 : i8
        calyx.assign %add.right = %r.out : i8
        calyx.assign %r.in = %add.out : i8
        calyx.assign %r.write_en = %true : i1
        calyx.group_done %r.done : i1
      }
      // CHECK-LABEL: calyx.group @Group2 {
      calyx.group @Group2 {
        // CHECK: calyx.assign %add.left = %[[ICMP1_GROUP2]] ? %c8_i8 : i8
        // CHECK: calyx.assign %add.right = %[[ICMP2_GROUP2]] ? %r.out : i8
        %0 = comb.add %r.out, %c8_i6 : i8
        %1 = comb.icmp slt %0, %c8_i6 : i8
        %2 = comb.icmp slt %r.out, %c8_i6 : i8
        calyx.assign %add.left = %1 ? %c8_i6 : i8
        calyx.assign %add.right = %2 ? %r.out : i8
        calyx.assign %r.in = %add.out : i8
        calyx.assign %r.write_en = %true : i1
        calyx.group_done %r.done : i1
      }
    }
    calyx.control {
      calyx.seq {
        calyx.enable @Group1
        calyx.enable @Group2
      }
    }
  }
}
