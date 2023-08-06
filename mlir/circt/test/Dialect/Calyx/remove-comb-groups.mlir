// RUN: circt-opt --split-input-file --calyx-remove-comb-groups %s --canonicalize | FileCheck %s

// CHECK-LABEL: calyx.component @main
calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
  %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
  %eq.left, %eq.right, %eq.out = calyx.std_eq @eq : i1, i1, i1
  %c1_1 = hw.constant 1 : i1
  calyx.wires {
// CHECK-NOT: calyx.comb_group @Cond
// CHECK-LABEL: calyx.group @Cond {
// CHECK:   calyx.assign %eq_reg.in = %eq.out : i1
// CHECK:   calyx.assign %eq_reg.write_en = %true : i1
// CHECK:   calyx.assign %eq.left = %true : i1
// CHECK:   calyx.assign %eq.right = %true : i1
// CHECK:   calyx.group_done %eq_reg.done ? %true : i1
    calyx.comb_group @Cond {
      calyx.assign %eq.left =  %c1_1 : i1
      calyx.assign %eq.right = %c1_1 : i1
    }
    calyx.group @A {
      calyx.assign %r.in = %c1_1 : i1
      calyx.assign %r.write_en = %c1_1 : i1
      calyx.group_done %r.done : i1
    }
  }

// CHECK: calyx.control {
// CHECK:   calyx.seq {
// CHECK:     calyx.enable @Cond
// CHECK:     calyx.if %eq_reg.out {
// CHECK:       calyx.seq {
// CHECK:         calyx.enable @A
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK: }
  calyx.control {
    calyx.if %eq.out with @Cond {
      calyx.seq {
        calyx.enable @A
      }
    }
  }
}

// -----

// Test the case where cells are shared across combinational groups.

calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
  %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
  %eq.left, %eq.right, %eq.out = calyx.std_eq @eq : i1, i1, i1
  %c1_1 = hw.constant 1 : i1
  calyx.wires {

// CHECK-NOT: calyx.comb_group @Cond1
// CHECK: calyx.group @Cond1 {
// CHECK:   calyx.assign %eq_reg.in = %eq.out : i1
// CHECK:   calyx.assign %eq_reg.write_en = %true : i1
// CHECK:   calyx.assign %eq.left = %true : i1
// CHECK:   calyx.assign %eq.right = %true : i1
// CHECK:   calyx.group_done %eq_reg.done ? %true : i1
// CHECK: }
    calyx.comb_group @Cond1 {
      calyx.assign %eq.left =  %c1_1 : i1
      calyx.assign %eq.right = %c1_1 : i1
    }

// CHECK-NOT: calyx.comb_group @Cond2
// CHECK: calyx.group @Cond2 {
// CHECK:   calyx.assign %eq_reg.in = %eq.out : i1
// CHECK:   calyx.assign %eq_reg.write_en = %true : i1
// CHECK:   calyx.assign %eq.left = %true : i1
// CHECK:   calyx.assign %eq.right = %true : i1
// CHECK:   calyx.group_done %eq_reg.done ? %true : i1
// CHECK: }
    calyx.comb_group @Cond2 {
      calyx.assign %eq.left =  %c1_1 : i1
      calyx.assign %eq.right = %c1_1 : i1
    }
    calyx.group @A {
      calyx.assign %r.in = %c1_1 : i1
      calyx.assign %r.write_en = %c1_1 : i1
      calyx.group_done %r.done : i1
    }
  }

// CHECK: calyx.control {
// CHECK:   calyx.par {
// CHECK:     calyx.enable @Cond1
// CHECK:     calyx.if %eq_reg.out {
// CHECK:       calyx.seq {
// CHECK:         calyx.enable @A
// CHECK:       }
// CHECK:     }
// CHECK:     calyx.enable @Cond2
// CHECK:     calyx.while %eq_reg.out {
// CHECK:       calyx.seq {
// CHECK:         calyx.enable @A
// CHECK:         calyx.enable @Cond2
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK: }
  calyx.control {
    calyx.par {
      calyx.if %eq.out with @Cond1 {
        calyx.seq {
          calyx.enable @A
        }
      }
      calyx.while %eq.out with @Cond2 {
        calyx.seq {
          calyx.enable @A
        }
      }
    }
  }
}

