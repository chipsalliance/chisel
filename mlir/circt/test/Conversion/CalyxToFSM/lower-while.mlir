// RUN: circt-opt --split-input-file -pass-pipeline='builtin.module(calyx.component(lower-calyx-to-fsm))' %s | FileCheck %s

// CHECK:      fsm.machine @control_main()
// CHECK-NEXT:   fsm.state @fsm_entry output {
// CHECK-NEXT:     fsm.output
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @seq_0_cond
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state @fsm_exit output {
// CHECK-NEXT:     fsm.output
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state @seq_1_while_header output {
// CHECK-NEXT:     fsm.output
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @seq_1_while_if guard {
// CHECK-NEXT:       fsm.return %lt_reg.out
// CHECK-NEXT:     }
// CHECK-NEXT:     fsm.transition @fsm_exit guard {
// CHECK-NEXT:       %true_0 = hw.constant true
// CHECK-NEXT:       %0 = comb.xor %lt_reg.out, %true_0 : i1
// CHECK-NEXT:       fsm.return %0
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state @seq_1_while_if output {
// CHECK-NEXT:     fsm.output
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @seq_1_while_if_then guard {
// CHECK-NEXT:       fsm.return %lt_reg.out
// CHECK-NEXT:     }
// CHECK-NEXT:     fsm.transition @seq_1_while_if_else guard {
// CHECK-NEXT:       %true_0 = hw.constant true
// CHECK-NEXT:       %0 = comb.xor %lt_reg.out, %true_0 : i1
// CHECK-NEXT:       fsm.return %0
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state @seq_1_while_if_then output {
// CHECK-NEXT:     fsm.output
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @seq_1_while_if_then_seq_0_do_add
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state @seq_1_while_if_then_seq_1_do_add output {
// CHECK-NEXT:     calyx.enable @do_add
// CHECK-NEXT:     fsm.output
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @seq_1_while_header
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state @seq_1_while_if_then_seq_0_do_add output {
// CHECK-NEXT:     calyx.enable @do_add
// CHECK-NEXT:     fsm.output
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @seq_1_while_if_then_seq_1_do_add
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state @seq_1_while_if_else output {
// CHECK-NEXT:     fsm.output
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @seq_1_while_if_else_seq_0_do_add
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state @seq_1_while_if_else_seq_0_do_add output {
// CHECK-NEXT:     calyx.enable @do_add
// CHECK-NEXT:     fsm.output
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @seq_1_while_header
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state @seq_0_cond output {
// CHECK-NEXT:     calyx.enable @cond
// CHECK-NEXT:     fsm.output
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @seq_1_while_header
// CHECK-NEXT:   }
// CHECK-NEXT: }

calyx.component @main(%go: i1 {go}, %reset: i1 {reset}, %clk: i1 {clk}) -> (%done: i1 {done}) {
  %true = hw.constant true
  %c1_i32 = hw.constant 1 : i32
  %c5_i32 = hw.constant 5 : i32
  %c4_i32 = hw.constant 4 : i32
  %lt_reg.in, %lt_reg.write_en, %lt_reg.clk, %lt_reg.reset, %lt_reg.out, %lt_reg.done = calyx.register @lt_reg : i1, i1, i1, i1, i1, i1
  %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i32, i1, i1, i1, i32, i1
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32
  %lt.left, %lt.right, %lt.out = calyx.std_lt @lt : i32, i32, i1
  calyx.wires {
    %0 = calyx.undef : i1
    calyx.group @do_add {
      %do_add.go = calyx.group_go %0 : i1
      calyx.assign %add.right = %do_add.go ? %c4_i32 : i32
      calyx.assign %add.left = %do_add.go ? %c4_i32 : i32
      calyx.assign %r.in = %do_add.go ? %add.out : i32
      calyx.assign %r.write_en = %do_add.go ? %true : i1
      calyx.group_done %r.done : i1
    }
    calyx.group @cond {
      %cond.go = calyx.group_go %0 : i1
      calyx.assign %lt_reg.in = %cond.go ? %lt.out : i1
      calyx.assign %lt_reg.write_en = %cond.go ? %true : i1
      calyx.assign %lt.right = %cond.go ? %c5_i32 : i32
      calyx.assign %lt.left = %cond.go ? %c1_i32 : i32
      calyx.group_done %lt_reg.done ? %true : i1
    }
  }
  calyx.control {
    calyx.seq {
      calyx.enable @cond
      calyx.while %lt_reg.out {
        calyx.if %lt_reg.out {
          calyx.seq {
            calyx.enable @do_add
            calyx.enable @do_add
          }
        } else {
          calyx.seq {
            calyx.enable @do_add
          }
        }
      }
    }
  }
}
