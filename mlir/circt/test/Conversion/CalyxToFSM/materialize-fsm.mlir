// RUN: circt-opt -pass-pipeline='builtin.module(calyx.component(materialize-calyx-to-fsm))' %s | FileCheck %s

// CHECK: fsm.machine @control(%[[A_DONE:.*]]: i1, %[[B_DONE:.*]]: i1, %[[COND_DONE:.*]]: i1, %[[TOP_LEVEL_GO:.*]]: i1) -> (i1, i1, i1, i1)
// CHECK-NEXT:   %[[C1:.*]] = hw.constant true
// CHECK-NEXT:   %[[C0:.*]] = hw.constant false
// CHECK-NEXT:   fsm.state @fsm_entry output {
// CHECK-NEXT:     fsm.output %[[C0]], %[[C0]], %[[C0]], %[[C0]] : i1, i1, i1, i1
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @[[SEQ_0_COND:.*]] guard {
// CHECK-NEXT:       fsm.return %[[TOP_LEVEL_GO]]
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state @[[FSM_EXIT:.*]] output {
// CHECK-NEXT:     fsm.output %[[C0]], %[[C0]], %[[C0]], %[[C1]] : i1, i1, i1, i1
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state @[[SEQ_1_IF:.*]] output {
// CHECK-NEXT:     fsm.output %[[C0]], %[[C0]], %[[C0]], %[[C0]] : i1, i1, i1, i1
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @[[SEQ_1_IF_THEN_A:.*]] guard {
// CHECK-NEXT:       fsm.return %lt_reg.out
// CHECK-NEXT:     }
// CHECK-NEXT:     fsm.transition @[[SEQ_1_IF_ELSE_B:.*]] guard {
// CHECK-NEXT:       %true_2 = hw.constant true
// CHECK-NEXT:       %0 = comb.xor %lt_reg.out, %true_2 : i1
// CHECK-NEXT:       fsm.return %0
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state @[[SEQ_1_IF_THEN_A]] output {
// CHECK-NEXT:     fsm.output %[[C1]], %[[C0]], %[[C0]], %[[C0]] : i1, i1, i1, i1
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @[[FSM_EXIT]] guard {
// CHECK-NEXT:       fsm.return %[[A_DONE]]
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state @[[SEQ_1_IF_ELSE_B]] output {
// CHECK-NEXT:     fsm.output %[[C0]], %[[C1]], %[[C0]], %[[C0]] : i1, i1, i1, i1
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @[[FSM_EXIT]] guard {
// CHECK-NEXT:       fsm.return %[[B_DONE]]
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state @[[SEQ_0_COND]] output {
// CHECK-NEXT:     fsm.output %[[C0]], %[[C0]], %[[C1]], %[[C0]] : i1, i1, i1, i1
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @[[SEQ_1_IF]] guard {
// CHECK-NEXT:       fsm.return %[[COND_DONE]]
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

calyx.component @main(%go: i1 {go}, %reset: i1 {reset}, %clk: i1 {clk}) -> (%done: i1 {done}) {
  %false = hw.constant false
  %true = hw.constant true
  %lt_reg.in, %lt_reg.write_en, %lt_reg.clk, %lt_reg.reset, %lt_reg.out, %lt_reg.done = calyx.register @lt_reg : i1, i1, i1, i1, i1, i1
  %t.in, %t.write_en, %t.clk, %t.reset, %t.out, %t.done = calyx.register @t : i1, i1, i1, i1, i1, i1
  %f.in, %f.write_en, %f.clk, %f.reset, %f.out, %f.done = calyx.register @f : i1, i1, i1, i1, i1, i1
  %lt.left, %lt.right, %lt.out = calyx.std_lt @lt : i1, i1, i1
  calyx.wires {
    %0 = calyx.undef : i1
    calyx.group @A {
      %A.go = calyx.group_go %0 : i1
      calyx.assign %t.in = %A.go ? %true : i1
      calyx.assign %t.write_en = %A.go ? %true : i1
      calyx.group_done %t.done : i1
    }
    calyx.group @B {
      %B.go = calyx.group_go %0 : i1
      calyx.assign %f.in = %B.go ? %true : i1
      calyx.assign %f.write_en = %B.go ? %true : i1
      calyx.group_done %f.done : i1
    }
    calyx.group @cond {
      %cond.go = calyx.group_go %0 : i1
      calyx.assign %lt_reg.in = %cond.go ? %lt.out : i1
      calyx.assign %lt_reg.write_en = %cond.go ? %true : i1
      calyx.assign %lt.left = %cond.go ? %true : i1
      calyx.assign %lt.right = %cond.go ? %false : i1
      calyx.group_done %lt_reg.done ? %true : i1
    }
  }
  calyx.control {
    fsm.machine @control() attributes {compiledGroups = [@A, @B, @cond], initialState = "fsm_entry"} {
      fsm.state @fsm_entry transitions {
        fsm.transition @seq_0_cond
      }

      fsm.state @fsm_exit

      fsm.state @seq_1_if transitions {
        fsm.transition @seq_1_if_then_seq_0_A guard {
          fsm.return %lt_reg.out
        }
        fsm.transition @seq_1_if_else_seq_0_B guard {
          %true_0 = hw.constant true
          %0 = comb.xor %lt_reg.out, %true_0 : i1
          fsm.return %0
        }
      }
      fsm.state @seq_1_if_then_seq_0_A output {
        calyx.enable @A
        fsm.output
      } transitions {
        fsm.transition @fsm_exit
      }
      fsm.state @seq_1_if_else_seq_0_B output {
        calyx.enable @B
        fsm.output
      } transitions {
        fsm.transition @fsm_exit
      }
      fsm.state @seq_0_cond output {
        calyx.enable @cond
        fsm.output
      } transitions {
        fsm.transition @seq_1_if
      }
    }
  }
}
