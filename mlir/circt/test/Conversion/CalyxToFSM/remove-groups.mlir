// RUN: circt-opt --split-input-file --calyx-remove-groups-fsm %s | FileCheck %s

// CHECK-LABEL:   fsm.machine @control(
// CHECK-SAME:        %[[VAL_0:.*]]: i1, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i8, %[[VAL_5:.*]]: i8) -> (i1, i1, i1, i1)
// CHECK:           %[[VAL_6:.*]] = hw.constant true
// CHECK:           %[[VAL_7:.*]] = hw.constant false
// CHECK:           fsm.state @fsm_entry output {
// CHECK:             fsm.output %[[VAL_7]], %[[VAL_7]], %[[VAL_7]], %[[VAL_7]] : i1, i1, i1, i1
// CHECK:           } transitions {
// CHECK:             fsm.transition @seq_0_A guard {
// CHECK:               %[[VAL_8:.*]] = comb.icmp eq %[[VAL_4]], %[[VAL_5]] : i8
// CHECK:               fsm.return %[[VAL_8]]
// CHECK:             }
// CHECK:           }
// CHECK:           fsm.state @fsm_exit output {
// CHECK:             fsm.output %[[VAL_7]], %[[VAL_7]], %[[VAL_7]], %[[VAL_6]] : i1, i1, i1, i1
// CHECK:           }
// CHECK:           fsm.state @seq_2_C output {
// CHECK:             fsm.output %[[VAL_6]], %[[VAL_7]], %[[VAL_7]], %[[VAL_7]] : i1, i1, i1, i1
// CHECK:           } transitions {
// CHECK:             fsm.transition @fsm_exit guard {
// CHECK:               fsm.return %[[VAL_0]]
// CHECK:             }
// CHECK:           }
// CHECK:           fsm.state @seq_1_B output {
// CHECK:             fsm.output %[[VAL_7]], %[[VAL_6]], %[[VAL_7]], %[[VAL_7]] : i1, i1, i1, i1
// CHECK:           } transitions {
// CHECK:             fsm.transition @seq_2_C guard {
// CHECK:               fsm.return %[[VAL_1]]
// CHECK:             }
// CHECK:           }
// CHECK:           fsm.state @seq_0_A output {
// CHECK:             fsm.output %[[VAL_7]], %[[VAL_7]], %[[VAL_6]], %[[VAL_7]] : i1, i1, i1, i1
// CHECK:           } transitions {
// CHECK:             fsm.transition @seq_1_B guard {
// CHECK:               fsm.return %[[VAL_2]]
// CHECK:             }
// CHECK:           }
// CHECK:         }

// CHECK-LABEL:   calyx.component @main(
// CHECK-SAME:                          %[[VAL_0:.*]]: i1 {go},
// CHECK-SAME:                          %[[VAL_1:.*]]: i1 {reset},
// CHECK-SAME:                          %[[VAL_2:.*]]: i1 {clk}) -> (
// CHECK-SAME:                          %[[VAL_3:.*]]: i1 {done}) {
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = calyx.std_wire @C_done : i1, i1
// CHECK:           %[[VAL_6:.*]]:4 = fsm.hw_instance "controller" @control(%[[VAL_5]], %[[VAL_7:.*]], %[[VAL_8:.*]], %[[VAL_0]], %[[VAL_9:.*]], %[[VAL_10:.*]]), clock %[[VAL_2]], reset %[[VAL_1]] : (i1, i1, i1, i1, i8, i8) -> (i1, i1, i1, i1)
// CHECK:           %[[VAL_11:.*]], %[[VAL_7]] = calyx.std_wire @B_done : i1, i1
// CHECK:           %[[VAL_12:.*]], %[[VAL_8]] = calyx.std_wire @A_done : i1, i1
// CHECK:           %[[VAL_13:.*]] = hw.constant 2 : i8
// CHECK:           %[[VAL_14:.*]] = hw.constant 1 : i8
// CHECK:           %[[VAL_15:.*]] = hw.constant 0 : i8
// CHECK:           %[[VAL_16:.*]] = hw.constant true
// CHECK:           %[[VAL_10]], %[[VAL_17:.*]], %[[VAL_18:.*]], %[[VAL_19:.*]], %[[VAL_20:.*]], %[[VAL_21:.*]] = calyx.register @a : i8, i1, i1, i1, i8, i1
// CHECK:           %[[VAL_9]], %[[VAL_22:.*]], %[[VAL_23:.*]], %[[VAL_24:.*]], %[[VAL_25:.*]], %[[VAL_26:.*]] = calyx.register @b : i8, i1, i1, i1, i8, i1
// CHECK:           %[[VAL_27:.*]], %[[VAL_28:.*]], %[[VAL_29:.*]], %[[VAL_30:.*]], %[[VAL_31:.*]], %[[VAL_32:.*]] = calyx.register @c : i8, i1, i1, i1, i8, i1
// CHECK:           calyx.wires {
// CHECK:             %[[VAL_33:.*]] = calyx.undef : i1
// CHECK:             calyx.assign %[[VAL_3]] = %[[VAL_6]]#3 : i1
// CHECK:             calyx.assign %[[VAL_12]] = %[[VAL_21]] : i1
// CHECK:             calyx.assign %[[VAL_11]] = %[[VAL_26]] : i1
// CHECK:             calyx.assign %[[VAL_4]] = %[[VAL_32]] : i1
// CHECK:             calyx.assign %[[VAL_10]] = %[[VAL_6]]#2 ? %[[VAL_15]] : i8
// CHECK:             calyx.assign %[[VAL_17]] = %[[VAL_6]]#2 ? %[[VAL_16]] : i1
// CHECK:             calyx.assign %[[VAL_9]] = %[[VAL_6]]#1 ? %[[VAL_14]] : i8
// CHECK:             calyx.assign %[[VAL_22]] = %[[VAL_6]]#1 ? %[[VAL_16]] : i1
// CHECK:             calyx.assign %[[VAL_27]] = %[[VAL_6]]#0 ? %[[VAL_13]] : i8
// CHECK:             calyx.assign %[[VAL_28]] = %[[VAL_6]]#0 ? %[[VAL_16]] : i1
// CHECK:           }
// CHECK:           calyx.control {
// CHECK:           }
// CHECK:         }

calyx.component @main(%go: i1 {go}, %reset: i1 {reset}, %clk: i1 {clk}) -> (%done: i1 {done}) {
  %c2_i8 = hw.constant 2 : i8
  %c1_i8 = hw.constant 1 : i8
  %c0_i8 = hw.constant 0 : i8
  %true = hw.constant true
  %a.in, %a.write_en, %a.clk, %a.reset, %a.out, %a.done = calyx.register @a : i8, i1, i1, i1, i8, i1
  %b.in, %b.write_en, %b.clk, %b.reset, %b.out, %b.done = calyx.register @b : i8, i1, i1, i1, i8, i1
  %c.in, %c.write_en, %c.clk, %c.reset, %c.out, %c.done = calyx.register @c : i8, i1, i1, i1, i8, i1
  calyx.wires {
    %0 = calyx.undef : i1
    calyx.group @A {
      calyx.assign %a.in =  %c0_i8 : i8
      calyx.assign %a.write_en = %true : i1
      calyx.group_done %a.done : i1
    }
    calyx.group @B {
      calyx.assign %b.in = %c1_i8 : i8
      calyx.assign %b.write_en = %true : i1
      calyx.group_done %b.done : i1
    }
    calyx.group @C {
      calyx.assign %c.in = %c2_i8 : i8
      calyx.assign %c.write_en = %true : i1
      calyx.group_done %c.done : i1
    }
  }
  calyx.control {
    fsm.machine @control(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1) -> (i1, i1, i1, i1) attributes
        {compiledGroups = [@C, @B, @A],
        calyx.fsm_group_done_inputs = {A = 2 : i64, B = 1 : i64, C = 0 : i64},
        calyx.fsm_group_go_outputs = {A = 2 : i64, B = 1 : i64, C = 0 : i64},
        calyx.fsm_top_level_done = 3 : i64,
        calyx.fsm_top_level_go = 3 : i64,
        initialState = "fsm_entry"} {
      %true_0 = hw.constant true
      %false = hw.constant false
      fsm.state @fsm_entry output {
        fsm.output %false, %false, %false, %false : i1, i1, i1, i1
      } transitions {
        fsm.transition @seq_0_A guard {
          %0 = comb.icmp eq %b.in, %a.in : i8
          fsm.return %0
        }
      }
      fsm.state @fsm_exit output {
        fsm.output %false, %false, %false, %true_0 : i1, i1, i1, i1
      } transitions {
      }
      fsm.state @seq_2_C output {
        fsm.output %true_0, %false, %false, %false : i1, i1, i1, i1
      } transitions {
        fsm.transition @fsm_exit guard {
          fsm.return %arg0
        }
      }
      fsm.state @seq_1_B output {
        fsm.output %false, %true_0, %false, %false : i1, i1, i1, i1
      } transitions {
        fsm.transition @seq_2_C guard {
          fsm.return %arg1
        }
      }
      fsm.state @seq_0_A output {
        fsm.output %false, %false, %true_0, %false : i1, i1, i1, i1
      } transitions {
        fsm.transition @seq_1_B guard {
          fsm.return %arg2
        }
      }
    }
  }
}
