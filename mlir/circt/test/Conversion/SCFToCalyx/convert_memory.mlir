// RUN: circt-opt %s --lower-scf-to-calyx -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL:   calyx.component @main(
// CHECK-SAME:                          %[[VAL_0:.*]]: i1 {clk},
// CHECK-SAME:                          %[[VAL_1:.*]]: i1 {reset},
// CHECK-SAME:                          %[[VAL_2:.*]]: i1 {go}) -> (
// CHECK-SAME:                          %[[VAL_3:.*]]: i1 {done}) {
// CHECK:           %[[VAL_4:.*]] = hw.constant true
// CHECK:           %[[VAL_5:.*]] = hw.constant 64 : i32
// CHECK:           %[[VAL_6:.*]] = hw.constant 1 : i32
// CHECK:           %[[VAL_7:.*]] = hw.constant 0 : i32
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = calyx.std_slice @std_slice_1 : i32, i6
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = calyx.std_slice @std_slice_0 : i32, i6
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]], %[[VAL_14:.*]] = calyx.std_add @std_add_0 : i32, i32, i32
// CHECK:           %[[VAL_15:.*]], %[[VAL_16:.*]], %[[VAL_17:.*]] = calyx.std_lt @std_lt_0 : i32, i32, i1
// CHECK:           %[[VAL_18:.*]], %[[VAL_19:.*]], %[[VAL_20:.*]], %[[VAL_21:.*]], %[[VAL_22:.*]], %[[VAL_23:.*]], %[[VAL_24:.*]], %[[VAL_25:.*]] = calyx.seq_mem @mem_1 <[64] x 32> [6] {external = true} : i6, i32, i1, i1, i1, i32, i1, i1
// CHECK:           %[[VAL_26:.*]], %[[VAL_27:.*]], %[[VAL_28:.*]], %[[VAL_29:.*]], %[[VAL_30:.*]], %[[VAL_31:.*]], %[[VAL_32:.*]], %[[VAL_33:.*]] = calyx.seq_mem @mem_0 <[64] x 32> [6] {external = true} : i6, i32, i1, i1, i1, i32, i1, i1
// CHECK:           %[[VAL_34:.*]], %[[VAL_35:.*]], %[[VAL_36:.*]], %[[VAL_37:.*]], %[[VAL_38:.*]], %[[VAL_39:.*]] = calyx.register @while_0_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK:           calyx.wires {
// CHECK:             calyx.group @assign_while_0_init_0 {
// CHECK:               calyx.assign %[[VAL_34]] = %[[VAL_7]] : i32
// CHECK:               calyx.assign %[[VAL_35]] = %[[VAL_4]] : i1
// CHECK:               calyx.group_done %[[VAL_39]] : i1
// CHECK:             }
// CHECK:             calyx.comb_group @bb0_0 {
// CHECK:               calyx.assign %[[VAL_15]] = %[[VAL_38]] : i32
// CHECK:               calyx.assign %[[VAL_16]] = %[[VAL_5]] : i32
// CHECK:             }
// CHECK:             calyx.group @bb0_1 {
// CHECK:               calyx.assign %[[VAL_8]] = %[[VAL_38]] : i32
// CHECK:               calyx.assign %[[VAL_26]] = %[[VAL_9]] : i6
// CHECK:               calyx.assign %[[VAL_32]] = %[[VAL_4]] : i1
// CHECK:               calyx.group_done %[[VAL_33]] : i1
// CHECK:             }
// CHECK:             calyx.group @bb0_2 {
// CHECK:               calyx.assign %[[VAL_10]] = %[[VAL_38]] : i32
// CHECK:               calyx.assign %[[VAL_18]] = %[[VAL_11]] : i6
// CHECK:               calyx.assign %[[VAL_19]] = %[[VAL_31]] : i32
// CHECK:               calyx.assign %[[VAL_20]] = %[[VAL_4]] : i1
// CHECK:               calyx.group_done %[[VAL_21]] : i1
// CHECK:             }
// CHECK:             calyx.group @assign_while_0_latch {
// CHECK:               calyx.assign %[[VAL_34]] = %[[VAL_14]] : i32
// CHECK:               calyx.assign %[[VAL_35]] = %[[VAL_4]] : i1
// CHECK:               calyx.assign %[[VAL_12]] = %[[VAL_38]] : i32
// CHECK:               calyx.assign %[[VAL_13]] = %[[VAL_6]] : i32
// CHECK:               calyx.group_done %[[VAL_39]] : i1
// CHECK:             }
// CHECK:           }
// CHECK:           calyx.control {
// CHECK:             calyx.seq {
// CHECK:               calyx.enable @assign_while_0_init_0
// CHECK:               calyx.while %[[VAL_17]] with @bb0_0 {
// CHECK:                 calyx.seq {
// CHECK:                   calyx.enable @bb0_1
// CHECK:                   calyx.enable @bb0_2
// CHECK:                   calyx.enable @assign_while_0_latch
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         } {toplevel}
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %0 = memref.alloc() : memref<64xi32>
    %1 = memref.alloc() : memref<64xi32>
    scf.while(%arg0 = %c0) : (index) -> (index) {
      %cond = arith.cmpi ult, %arg0, %c64 : index
      scf.condition(%cond) %arg0 : index
    } do {
    ^bb0(%arg1: index):
      %v = memref.load %0[%arg1] : memref<64xi32>
      memref.store %v, %1[%arg1] : memref<64xi32>
      %inc = arith.addi %arg1, %c1 : index
      scf.yield %inc : index
    }
    return
  }
}

// -----

// Test combinational value used across sequential group boundary. This requires
// that any referenced combinational assignments are re-applied in each
// sequential group.

// CHECK-LABEL:   calyx.component @main(
// CHECK-SAME:                          %[[VAL_0:in0]]: i32,
// CHECK-SAME:                          %[[VAL_1:.*]]: i1 {clk},
// CHECK-SAME:                          %[[VAL_2:.*]]: i1 {reset},
// CHECK-SAME:                          %[[VAL_3:.*]]: i1 {go}) -> (
// CHECK-SAME:                          %[[VAL_4:.*]]: i32,
// CHECK-SAME:                          %[[VAL_5:.*]]: i1 {done}) {
// CHECK:           %[[VAL_6:.*]] = hw.constant true
// CHECK:           %[[VAL_7:.*]] = hw.constant 1 : i32
// CHECK:           %[[VAL_8:.*]] = hw.constant 0 : i32
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = calyx.std_slice @std_slice_0 : i32, i6
// CHECK:           %[[VAL_11:.*]], %[[VAL_12:.*]], %[[VAL_13:.*]] = calyx.std_add @std_add_1 : i32, i32, i32
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]], %[[VAL_16:.*]] = calyx.std_add @std_add_0 : i32, i32, i32
// CHECK:           %[[VAL_17:.*]], %[[VAL_18:.*]], %[[VAL_19:.*]], %[[VAL_20:.*]], %[[VAL_21:.*]], %[[VAL_22:.*]], %[[VAL_23:.*]], %[[VAL_24:.*]] = calyx.seq_mem @mem_0 <[64] x 32> [6] {external = true} : i6, i32, i1, i1, i1, i32, i1, i1
// CHECK:           %[[VAL_25:.*]], %[[VAL_26:.*]], %[[VAL_27:.*]], %[[VAL_28:.*]], %[[VAL_29:.*]], %[[VAL_30:.*]] = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK:           calyx.wires {
// CHECK:             calyx.assign %[[VAL_4]] = %[[VAL_29]] : i32
// CHECK:             calyx.group @bb0_1 {
// CHECK:               calyx.assign %[[VAL_9]] = %[[VAL_8]] : i32
// CHECK:               calyx.assign %[[VAL_17]] = %[[VAL_10]] : i6
// CHECK:               calyx.assign %[[VAL_18]] = %[[VAL_16]] : i32
// CHECK:               calyx.assign %[[VAL_19]] = %[[VAL_6]] : i1
// CHECK:               calyx.assign %[[VAL_14]] = %[[VAL_0]] : i32
// CHECK:               calyx.assign %[[VAL_15]] = %[[VAL_7]] : i32
// CHECK:               calyx.group_done %[[VAL_20]] : i1
// CHECK:             }
// CHECK:             calyx.group @ret_assign_0 {
// CHECK:               calyx.assign %[[VAL_25]] = %[[VAL_13]] : i32
// CHECK:               calyx.assign %[[VAL_26]] = %[[VAL_6]] : i1
// CHECK:               calyx.assign %[[VAL_11]] = %[[VAL_16]] : i32
// CHECK:               calyx.assign %[[VAL_14]] = %[[VAL_0]] : i32
// CHECK:               calyx.assign %[[VAL_15]] = %[[VAL_7]] : i32
// CHECK:               calyx.assign %[[VAL_12]] = %[[VAL_7]] : i32
// CHECK:               calyx.group_done %[[VAL_30]] : i1
// CHECK:             }
// CHECK:           }
// CHECK:           calyx.control {
// CHECK:             calyx.seq {
// CHECK:               calyx.enable @bb0_1
// CHECK:               calyx.enable @ret_assign_0
// CHECK:             }
// CHECK:           }
// CHECK:         } {toplevel}
module {
  func.func @main(%arg0 : i32) -> i32 {
    %0 = memref.alloc() : memref<64xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32
    %1 = arith.addi %arg0, %c1 : i32
    memref.store %1, %0[%c0] : memref<64xi32>
    %3 = arith.addi %1, %c1 : i32
    return %3 : i32
  }
}

// -----

// CHECK-LABEL:   calyx.component @main(
// CHECK-SAME:                          %[[VAL_0:in0]]: i32,
// CHECK-SAME:                          %[[VAL_1:.*]]: i1 {clk},
// CHECK-SAME:                          %[[VAL_2:.*]]: i1 {reset},
// CHECK-SAME:                          %[[VAL_3:.*]]: i1 {go}) -> (
// CHECK-SAME:                          %[[VAL_4:.*]]: i32,
// CHECK-SAME:                          %[[VAL_5:.*]]: i1 {done}) {
// CHECK:           %[[VAL_6:.*]] = hw.constant true
// CHECK:           %[[VAL_7:.*]] = hw.constant 1 : i32
// CHECK:           %[[VAL_8:.*]] = hw.constant 0 : i32
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = calyx.std_slice @std_slice_0 : i32, i6
// CHECK:           %[[VAL_11:.*]], %[[VAL_12:.*]], %[[VAL_13:.*]] = calyx.std_add @std_add_2 : i32, i32, i32
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]], %[[VAL_16:.*]] = calyx.std_add @std_add_1 : i32, i32, i32
// CHECK:           %[[VAL_17:.*]], %[[VAL_18:.*]], %[[VAL_19:.*]] = calyx.std_add @std_add_0 : i32, i32, i32
// CHECK:           %[[VAL_20:.*]], %[[VAL_21:.*]], %[[VAL_22:.*]], %[[VAL_23:.*]], %[[VAL_24:.*]], %[[VAL_25:.*]], %[[VAL_26:.*]], %[[VAL_27:.*]] = calyx.seq_mem @mem_0 <[64] x 32> [6] {external = true} : i6, i32, i1, i1, i1, i32, i1, i1
// CHECK:           %[[VAL_28:.*]], %[[VAL_29:.*]], %[[VAL_30:.*]], %[[VAL_31:.*]], %[[VAL_32:.*]], %[[VAL_33:.*]] = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK:           calyx.wires {
// CHECK:             calyx.assign %[[VAL_4]] = %[[VAL_32]] : i32
// CHECK:             calyx.group @bb0_2 {
// CHECK:               calyx.assign %[[VAL_9]] = %[[VAL_8]] : i32
// CHECK:               calyx.assign %[[VAL_20]] = %[[VAL_10]] : i6
// CHECK:               calyx.assign %[[VAL_21]] = %[[VAL_19]] : i32
// CHECK:               calyx.assign %[[VAL_22]] = %[[VAL_6]] : i1
// CHECK:               calyx.assign %[[VAL_17]] = %[[VAL_0]] : i32
// CHECK:               calyx.assign %[[VAL_18]] = %[[VAL_7]] : i32
// CHECK:               calyx.group_done %[[VAL_23]] : i1
// CHECK:             }
// CHECK:             calyx.group @ret_assign_0 {
// CHECK:               calyx.assign %[[VAL_28]] = %[[VAL_13]] : i32
// CHECK:               calyx.assign %[[VAL_29]] = %[[VAL_6]] : i1
// CHECK:               calyx.assign %[[VAL_11]] = %[[VAL_16]] : i32
// CHECK:               calyx.assign %[[VAL_14]] = %[[VAL_19]] : i32
// CHECK:               calyx.assign %[[VAL_17]] = %[[VAL_0]] : i32
// CHECK:               calyx.assign %[[VAL_18]] = %[[VAL_7]] : i32
// CHECK:               calyx.assign %[[VAL_15]] = %[[VAL_7]] : i32
// CHECK:               calyx.assign %[[VAL_12]] = %[[VAL_7]] : i32
// CHECK:               calyx.group_done %[[VAL_33]] : i1
// CHECK:             }
// CHECK:           }
// CHECK:           calyx.control {
// CHECK:             calyx.seq {
// CHECK:               calyx.enable @bb0_2
// CHECK:               calyx.enable @ret_assign_0
// CHECK:             }
// CHECK:           }
// CHECK:         } {toplevel}
module {
  func.func @main(%arg0 : i32) -> i32 {
    %0 = memref.alloc() : memref<64xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32
    %1 = arith.addi %arg0, %c1 : i32
    %2 = arith.addi %1, %c1 : i32
    memref.store %1, %0[%c0] : memref<64xi32>
    %3 = arith.addi %2, %c1 : i32
    return %3 : i32
  }
}

// -----
// Test multiple reads from the same memory (structural hazard).

// CHECK-LABEL:   calyx.component @main(
// CHECK-SAME:                          %[[VAL_0:.*]]: i6,
// CHECK-SAME:                          %[[VAL_1:.*]]: i1 {clk},
// CHECK-SAME:                          %[[VAL_2:.*]]: i1 {reset},
// CHECK-SAME:                          %[[VAL_3:.*]]: i1 {go}) -> (
// CHECK-SAME:                          %[[VAL_4:.*]]: i32,
// CHECK-SAME:                          %[[VAL_5:.*]]: i1 {done}) {
// CHECK:           %[[VAL_6:.*]] = hw.constant true
// CHECK:           %[[VAL_7:.*]] = hw.constant 1 : i32
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = calyx.std_slice @std_slice_1 : i32, i6
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = calyx.std_slice @std_slice_0 : i32, i6
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]], %[[VAL_14:.*]] = calyx.std_add @std_add_0 : i32, i32, i32
// CHECK:           %[[VAL_15:.*]], %[[VAL_16:.*]], %[[VAL_17:.*]], %[[VAL_18:.*]], %[[VAL_19:.*]], %[[VAL_20:.*]] = calyx.register @load_1_reg : i32, i1, i1, i1, i32, i1
// CHECK:           %[[VAL_21:.*]], %[[VAL_22:.*]], %[[VAL_23:.*]], %[[VAL_24:.*]], %[[VAL_25:.*]], %[[VAL_26:.*]] = calyx.register @load_0_reg : i32, i1, i1, i1, i32, i1
// CHECK:           %[[VAL_27:.*]], %[[VAL_28:.*]] = calyx.std_pad @std_pad_0 : i6, i32
// CHECK:           %[[VAL_29:.*]], %[[VAL_30:.*]], %[[VAL_31:.*]], %[[VAL_32:.*]], %[[VAL_33:.*]], %[[VAL_34:.*]], %[[VAL_35:.*]], %[[VAL_36:.*]] = calyx.seq_mem @mem_0 <[64] x 32> [6] {external = true} : i6, i32, i1, i1, i1, i32, i1, i1
// CHECK:           %[[VAL_37:.*]], %[[VAL_38:.*]], %[[VAL_39:.*]], %[[VAL_40:.*]], %[[VAL_41:.*]], %[[VAL_42:.*]] = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK:           calyx.wires {
// CHECK:             calyx.assign %[[VAL_4]] = %[[VAL_41]] : i32
// CHECK:             calyx.group @bb0_1 {
// CHECK:               calyx.assign %[[VAL_8]] = %[[VAL_28]] : i32
// CHECK:               calyx.assign %[[VAL_29]] = %[[VAL_9]] : i6
// CHECK:               calyx.assign %[[VAL_35]] = %[[VAL_6]] : i1
// CHECK:               calyx.assign %[[VAL_21]] = %[[VAL_34]] : i32
// CHECK:               calyx.assign %[[VAL_22]] = %[[VAL_36]] : i1
// CHECK:               calyx.assign %[[VAL_27]] = %[[VAL_0]] : i6
// CHECK:               calyx.group_done %[[VAL_26]] : i1
// CHECK:             }
// CHECK:             calyx.group @bb0_2 {
// CHECK:               calyx.assign %[[VAL_10]] = %[[VAL_7]] : i32
// CHECK:               calyx.assign %[[VAL_29]] = %[[VAL_11]] : i6
// CHECK:               calyx.assign %[[VAL_35]] = %[[VAL_6]] : i1
// CHECK:               calyx.assign %[[VAL_15]] = %[[VAL_34]] : i32
// CHECK:               calyx.assign %[[VAL_16]] = %[[VAL_36]] : i1
// CHECK:               calyx.group_done %[[VAL_20]] : i1
// CHECK:             }
// CHECK:             calyx.group @ret_assign_0 {
// CHECK:               calyx.assign %[[VAL_37]] = %[[VAL_14]] : i32
// CHECK:               calyx.assign %[[VAL_38]] = %[[VAL_6]] : i1
// CHECK:               calyx.assign %[[VAL_12]] = %[[VAL_25]] : i32
// CHECK:               calyx.assign %[[VAL_13]] = %[[VAL_19]] : i32
// CHECK:               calyx.group_done %[[VAL_42]] : i1
// CHECK:             }
// CHECK:           }
// CHECK:           calyx.control {
// CHECK:             calyx.seq {
// CHECK:               calyx.enable @bb0_1
// CHECK:               calyx.enable @bb0_2
// CHECK:               calyx.enable @ret_assign_0
// CHECK:             }
// CHECK:           }
// CHECK:         } {toplevel}
module {
  func.func @main(%arg0 : i6) -> i32 {
    %0 = memref.alloc() : memref<64xi32>
    %c1 = arith.constant 1 : index
    %arg0_idx =  arith.index_cast %arg0 : i6 to index
    %1 = memref.load %0[%arg0_idx] : memref<64xi32>
    %2 = memref.load %0[%c1] : memref<64xi32>
    %3 = arith.addi %1, %2 : i32
    return %3 : i32
  }
}

// -----

// Test index types as inputs.

// CHECK-LABEL:   calyx.component @main(
// CHECK-SAME:                          %[[VAL_0:in0]]: i32,
// CHECK-SAME:                          %[[VAL_1:.*]]: i1 {clk},
// CHECK-SAME:                          %[[VAL_2:.*]]: i1 {reset},
// CHECK-SAME:                          %[[VAL_3:.*]]: i1 {go}) -> (
// CHECK-SAME:                          %[[VAL_4:.*]]: i32,
// CHECK-SAME:                          %[[VAL_5:.*]]: i1 {done}) {
// CHECK:           %[[VAL_6:.*]] = hw.constant true
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = calyx.std_slice @std_slice_0 : i32, i6
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]], %[[VAL_11:.*]], %[[VAL_12:.*]], %[[VAL_13:.*]], %[[VAL_14:.*]], %[[VAL_15:.*]], %[[VAL_16:.*]] = calyx.seq_mem @mem_0 <[64] x 32> [6] {external = true} : i6, i32, i1, i1, i1, i32, i1, i1
// CHECK:           %[[VAL_17:.*]], %[[VAL_18:.*]], %[[VAL_19:.*]], %[[VAL_20:.*]], %[[VAL_21:.*]], %[[VAL_22:.*]] = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK:           calyx.wires {
// CHECK:             calyx.assign %[[VAL_4]] = %[[VAL_21]] : i32
// CHECK:             calyx.group @bb0_0 {
// CHECK:               calyx.assign %[[VAL_7]] = %[[VAL_0]] : i32
// CHECK:               calyx.assign %[[VAL_9]] = %[[VAL_8]] : i6
// CHECK:               calyx.assign %[[VAL_15]] = %[[VAL_6]] : i1
// CHECK:               calyx.group_done %[[VAL_16]] : i1
// CHECK:             }
// CHECK:             calyx.group @ret_assign_0 {
// CHECK:               calyx.assign %[[VAL_17]] = %[[VAL_14]] : i32
// CHECK:               calyx.assign %[[VAL_18]] = %[[VAL_6]] : i1
// CHECK:               calyx.group_done %[[VAL_22]] : i1
// CHECK:             }
// CHECK:           }
// CHECK:           calyx.control {
// CHECK:             calyx.seq {
// CHECK:               calyx.enable @bb0_0
// CHECK:               calyx.enable @ret_assign_0
// CHECK:             }
// CHECK:           }
// CHECK:         } {toplevel}
module {
  func.func @main(%i : index) -> i32 {
    %0 = memref.alloc() : memref<64xi32>
    %1 = memref.load %0[%i] : memref<64xi32>
    return %1 : i32
  }
}

// -----

// Test index types as outputs.

// CHECK-LABEL:   calyx.component @main(
// CHECK-SAME:                          %[[VAL_0:.*]]: i8,
// CHECK-SAME:                          %[[VAL_1:.*]]: i1 {clk},
// CHECK-SAME:                          %[[VAL_2:.*]]: i1 {reset},
// CHECK-SAME:                          %[[VAL_3:.*]]: i1 {go}) -> (
// CHECK-SAME:                          %[[VAL_4:.*]]: i32,
// CHECK-SAME:                          %[[VAL_5:.*]]: i1 {done}) {
// CHECK:           %[[VAL_6:.*]] = hw.constant true
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = calyx.std_pad @std_pad_0 : i8, i32
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]], %[[VAL_11:.*]], %[[VAL_12:.*]], %[[VAL_13:.*]], %[[VAL_14:.*]] = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK:           calyx.wires {
// CHECK:             calyx.assign %[[VAL_4]] = %[[VAL_13]] : i32
// CHECK:             calyx.group @ret_assign_0 {
// CHECK:               calyx.assign %[[VAL_9]] = %[[VAL_8]] : i32
// CHECK:               calyx.assign %[[VAL_10]] = %[[VAL_6]] : i1
// CHECK:               calyx.assign %[[VAL_7]] = %[[VAL_0]] : i8
// CHECK:               calyx.group_done %[[VAL_14]] : i1
// CHECK:             }
// CHECK:           }
// CHECK:           calyx.control {
// CHECK:             calyx.seq {
// CHECK:               calyx.enable @ret_assign_0
// CHECK:             }
// CHECK:           }
// CHECK:         } {toplevel}
module {
  func.func @main(%i : i8) -> index {
    %0 = arith.index_cast %i : i8 to index
    return %0 : index
  }
}

// -----

// External memory store.

// CHECK-LABEL:   calyx.component @main(
// CHECK-SAME:                          %[[VAL_0:in0]]: i32,
// CHECK-SAME:                          %[[VAL_1:.*]]: i32 {mem = {id = 0 : i32, tag = "read_data"}},
// CHECK-SAME:                          %[[VAL_2:.*]]: i1 {mem = {id = 0 : i32, tag = "done"}},
// CHECK-SAME:                          %[[VAL_3:in2]]: i32,
// CHECK-SAME:                          %[[VAL_4:.*]]: i1 {clk},
// CHECK-SAME:                          %[[VAL_5:.*]]: i1 {reset},
// CHECK-SAME:                          %[[VAL_6:.*]]: i1 {go}) -> (
// CHECK-SAME:                          %[[VAL_7:.*]]: i32 {mem = {id = 0 : i32, tag = "write_data"}},
// CHECK-SAME:                          %[[VAL_8:.*]]: i3 {mem = {addr_idx = 0 : i32, id = 0 : i32, tag = "addr"}},
// CHECK-SAME:                          %[[VAL_9:.*]]: i1 {mem = {id = 0 : i32, tag = "write_en"}},
// CHECK-SAME:                          %[[VAL_10:.*]]: i1 {done}) {
// CHECK:           %[[VAL_11:.*]] = hw.constant true
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = calyx.std_slice @std_slice_0 : i32, i3
// CHECK:           calyx.wires {
// CHECK:             calyx.group @bb0_0 {
// CHECK:               calyx.assign %[[VAL_12]] = %[[VAL_3]] : i32
// CHECK:               calyx.assign %[[VAL_8]] = %[[VAL_13]] : i3
// CHECK:               calyx.assign %[[VAL_7]] = %[[VAL_0]] : i32
// CHECK:               calyx.assign %[[VAL_9]] = %[[VAL_11]] : i1
// CHECK:               calyx.group_done %[[VAL_2]] : i1
// CHECK:             }
// CHECK:           }
// CHECK:           calyx.control {
// CHECK:             calyx.seq {
// CHECK:               calyx.enable @bb0_0
// CHECK:             }
// CHECK:           }
// CHECK:         } {toplevel}
module {
  func.func @main(%arg0 : i32, %mem0 : memref<8xi32>, %i : index) {
    memref.store %arg0, %mem0[%i] : memref<8xi32>
    return
  }
}

// -----

// External memory load.

// CHECK-LABEL:   calyx.component @main(
// CHECK-SAME:                          %[[VAL_0:in0]]: i32,
// CHECK-SAME:                          %[[VAL_1:.*]]: i32 {mem = {id = 0 : i32, tag = "read_data"}},
// CHECK-SAME:                          %[[VAL_2:.*]]: i1 {mem = {id = 0 : i32, tag = "done"}},
// CHECK-SAME:                          %[[VAL_3:.*]]: i1 {clk},
// CHECK-SAME:                          %[[VAL_4:.*]]: i1 {reset},
// CHECK-SAME:                          %[[VAL_5:.*]]: i1 {go}) -> (
// CHECK-SAME:                          %[[VAL_6:.*]]: i32 {mem = {id = 0 : i32, tag = "write_data"}},
// CHECK-SAME:                          %[[VAL_7:.*]]: i3 {mem = {addr_idx = 0 : i32, id = 0 : i32, tag = "addr"}},
// CHECK-SAME:                          %[[VAL_8:.*]]: i1 {mem = {id = 0 : i32, tag = "write_en"}},
// CHECK-SAME:                          %[[VAL_9:.*]]: i32,
// CHECK-SAME:                          %[[VAL_10:.*]]: i1 {done}) {
// CHECK:           %[[VAL_11:.*]] = hw.constant true
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = calyx.std_slice @std_slice_0 : i32, i3
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]], %[[VAL_16:.*]], %[[VAL_17:.*]], %[[VAL_18:.*]], %[[VAL_19:.*]] = calyx.register @load_0_reg : i32, i1, i1, i1, i32, i1
// CHECK:           %[[VAL_20:.*]], %[[VAL_21:.*]], %[[VAL_22:.*]], %[[VAL_23:.*]], %[[VAL_24:.*]], %[[VAL_25:.*]] = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK:           calyx.wires {
// CHECK:             calyx.assign %[[VAL_9]] = %[[VAL_24]] : i32
// CHECK:             calyx.group @bb0_0 {
// CHECK:               calyx.assign %[[VAL_12]] = %[[VAL_0]] : i32
// CHECK:               calyx.assign %[[VAL_7]] = %[[VAL_13]] : i3
// CHECK:               calyx.assign %[[VAL_14]] = %[[VAL_1]] : i32
// CHECK:               calyx.assign %[[VAL_15]] = %[[VAL_11]] : i1
// CHECK:               calyx.group_done %[[VAL_19]] : i1
// CHECK:             }
// CHECK:             calyx.group @ret_assign_0 {
// CHECK:               calyx.assign %[[VAL_20]] = %[[VAL_18]] : i32
// CHECK:               calyx.assign %[[VAL_21]] = %[[VAL_11]] : i1
// CHECK:               calyx.group_done %[[VAL_25]] : i1
// CHECK:             }
// CHECK:           }
// CHECK:           calyx.control {
// CHECK:             calyx.seq {
// CHECK:               calyx.enable @bb0_0
// CHECK:               calyx.enable @ret_assign_0
// CHECK:             }
// CHECK:           }
// CHECK:         } {toplevel}
module {
  func.func @main(%i : index, %mem0 : memref<8xi32>) -> i32 {
    %0 = memref.load %mem0[%i] : memref<8xi32>
    return %0 : i32
  }
}

// -----

// External memory hazard.

// CHECK-LABEL:   calyx.component @main(
// CHECK-SAME:                          %[[VAL_0:in0]]: i32,
// CHECK-SAME:                          %[[VAL_1:in1]]: i32,
// CHECK-SAME:                          %[[VAL_2:.*]]: i32 {mem = {id = 0 : i32, tag = "read_data"}},
// CHECK-SAME:                          %[[VAL_3:.*]]: i1 {mem = {id = 0 : i32, tag = "done"}},
// CHECK-SAME:                          %[[VAL_4:.*]]: i1 {clk},
// CHECK-SAME:                          %[[VAL_5:.*]]: i1 {reset},
// CHECK-SAME:                          %[[VAL_6:.*]]: i1 {go}) -> (
// CHECK-SAME:                          %[[VAL_7:.*]]: i32 {mem = {id = 0 : i32, tag = "write_data"}},
// CHECK-SAME:                          %[[VAL_8:.*]]: i3 {mem = {addr_idx = 0 : i32, id = 0 : i32, tag = "addr"}},
// CHECK-SAME:                          %[[VAL_9:.*]]: i1 {mem = {id = 0 : i32, tag = "write_en"}},
// CHECK-SAME:                          %[[VAL_10:out0]]: i32,
// CHECK-SAME:                          %[[VAL_11:out1]]: i32,
// CHECK-SAME:                          %[[VAL_12:.*]]: i1 {done}) {
// CHECK:           %[[VAL_13:.*]] = hw.constant true
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = calyx.std_slice @std_slice_1 : i32, i3
// CHECK:           %[[VAL_16:.*]], %[[VAL_17:.*]] = calyx.std_slice @std_slice_0 : i32, i3
// CHECK:           %[[VAL_18:.*]], %[[VAL_19:.*]], %[[VAL_20:.*]], %[[VAL_21:.*]], %[[VAL_22:.*]], %[[VAL_23:.*]] = calyx.register @load_1_reg : i32, i1, i1, i1, i32, i1
// CHECK:           %[[VAL_24:.*]], %[[VAL_25:.*]], %[[VAL_26:.*]], %[[VAL_27:.*]], %[[VAL_28:.*]], %[[VAL_29:.*]] = calyx.register @load_0_reg : i32, i1, i1, i1, i32, i1
// CHECK:           %[[VAL_30:.*]], %[[VAL_31:.*]], %[[VAL_32:.*]], %[[VAL_33:.*]], %[[VAL_34:.*]], %[[VAL_35:.*]] = calyx.register @ret_arg1_reg : i32, i1, i1, i1, i32, i1
// CHECK:           %[[VAL_36:.*]], %[[VAL_37:.*]], %[[VAL_38:.*]], %[[VAL_39:.*]], %[[VAL_40:.*]], %[[VAL_41:.*]] = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK:           calyx.wires {
// CHECK:             calyx.assign %[[VAL_11]] = %[[VAL_34]] : i32
// CHECK:             calyx.assign %[[VAL_10]] = %[[VAL_40]] : i32
// CHECK:             calyx.group @bb0_0 {
// CHECK:               calyx.assign %[[VAL_14]] = %[[VAL_0]] : i32
// CHECK:               calyx.assign %[[VAL_8]] = %[[VAL_15]] : i3
// CHECK:               calyx.assign %[[VAL_24]] = %[[VAL_2]] : i32
// CHECK:               calyx.assign %[[VAL_25]] = %[[VAL_13]] : i1
// CHECK:               calyx.group_done %[[VAL_29]] : i1
// CHECK:             }
// CHECK:             calyx.group @bb0_1 {
// CHECK:               calyx.assign %[[VAL_16]] = %[[VAL_1]] : i32
// CHECK:               calyx.assign %[[VAL_8]] = %[[VAL_17]] : i3
// CHECK:               calyx.assign %[[VAL_18]] = %[[VAL_2]] : i32
// CHECK:               calyx.assign %[[VAL_19]] = %[[VAL_13]] : i1
// CHECK:               calyx.group_done %[[VAL_23]] : i1
// CHECK:             }
// CHECK:             calyx.group @ret_assign_0 {
// CHECK:               calyx.assign %[[VAL_36]] = %[[VAL_28]] : i32
// CHECK:               calyx.assign %[[VAL_37]] = %[[VAL_13]] : i1
// CHECK:               calyx.assign %[[VAL_30]] = %[[VAL_22]] : i32
// CHECK:               calyx.assign %[[VAL_31]] = %[[VAL_13]] : i1
// CHECK:               %[[VAL_42:.*]] = comb.and %[[VAL_35]], %[[VAL_41]] : i1
// CHECK:               calyx.group_done %[[VAL_42]] ? %[[VAL_13]] : i1
// CHECK:             }
// CHECK:           }
// CHECK:           calyx.control {
// CHECK:             calyx.seq {
// CHECK:               calyx.enable @bb0_0
// CHECK:               calyx.enable @bb0_1
// CHECK:               calyx.enable @ret_assign_0
// CHECK:             }
// CHECK:           }
// CHECK:         } {toplevel}
module {
  func.func @main(%i0 : index, %i1 : index, %mem0 : memref<8xi32>) -> (i32, i32) {
    %0 = memref.load %mem0[%i0] : memref<8xi32>
    %1 = memref.load %mem0[%i1] : memref<8xi32>
    return %0, %1 : i32, i32
  }
}

// -----

// Load followed by store to the same memory should be placed in separate groups.

// CHECK-LABEL:   calyx.component @main(
// CHECK-SAME:                          %[[VAL_0:in0]]: i32,
// CHECK-SAME:                          %[[VAL_1:.*]]: i1 {clk},
// CHECK-SAME:                          %[[VAL_2:.*]]: i1 {reset},
// CHECK-SAME:                          %[[VAL_3:.*]]: i1 {go}) -> (
// CHECK-SAME:                          %[[VAL_4:.*]]: i32,
// CHECK-SAME:                          %[[VAL_5:.*]]: i1 {done}) {
// CHECK:           %[[VAL_6:.*]] = hw.constant true
// CHECK:           %[[VAL_7:.*]] = hw.constant 1 : i32
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = calyx.std_slice @std_slice_1 : i32, i1
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = calyx.std_slice @std_slice_0 : i32, i1
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]], %[[VAL_14:.*]], %[[VAL_15:.*]], %[[VAL_16:.*]], %[[VAL_17:.*]] = calyx.register @load_0_reg : i32, i1, i1, i1, i32, i1
// CHECK:           %[[VAL_18:.*]], %[[VAL_19:.*]], %[[VAL_20:.*]], %[[VAL_21:.*]], %[[VAL_22:.*]], %[[VAL_23:.*]], %[[VAL_24:.*]], %[[VAL_25:.*]] = calyx.seq_mem @mem_0 <[1] x 32> [1] {external = true} : i1, i32, i1, i1, i1, i32, i1, i1
// CHECK:           %[[VAL_26:.*]], %[[VAL_27:.*]], %[[VAL_28:.*]], %[[VAL_29:.*]], %[[VAL_30:.*]], %[[VAL_31:.*]] = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK:           calyx.wires {
// CHECK:             calyx.assign %[[VAL_4]] = %[[VAL_30]] : i32
// CHECK:             calyx.group @bb0_0 {
// CHECK:               calyx.assign %[[VAL_8]] = %[[VAL_0]] : i32
// CHECK:               calyx.assign %[[VAL_18]] = %[[VAL_9]] : i1
// CHECK:               calyx.assign %[[VAL_24]] = %[[VAL_6]] : i1
// CHECK:               calyx.assign %[[VAL_12]] = %[[VAL_23]] : i32
// CHECK:               calyx.assign %[[VAL_13]] = %[[VAL_25]] : i1
// CHECK:               calyx.group_done %[[VAL_17]] : i1
// CHECK:             }
// CHECK:             calyx.group @bb0_1 {
// CHECK:               calyx.assign %[[VAL_10]] = %[[VAL_0]] : i32
// CHECK:               calyx.assign %[[VAL_18]] = %[[VAL_11]] : i1
// CHECK:               calyx.assign %[[VAL_19]] = %[[VAL_7]] : i32
// CHECK:               calyx.assign %[[VAL_20]] = %[[VAL_6]] : i1
// CHECK:               calyx.group_done %[[VAL_21]] : i1
// CHECK:             }
// CHECK:             calyx.group @ret_assign_0 {
// CHECK:               calyx.assign %[[VAL_26]] = %[[VAL_16]] : i32
// CHECK:               calyx.assign %[[VAL_27]] = %[[VAL_6]] : i1
// CHECK:               calyx.group_done %[[VAL_31]] : i1
// CHECK:             }
// CHECK:           }
// CHECK:           calyx.control {
// CHECK:             calyx.seq {
// CHECK:               calyx.enable @bb0_0
// CHECK:               calyx.enable @bb0_1
// CHECK:               calyx.enable @ret_assign_0
// CHECK:             }
// CHECK:           }
// CHECK:         } {toplevel}
module {
  func.func @main(%i : index) -> i32 {
    %c1_32 = arith.constant 1 : i32
    %0 = memref.alloc() : memref<1xi32>
    %1 = memref.load %0[%i] : memref<1xi32>
    memref.store %c1_32, %0[%i] : memref<1xi32>
    return %1 : i32
  }
}

// -----

// Load from memory with more elements than index width (32 bits).

// CHECK: calyx.std_slice {{.*}} i32, i6
module {
  func.func @main(%mem : memref<33xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %0 = memref.load %mem[%c0] : memref<33xi32>
    return %0 : i32
  }
}

// -----

// Check nonzero-width memref address ports for memrefs with some dimension = 1
// See: https://github.com/llvm/circt/issues/2660 and https://github.com/llvm/circt/pull/2661

// CHECK-DAG:       %std_slice_3.in, %std_slice_3.out = calyx.std_slice @std_slice_3 : i32, i1
// CHECK-DAG:       %std_slice_2.in, %std_slice_2.out = calyx.std_slice @std_slice_2 : i32, i1
// CHECK-DAG:           calyx.assign %mem_0.addr0 = %std_slice_3.out : i1
// CHECK-DAG:           calyx.assign %mem_0.addr1 = %std_slice_2.out : i1
module {
  func.func @main() {
    %c1_32 = arith.constant 1 : i32
    %i = arith.constant 0 : index
    %0 = memref.alloc() : memref<1x1x1x1xi32>
    memref.store %c1_32, %0[%i, %i, %i, %i] : memref<1x1x1x1xi32>
    return
  }
}

// -----

// Convert memrefs w/o shape (e.g., memref<i32>) to 1 dimensional Calyx memories 
// of size 1

// CHECK-DAG: %mem_0.addr0, %mem_0.write_data, %mem_0.write_en, %mem_0.write_done, %mem_0.clk, %mem_0.read_data, %mem_0.read_en, %mem_0.read_done = calyx.seq_mem @mem_0 <[1] x 32> [1] {external = true} : i1, i32, i1, i1, i1, i32, i1, i1
//CHECK-NEXT: calyx.wires {
//CHECK-NEXT:   calyx.group @bb0_0 {
//CHECK-NEXT:     calyx.assign %mem_0.addr0 = %false : i1
//CHECK-NEXT:     calyx.assign %mem_0.write_data = %c1_i32 : i32
//CHECK-NEXT:     calyx.assign %mem_0.write_en = %true : i1
//CHECK-NEXT:     calyx.group_done %mem_0.write_done : i1
//CHECK-NEXT:   }
//CHECK-NEXT: }
module {
  func.func @main() {
    %c1_i32 = arith.constant 1 : i32
    %alloca = memref.alloca() : memref<i32>
    memref.store %c1_i32, %alloca[] : memref<i32>
    return
  }
}

