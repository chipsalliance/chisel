// RUN: circt-opt --maximize-ssa %s --split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @noWorkNeeded(
// CHECK-SAME:                            %[[VAL_0:.*]]: i1) -> i32 {
// CHECK:           cf.cond_br %[[VAL_0]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_3:.*]] = arith.cmpi eq, %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:           cf.cond_br %[[VAL_3]], ^bb2, ^bb3
// CHECK:         ^bb2:
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           %[[VAL_4:.*]] = arith.constant 42 : i32
// CHECK:           return %[[VAL_4]] : i32
// CHECK:         }
func.func @noWorkNeeded(%cond: i1) -> i32 {
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %eq = arith.cmpi eq, %c1, %c2 : i32
  cf.cond_br %eq, ^bb2, ^bb3
^bb2:
  cf.br ^bb3
^bb3:
  %c42 = arith.constant 42 : i32
  return %c42 : i32
}

// -----


// CHECK-LABEL:   func.func @forwardThrough(
// CHECK-SAME:                              %[[VAL_0:.*]]: i32) -> i32 {
// CHECK:           cf.br ^bb1(%[[VAL_0]] : i32)
// CHECK:         ^bb1(%[[VAL_1:.*]]: i32):
// CHECK:           cf.br ^bb2(%[[VAL_1]] : i32)
// CHECK:         ^bb2(%[[VAL_2:.*]]: i32):
// CHECK:           cf.br ^bb3(%[[VAL_2]] : i32)
// CHECK:         ^bb3(%[[VAL_3:.*]]: i32):
// CHECK:           cf.br ^bb4(%[[VAL_3]] : i32)
// CHECK:         ^bb4(%[[VAL_4:.*]]: i32):
// CHECK:           return %[[VAL_4]] : i32
// CHECK:         }
func.func @forwardThrough(%data: i32) -> i32 {
  cf.br ^bb1
^bb1:
  cf.br ^bb2
^bb2:
  cf.br ^bb3
^bb3:
  cf.br ^bb4
^bb4:
  return %data : i32
}


// -----

// CHECK-LABEL:   func.func @forwardMultiplePaths(
// CHECK-SAME:                                    %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                    %[[VAL_1:.*]]: i32) -> i32 {
// CHECK:           cf.cond_br %[[VAL_0]], ^bb1(%[[VAL_1]] : i32), ^bb2(%[[VAL_1]] : i32)
// CHECK:         ^bb1(%[[VAL_2:.*]]: i32):
// CHECK:           cf.br ^bb4(%[[VAL_2]] : i32)
// CHECK:         ^bb2(%[[VAL_3:.*]]: i32):
// CHECK:           cf.br ^bb3(%[[VAL_3]] : i32)
// CHECK:         ^bb3(%[[VAL_4:.*]]: i32):
// CHECK:           cf.br ^bb1(%[[VAL_4]] : i32)
// CHECK:         ^bb4(%[[VAL_5:.*]]: i32):
// CHECK:           return %[[VAL_5]] : i32
// CHECK:         }
func.func @forwardMultiplePaths(%cond: i1, %data: i32) -> i32 {
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  cf.br ^bb4
^bb2:
  cf.br ^bb3
^bb3:
  cf.br ^bb1
^bb4:
  return %data : i32
}

// -----


// CHECK-LABEL:   func.func @forwardSameSuccessors(
// CHECK-SAME:                                     %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                     %[[VAL_1:.*]]: i32) -> i32 {
// CHECK:           cf.cond_br %[[VAL_0]], ^bb1(%[[VAL_0]], %[[VAL_1]] : i1, i32), ^bb1(%[[VAL_0]], %[[VAL_1]] : i1, i32)
// CHECK:         ^bb1(%[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i32):
// CHECK:           cf.cond_br %[[VAL_2]], ^bb2(%[[VAL_3]] : i32), ^bb2(%[[VAL_3]] : i32)
// CHECK:         ^bb2(%[[VAL_4:.*]]: i32):
// CHECK:           return %[[VAL_4]] : i32
// CHECK:         }
func.func @forwardSameSuccessors(%cond: i1, %data: i32) -> i32 {
  cf.cond_br %cond, ^bb1, ^bb1
^bb1:
  cf.cond_br %cond, ^bb2, ^bb2
^bb2:
  return %data : i32
}

// -----

// CHECK-LABEL:   func.func @changeMultipleUsesPerOp(
// CHECK-SAME:                                       %[[VAL_0:.*]]: i32,
// CHECK-SAME:                                       %[[VAL_1:.*]]: i32) -> i32 {
// CHECK:           cf.br ^bb1(%[[VAL_0]], %[[VAL_1]] : i32, i32)
// CHECK:         ^bb1(%[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32):
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_2]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_3]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_6:.*]] = arith.addi %[[VAL_4]], %[[VAL_5]] : i32
// CHECK:           cf.br ^bb2(%[[VAL_6]] : i32)
// CHECK:         ^bb2(%[[VAL_7:.*]]: i32):
// CHECK:           return %[[VAL_7]] : i32
// CHECK:         }
func.func @changeMultipleUsesPerOp(%data0: i32, %data1: i32) -> i32 {
  cf.br ^bb1
^bb1:
  %0 = arith.addi %data0, %data0: i32
  %1 = arith.addi %data1, %data1: i32
  %2 = arith.addi %0, %1: i32
  cf.br ^bb2(%2: i32)
^bb2(%res: i32):
  return %res : i32
}

// -----

// CHECK-LABEL:   func.func @maximizeBlockArgs(
// CHECK-SAME:                                 %[[VAL_0:.*]]: i32,
// CHECK-SAME:                                 %[[VAL_1:.*]]: i32) -> i32 {
// CHECK:           cf.br ^bb1(%[[VAL_0]], %[[VAL_1]], %[[VAL_0]], %[[VAL_1]] : i32, i32, i32, i32)
// CHECK:         ^bb1(%[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32):
// CHECK:           cf.br ^bb2(%[[VAL_4]], %[[VAL_5]], %[[VAL_2]], %[[VAL_3]] : i32, i32, i32, i32)
// CHECK:         ^bb2(%[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: i32):
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_8]], %[[VAL_9]] : i32
// CHECK:           cf.br ^bb3(%[[VAL_10]], %[[VAL_6]], %[[VAL_7]] : i32, i32, i32)
// CHECK:         ^bb3(%[[VAL_11:.*]]: i32, %[[VAL_12:.*]]: i32, %[[VAL_13:.*]]: i32):
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_12]], %[[VAL_13]] : i32
// CHECK:           cf.br ^bb4(%[[VAL_14]], %[[VAL_11]] : i32, i32)
// CHECK:         ^bb4(%[[VAL_15:.*]]: i32, %[[VAL_16:.*]]: i32):
// CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_15]] : i32
// CHECK:           return %[[VAL_16]] : i32
// CHECK:         }
func.func @maximizeBlockArgs(%n0 : i32, %n1 : i32) -> i32 {
  cf.br ^bb1(%n0, %n1 : i32, i32)
^bb1(%0: i32, %1: i32):
  cf.br ^bb2
^bb2:
  %2 = arith.addi %0, %1 : i32
  cf.br ^bb3(%2: i32)
^bb3(%3: i32):
  %4 = arith.addi %n0, %n1 : i32
  cf.br ^bb4(%4: i32)
^bb4(%5: i32):
  %res = arith.addi %3, %5 : i32
  return %3 : i32
}

// -----

// CHECK-LABEL:   func.func @maximizeOpResults(
// CHECK-SAME:                                 %[[VAL_0:.*]]: i32,
// CHECK-SAME:                                 %[[VAL_1:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_0]], %[[VAL_0]] : i32
// CHECK:           cf.br ^bb1(%[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : i32, i32, i32)
// CHECK:         ^bb1(%[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32):
// CHECK:           %[[VAL_8:.*]] = arith.addi %[[VAL_6]], %[[VAL_7]] : i32
// CHECK:           %[[VAL_9:.*]] = arith.cmpi eq, %[[VAL_8]], %[[VAL_5]] : i32
// CHECK:           cf.cond_br %[[VAL_9]], ^bb2(%[[VAL_5]], %[[VAL_6]], %[[VAL_8]] : i32, i32, i32), ^bb3(%[[VAL_5]], %[[VAL_7]], %[[VAL_8]] : i32, i32, i32)
// CHECK:         ^bb2(%[[VAL_10:.*]]: i32, %[[VAL_11:.*]]: i32, %[[VAL_12:.*]]: i32):
// CHECK:           %[[VAL_13:.*]] = arith.addi %[[VAL_11]], %[[VAL_12]] : i32
// CHECK:           cf.br ^bb4(%[[VAL_10]] : i32)
// CHECK:         ^bb3(%[[VAL_14:.*]]: i32, %[[VAL_15:.*]]: i32, %[[VAL_16:.*]]: i32):
// CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_15]], %[[VAL_16]] : i32
// CHECK:           cf.br ^bb4(%[[VAL_14]] : i32)
// CHECK:         ^bb4(%[[VAL_18:.*]]: i32):
// CHECK:           return %[[VAL_18]] : i32
// CHECK:         }
func.func @maximizeOpResults(%n0 : i32, %n1 : i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %0 = arith.addi %n0, %n1 : i32
  %1 = arith.addi %n0, %n0 : i32
  cf.br ^bb1
^bb1:
  %2 = arith.addi %0, %1 : i32
  %cond = arith.cmpi eq, %2, %c0 : i32
  cf.cond_br %cond, ^bb2, ^bb3
^bb2:
  %3 = arith.addi %0, %2 : i32
  cf.br ^bb4
^bb3:
  %4 = arith.addi %1, %2 : i32
  cf.br ^bb4
^bb4: // 2 preds: ^bb2, ^bb3
  return %c0 : i32
}

// -----

// CHECK-LABEL:   func.func @simpleLoop(
// CHECK-SAME:                          %[[VAL_0:.*]]: index,
// CHECK-SAME:                          %[[VAL_1:.*]]: memref<?xindex>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           cf.br ^bb1(%[[VAL_2]], %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : index, index, memref<?xindex>, index, index)
// CHECK:         ^bb1(%[[VAL_4:.*]]: index, %[[VAL_5:.*]]: index, %[[VAL_6:.*]]: memref<?xindex>, %[[VAL_7:.*]]: index, %[[VAL_8:.*]]: index):
// CHECK:           %[[VAL_9:.*]] = arith.cmpi eq, %[[VAL_4]], %[[VAL_5]] : index
// CHECK:           cf.cond_br %[[VAL_9]], ^bb2(%[[VAL_5]], %[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %[[VAL_4]] : index, memref<?xindex>, index, index, index), ^bb3
// CHECK:         ^bb2(%[[VAL_10:.*]]: index, %[[VAL_11:.*]]: memref<?xindex>, %[[VAL_12:.*]]: index, %[[VAL_13:.*]]: index, %[[VAL_14:.*]]: index):
// CHECK:           memref.store %[[VAL_12]], %[[VAL_11]]{{\[}}%[[VAL_14]]] : memref<?xindex>
// CHECK:           %[[VAL_15:.*]] = arith.addi %[[VAL_13]], %[[VAL_14]] : index
// CHECK:           cf.br ^bb1(%[[VAL_15]], %[[VAL_10]], %[[VAL_11]], %[[VAL_12]], %[[VAL_13]] : index, index, memref<?xindex>, index, index)
// CHECK:         ^bb3:
// CHECK:           return
// CHECK:         }
func.func @simpleLoop(%n: index, %array: memref<?xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  cf.br ^loop_entry(%c0 : index)
^loop_entry(%i: index):
  %cond = arith.cmpi eq, %i, %n : index
  cf.cond_br %cond, ^loop_body, ^end
^loop_body:
  memref.store %c0, %array[%i] : memref<?xindex>
  %new_idx = arith.addi %c1, %i : index
  cf.br ^loop_entry(%new_idx: index)
^end:
  return
}

// -----

// CHECK-LABEL:   func.func @complexControlFlow(
// CHECK-SAME:                                  %[[VAL_0:.*]]: index,
// CHECK-SAME:                                  %[[VAL_1:.*]]: memref<?xindex>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK:           cf.br ^bb1(%[[VAL_2]], %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : index, index, memref<?xindex>, index, index, index)
// CHECK:         ^bb1(%[[VAL_5:.*]]: index, %[[VAL_6:.*]]: index, %[[VAL_7:.*]]: memref<?xindex>, %[[VAL_8:.*]]: index, %[[VAL_9:.*]]: index, %[[VAL_10:.*]]: index):
// CHECK:           %[[VAL_11:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_6]] : index
// CHECK:           %[[VAL_12:.*]] = arith.addi %[[VAL_5]], %[[VAL_5]] : index
// CHECK:           cf.cond_br %[[VAL_11]], ^bb2(%[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %[[VAL_9]], %[[VAL_10]], %[[VAL_5]], %[[VAL_12]] : index, memref<?xindex>, index, index, index, index, index), ^bb6
// CHECK:         ^bb2(%[[VAL_13:.*]]: index, %[[VAL_14:.*]]: memref<?xindex>, %[[VAL_15:.*]]: index, %[[VAL_16:.*]]: index, %[[VAL_17:.*]]: index, %[[VAL_18:.*]]: index, %[[VAL_19:.*]]: index):
// CHECK:           %[[VAL_20:.*]] = arith.remui %[[VAL_18]], %[[VAL_17]] : index
// CHECK:           %[[VAL_21:.*]] = arith.cmpi eq, %[[VAL_20]], %[[VAL_16]] : index
// CHECK:           cf.cond_br %[[VAL_21]], ^bb3(%[[VAL_15]], %[[VAL_13]], %[[VAL_14]], %[[VAL_15]], %[[VAL_16]], %[[VAL_17]], %[[VAL_18]], %[[VAL_19]] : index, index, memref<?xindex>, index, index, index, index, index), ^bb5(%[[VAL_13]], %[[VAL_14]], %[[VAL_15]], %[[VAL_16]], %[[VAL_17]], %[[VAL_18]] : index, memref<?xindex>, index, index, index, index)
// CHECK:         ^bb3(%[[VAL_22:.*]]: index, %[[VAL_23:.*]]: index, %[[VAL_24:.*]]: memref<?xindex>, %[[VAL_25:.*]]: index, %[[VAL_26:.*]]: index, %[[VAL_27:.*]]: index, %[[VAL_28:.*]]: index, %[[VAL_29:.*]]: index):
// CHECK:           %[[VAL_30:.*]] = arith.cmpi eq, %[[VAL_22]], %[[VAL_29]] : index
// CHECK:           cf.cond_br %[[VAL_30]], ^bb4(%[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_26]], %[[VAL_27]], %[[VAL_28]], %[[VAL_29]], %[[VAL_22]] : index, memref<?xindex>, index, index, index, index, index, index), ^bb5(%[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_26]], %[[VAL_27]], %[[VAL_28]] : index, memref<?xindex>, index, index, index, index)
// CHECK:         ^bb4(%[[VAL_31:.*]]: index, %[[VAL_32:.*]]: memref<?xindex>, %[[VAL_33:.*]]: index, %[[VAL_34:.*]]: index, %[[VAL_35:.*]]: index, %[[VAL_36:.*]]: index, %[[VAL_37:.*]]: index, %[[VAL_38:.*]]: index):
// CHECK:           %[[VAL_39:.*]] = memref.load %[[VAL_32]]{{\[}}%[[VAL_38]]] : memref<?xindex>
// CHECK:           %[[VAL_40:.*]] = arith.addi %[[VAL_36]], %[[VAL_39]] : index
// CHECK:           %[[VAL_41:.*]] = arith.addi %[[VAL_38]], %[[VAL_40]] : index
// CHECK:           memref.store %[[VAL_41]], %[[VAL_32]]{{\[}}%[[VAL_38]]] : memref<?xindex>
// CHECK:           %[[VAL_42:.*]] = arith.addi %[[VAL_36]], %[[VAL_34]] : index
// CHECK:           cf.br ^bb3(%[[VAL_42]], %[[VAL_31]], %[[VAL_32]], %[[VAL_33]], %[[VAL_34]], %[[VAL_35]], %[[VAL_36]], %[[VAL_37]] : index, index, memref<?xindex>, index, index, index, index, index)
// CHECK:         ^bb5(%[[VAL_43:.*]]: index, %[[VAL_44:.*]]: memref<?xindex>, %[[VAL_45:.*]]: index, %[[VAL_46:.*]]: index, %[[VAL_47:.*]]: index, %[[VAL_48:.*]]: index):
// CHECK:           %[[VAL_49:.*]] = arith.addi %[[VAL_48]], %[[VAL_46]] : index
// CHECK:           cf.br ^bb1(%[[VAL_49]], %[[VAL_43]], %[[VAL_44]], %[[VAL_45]], %[[VAL_46]], %[[VAL_47]] : index, index, memref<?xindex>, index, index, index)
// CHECK:         ^bb6:
// CHECK:           return
// CHECK:         }
func.func @complexControlFlow(%outer_lim: index, %array: memref<?xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  cf.br ^outer_loop_entry(%c0: index)
^outer_loop_entry(%outer_idx: index):
  %outer_loop_cond = arith.cmpi eq, %outer_idx, %outer_lim : index
  %0 = arith.addi %outer_idx, %outer_idx : index
  cf.cond_br %outer_loop_cond, ^outer_loop_body, ^end
^outer_loop_body:
  %1 = arith.remui %outer_idx, %c2 : index
  %outer_cond = arith.cmpi eq, %1, %c1 : index
  cf.cond_br %outer_cond, ^inner_loop_entry(%c0: index), ^outer_loop_end
^inner_loop_entry(%inner_idx: index):
  %inner_loop_cond = arith.cmpi eq, %inner_idx, %0 : index
  cf.cond_br %inner_loop_cond, ^inner_loop_body, ^outer_loop_end
^inner_loop_body:
  %3 = memref.load %array[%inner_idx] : memref<?xindex>
  %4 = arith.addi %outer_idx, %3 : index
  %5 = arith.addi %inner_idx, %4 : index
  memref.store %5, %array[%inner_idx] : memref<?xindex>
  %new_inner_idx = arith.addi %outer_idx, %c1 : index
  cf.br ^inner_loop_entry(%new_inner_idx : index)
^outer_loop_end:
  %new_outer_idx = arith.addi %outer_idx, %c1 : index
  cf.br ^outer_loop_entry(%new_outer_idx: index)
^end:
  return
}
