// RUN: circt-opt -lower-std-to-handshake %s --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func @remove_unused_mem(
// CHECK-SAME:                         %[[VAL_0:.*]]: none, ...) -> none
// CHECK:           %[[VAL_0x:.*]] = merge %[[VAL_0]] : none
// CHECK:           return %[[VAL_0x]] : none
// CHECK:         }
func.func @remove_unused_mem() {
  %0 = memref.alloc() : memref<100xf32>
  return
}

// -----

// CHECK-LABEL:   handshake.func @load_store(
// CHECK-SAME:                               %[[VAL_0:.*]]: index,
// CHECK-SAME:                               %[[VAL_1:.*]]: none, ...) -> none
// CHECK:           %[[VAL_2:.*]]:3 = memory[ld = 1, st = 1] (%[[VAL_3:.*]], %[[VAL_4:.*]], %[[VAL_5:.*]]) {id = 0 : i32, lsq = false} : memref<4xi32>, (i32, index, index) -> (i32, none, none)
// CHECK:           %[[VAL_6:.*]] = merge %[[VAL_0]] : index
// CHECK:           %[[VAL_1x:.*]] = merge %[[VAL_1]] : none
// CHECK:           %[[VAL_7:.*]] = join %[[VAL_1x]], %[[VAL_2]]#1, %[[VAL_2]]#2 : none, none, none
// CHECK:           %[[VAL_8:.*]] = constant %[[VAL_1x]] {value = 11 : i32} : i32
// CHECK:           %[[VAL_3]], %[[VAL_4]] = store {{\[}}%[[VAL_6]]] %[[VAL_8]], %[[VAL_1x]] : index, i32
// CHECK:           %[[VAL_9:.*]] = join %[[VAL_1x]], %[[VAL_2]]#1 : none, none
// CHECK:           %[[VAL_10:.*]], %[[VAL_5]] = load {{\[}}%[[VAL_6]]] %[[VAL_2]]#0, %[[VAL_9]] : index, i32
// CHECK:           return %[[VAL_7]] : none
// CHECK:         }
func.func @load_store(%1 : index) {
  %0 = memref.alloc() : memref<4xi32>
  %c1 = arith.constant 11 : i32
  memref.store %c1, %0[%1] : memref<4xi32>
  %3 = memref.load %0[%1] : memref<4xi32>
  return
}

// -----

// CHECK-LABEL:   handshake.func @store_mul_blocks(
// CHECK-SAME:                                      %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                      %[[VAL_1:.*]]: index,
// CHECK-SAME:                                      %[[VAL_2:.*]]: none, ...) -> none 
// CHECK:           %[[VAL_3:.*]]:2 = memory[ld = 0, st = 2] (%[[VAL_4:.*]], %[[VAL_5:.*]], %[[VAL_6:.*]], %[[VAL_7:.*]]) {id = 0 : i32, lsq = false} : memref<4xi32>, (i32, index, i32, index) -> (none, none)
// CHECK:           %[[VAL_8:.*]] = merge %[[VAL_0]] : i1
// CHECK:           %[[VAL_9:.*]] = buffer [2] fifo %[[VAL_8]] : i1
// CHECK:           %[[VAL_10:.*]] = merge %[[VAL_1]] : index
// CHECK:           %[[VAL_11:.*]] = merge %[[VAL_2]] : none
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = cond_br %[[VAL_8]], %[[VAL_10]] : index
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = cond_br %[[VAL_8]], %[[VAL_11]] : none
// CHECK:           %[[VAL_16:.*]] = merge %[[VAL_12]] : index
// CHECK:           %[[VAL_17:.*]], %[[VAL_18:.*]] = control_merge %[[VAL_14]] : none, index
// CHECK:           %[[VAL_19:.*]] = join %[[VAL_17]], %[[VAL_3]]#0 : none, none
// CHECK:           %[[VAL_20:.*]] = constant %[[VAL_17]] {value = 1 : i32} : i32
// CHECK:           %[[VAL_4]], %[[VAL_5]] = store {{\[}}%[[VAL_16]]] %[[VAL_20]], %[[VAL_17]] : index, i32
// CHECK:           %[[VAL_21:.*]] = br %[[VAL_19]] : none
// CHECK:           %[[VAL_22:.*]] = merge %[[VAL_13]] : index
// CHECK:           %[[VAL_23:.*]], %[[VAL_24:.*]] = control_merge %[[VAL_15]] : none, index
// CHECK:           %[[VAL_25:.*]] = join %[[VAL_23]], %[[VAL_3]]#1 : none, none
// CHECK:           %[[VAL_26:.*]] = constant %[[VAL_23]] {value = 2 : i32} : i32
// CHECK:           %[[VAL_6]], %[[VAL_7]] = store {{\[}}%[[VAL_22]]] %[[VAL_26]], %[[VAL_23]] : index, i32
// CHECK:           %[[VAL_27:.*]] = br %[[VAL_25]] : none
// CHECK:           %[[VAL_28:.*]] = mux %[[VAL_9]] {{\[}}%[[VAL_27]], %[[VAL_21]]] : i1, none
// CHECK:           %[[VAL_29:.*]] = arith.index_cast %[[VAL_9]] : i1 to index
// CHECK:           return %[[VAL_28]] : none
// CHECK:         }
func.func @store_mul_blocks(%arg0 : i1, %arg1 : index) {
  %0 = memref.alloc() : memref<4xi32>
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  %c1 = arith.constant 1 : i32
  memref.store %c1, %0[%arg1] : memref<4xi32>
  cf.br ^bb3
^bb2:
  %c2 = arith.constant 2 : i32
  memref.store %c2, %0[%arg1] : memref<4xi32>
  cf.br ^bb3
^bb3:
  return
}

// -----

// CHECK-LABEL:   handshake.func @forward_load_to_bb(
// CHECK-SAME:                                       %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                       %[[VAL_1:.*]]: memref<32xi32>,
// CHECK-SAME:                                       %[[VAL_2:.*]]: index,
// CHECK-SAME:                                       %[[VAL_3:.*]]: none, ...) -> none
// CHECK:           %[[VAL_4:.*]]:2 = extmemory[ld = 1, st = 0] (%[[VAL_1]] : memref<32xi32>) (%[[VAL_5:.*]]) {id = 0 : i32} : (index) -> (i32, none)
// CHECK:           %[[VAL_6:.*]] = merge %[[VAL_0]] : i1
// CHECK:           %[[VAL_7:.*]] = buffer [2] fifo %[[VAL_6]] : i1
// CHECK:           %[[VAL_8:.*]] = merge %[[VAL_2]] : index
// CHECK:           %[[VAL_9:.*]] = merge %[[VAL_3]] : none
// CHECK:           %[[VAL_10:.*]] = join %[[VAL_9]], %[[VAL_4]]#1 : none, none
// CHECK:           %[[VAL_11:.*]], %[[VAL_5]] = load {{\[}}%[[VAL_8]]] %[[VAL_4]]#0, %[[VAL_9]] : index, i32
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = cond_br %[[VAL_6]], %[[VAL_10]] : none
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = cond_br %[[VAL_6]], %[[VAL_11]] : i32
// CHECK:           %[[VAL_16:.*]] = merge %[[VAL_14]] : i32
// CHECK:           %[[VAL_17:.*]], %[[VAL_18:.*]] = control_merge %[[VAL_12]] : none, index
// CHECK:           %[[VAL_19:.*]] = constant %[[VAL_17]] {value = 1 : i32} : i32
// CHECK:           %[[VAL_20:.*]] = arith.addi %[[VAL_16]], %[[VAL_19]] : i32
// CHECK:           %[[VAL_21:.*]] = br %[[VAL_17]] : none
// CHECK:           %[[VAL_22:.*]] = mux %[[VAL_7]] {{\[}}%[[VAL_13]], %[[VAL_21]]] : i1, none
// CHECK:           %[[VAL_23:.*]] = constant %[[VAL_22]] {value = true} : i1
// CHECK:           %[[VAL_24:.*]] = arith.xori %[[VAL_7]], %[[VAL_23]] : i1
// CHECK:           %[[VAL_25:.*]] = arith.index_cast %[[VAL_24]] : i1 to index
// CHECK:           return %[[VAL_22]] : none
// CHECK:         }
func.func @forward_load_to_bb(%arg0 : i1, %arg1: memref<32xi32>, %arg2 : index) {
  %0 = memref.load %arg1[%arg2] : memref<32xi32>
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  %c1 = arith.constant 1 : i32
  %1 = arith.addi %0, %c1 : i32 
  cf.br ^bb2
^bb2:
  return
}

// -----

// CHECK-LABEL:   handshake.func @dma(
// CHECK-SAME:                             %[[VAL_0:.*]]: index,
// CHECK-SAME:                             %[[VAL_1:.*]]: none, ...) -> none
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] : index
// CHECK:           %[[VAL_1x:.*]] = merge %[[VAL_1]] : none
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<10xf32>
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<10xf32>
// CHECK:           %[[VAL_5:.*]] = memref.alloc() : memref<1xi32>
// CHECK:           %[[VAL_6:.*]] = constant %[[VAL_1x]] {value = 1 : index} : index
// CHECK:           %[[VAL_7:.*]] = constant %[[VAL_1x]] {value = 1 : index} : index
// CHECK:           memref.dma_start %[[VAL_3]]{{\[}}%[[VAL_2]]], %[[VAL_4]]{{\[}}%[[VAL_2]]], %[[VAL_7]], %[[VAL_5]]{{\[}}%[[VAL_6]]] : memref<10xf32>, memref<10xf32>, memref<1xi32>
// CHECK:           memref.dma_wait %[[VAL_5]]{{\[}}%[[VAL_6]]], %[[VAL_7]] : memref<1xi32>
// CHECK:           return %[[VAL_1x]] : none
// CHECK:         }
func.func @dma(%1 : index) {
  %mem0 = memref.alloc() : memref<10xf32>
  %mem1 = memref.alloc() : memref<10xf32>
  %tag = memref.alloc() : memref<1xi32>
  %c0 = arith.constant 1 : index
  %c1 = arith.constant 1 : index
  memref.dma_start %mem0[%1], %mem1[%1], %c1, %tag[%c0] : memref<10xf32>, memref<10xf32>, memref<1xi32>
  memref.dma_wait %tag[%c0], %c1 : memref<1xi32>
  return
}
