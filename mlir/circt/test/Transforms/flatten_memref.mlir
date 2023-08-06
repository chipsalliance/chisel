// RUN: circt-opt -split-input-file --flatten-memref %s | FileCheck %s

// CHECK-LABEL:   func @as_func_arg(
// CHECK:                      %[[VAL_0:.*]]: memref<16xi32>,
// CHECK:                      %[[VAL_1:.*]]: index) -> i32 {
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_3:.*]] = arith.shli %[[VAL_1]], %[[VAL_2]] : index
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_1]], %[[VAL_3]] : index
// CHECK:           %[[VAL_5:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_4]]] : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_7:.*]] = arith.shli %[[VAL_1]], %[[VAL_6]] : index
// CHECK:           %[[VAL_8:.*]] = arith.addi %[[VAL_1]], %[[VAL_7]] : index
// CHECK:           memref.store %[[VAL_5]], %[[VAL_0]]{{\[}}%[[VAL_8]]] : memref<16xi32>
// CHECK:           return %[[VAL_5]] : i32
// CHECK:         }
func.func @as_func_arg(%a : memref<4x4xi32>, %i : index) -> i32 {
  %0 = memref.load %a[%i, %i] : memref<4x4xi32>
  memref.store %0, %a[%i, %i] : memref<4x4xi32>
  return %0 : i32
}

// -----

// CHECK-LABEL:   func @multidim3(
// CHECK:                    %[[VAL_0:.*]]: memref<210xi32>, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: index) -> i32 {
// CHECK:           %[[VAL_4:.*]] = arith.constant 5 : index
// CHECK:           %[[VAL_5:.*]] = arith.muli %[[VAL_2]], %[[VAL_4]] : index
// CHECK:           %[[VAL_6:.*]] = arith.addi %[[VAL_1]], %[[VAL_5]] : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 30 : index
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_3]], %[[VAL_7]] : index
// CHECK:           %[[VAL_9:.*]] = arith.addi %[[VAL_6]], %[[VAL_8]] : index
// CHECK:           %[[VAL_10:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_9]]] : memref<210xi32>
// CHECK:           return %[[VAL_10]] : i32
// CHECK:         }
func.func @multidim3(%a : memref<5x6x7xi32>, %i1 : index, %i2 : index, %i3 : index) -> i32 {
  %0 = memref.load %a[%i1, %i2, %i3] : memref<5x6x7xi32>
  return %0 : i32
}

// -----

// CHECK-LABEL:   func @multidim5(
// CHECK:                    %[[VAL_0:.*]]: memref<18900xi32>,
// CHECK:                    %[[VAL_1:.*]]: index) -> i32 {
// CHECK:           %[[VAL_2:.*]] = arith.constant 5 : index
// CHECK:           %[[VAL_3:.*]] = arith.muli %[[VAL_1]], %[[VAL_2]] : index
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_1]], %[[VAL_3]] : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 30 : index
// CHECK:           %[[VAL_6:.*]] = arith.muli %[[VAL_1]], %[[VAL_5]] : index
// CHECK:           %[[VAL_7:.*]] = arith.addi %[[VAL_4]], %[[VAL_6]] : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 210 : index
// CHECK:           %[[VAL_9:.*]] = arith.muli %[[VAL_1]], %[[VAL_8]] : index
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_7]], %[[VAL_9]] : index
// CHECK:           %[[VAL_11:.*]] = arith.constant 1890 : index
// CHECK:           %[[VAL_12:.*]] = arith.muli %[[VAL_1]], %[[VAL_11]] : index
// CHECK:           %[[VAL_13:.*]] = arith.addi %[[VAL_10]], %[[VAL_12]] : index
// CHECK:           %[[VAL_14:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_13]]] : memref<18900xi32>
// CHECK:           return %[[VAL_14]] : i32
// CHECK:         }
func.func @multidim5(%a : memref<5x6x7x9x10xi32>, %i : index) -> i32 {
  %0 = memref.load %a[%i, %i, %i, %i, %i] : memref<5x6x7x9x10xi32>
  return %0 : i32
}

// -----

// CHECK-LABEL:   func @multidim5_p2(
// CHECK:                       %[[VAL_0:.*]]: memref<512xi32>,
// CHECK:                       %[[VAL_1:.*]]: index) -> i32 {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.shli %[[VAL_1]], %[[VAL_2]] : index
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_1]], %[[VAL_3]] : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_6:.*]] = arith.shli %[[VAL_1]], %[[VAL_5]] : index
// CHECK:           %[[VAL_7:.*]] = arith.addi %[[VAL_4]], %[[VAL_6]] : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 6 : index
// CHECK:           %[[VAL_9:.*]] = arith.shli %[[VAL_1]], %[[VAL_8]] : index
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_7]], %[[VAL_9]] : index
// CHECK:           %[[VAL_11:.*]] = arith.constant 7 : index
// CHECK:           %[[VAL_12:.*]] = arith.shli %[[VAL_1]], %[[VAL_11]] : index
// CHECK:           %[[VAL_13:.*]] = arith.addi %[[VAL_10]], %[[VAL_12]] : index
// CHECK:           %[[VAL_14:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_13]]] : memref<512xi32>
// CHECK:           return %[[VAL_14]] : i32
// CHECK:         }
func.func @multidim5_p2(%a : memref<2x4x8x2x4xi32>, %i : index) -> i32 {
  %0 = memref.load %a[%i, %i, %i, %i, %i] : memref<2x4x8x2x4xi32>
  return %0 : i32
}

// -----

// CHECK-LABEL:   func @as_func_ret(
// CHECK:                      %[[VAL_0:.*]]: memref<16xi32>) -> memref<16xi32> {
// CHECK:           return %[[VAL_0]] : memref<16xi32>
// CHECK:         }
func.func @as_func_ret(%a : memref<4x4xi32>) -> memref<4x4xi32> {
  return %a : memref<4x4xi32>
}

// -----

// CHECK-LABEL:   func @allocs() -> memref<16xi32> {
// CHECK:           %[[VAL_0:.*]] = memref.alloc() : memref<16xi32>
// CHECK:           return %[[VAL_0]] : memref<16xi32>
// CHECK:         }
func.func @allocs() -> memref<4x4xi32> {
  %0 = memref.alloc() : memref<4x4xi32>
  return %0 : memref<4x4xi32>
}

// -----

// CHECK-LABEL:   func @across_bbs(
// CHECK:                     %[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: i1) {
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<16xi32>
// CHECK:           cf.cond_br %[[VAL_2]], ^bb1(%[[VAL_3]], %[[VAL_4]] : memref<16xi32>, memref<16xi32>), ^bb1(%[[VAL_4]], %[[VAL_3]] : memref<16xi32>, memref<16xi32>)
// CHECK:         ^bb1(%[[VAL_5:.*]]: memref<16xi32>, %[[VAL_6:.*]]: memref<16xi32>):
// CHECK:           %[[VAL_7:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_8:.*]] = arith.shli %[[VAL_1]], %[[VAL_7]] : index
// CHECK:           %[[VAL_9:.*]] = arith.addi %[[VAL_0]], %[[VAL_8]] : index
// CHECK:           %[[VAL_10:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_9]]] : memref<16xi32>
// CHECK:           %[[VAL_11:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_12:.*]] = arith.shli %[[VAL_1]], %[[VAL_11]] : index
// CHECK:           %[[VAL_13:.*]] = arith.addi %[[VAL_0]], %[[VAL_12]] : index
// CHECK:           memref.store %[[VAL_10]], %[[VAL_6]]{{\[}}%[[VAL_13]]] : memref<16xi32>
// CHECK:           return
// CHECK:         }
func.func @across_bbs(%i1 : index, %i2 : index, %c : i1) {
  %0 = memref.alloc() : memref<4x4xi32>
  %1 = memref.alloc() : memref<4x4xi32>
  cf.cond_br %c,
    ^bb1(%0, %1 : memref<4x4xi32>, memref<4x4xi32>),
    ^bb1(%1, %0 : memref<4x4xi32>, memref<4x4xi32>)
^bb1(%m1 : memref<4x4xi32>, %m2 : memref<4x4xi32>):
  %2 = memref.load %m1[%i1, %i2] : memref<4x4xi32>
  memref.store %2, %m2[%i1, %i2] : memref<4x4xi32>
  return
}

// -----

func.func @foo(%0 : memref<4x4xi32>) -> memref<4x4xi32> {
  return %0 : memref<4x4xi32>
}

// CHECK-LABEL:   func @calls() {
// CHECK:           %[[VAL_0:.*]] = memref.alloc() : memref<16xi32>
// CHECK:           %[[VAL_1:.*]] = call @foo(%[[VAL_0]]) : (memref<16xi32>) -> memref<16xi32>
// CHECK:           return
// CHECK:         }
func.func @calls() {
  %0 = memref.alloc() : memref<4x4xi32>
  %1 = call @foo(%0) : (memref<4x4xi32>) -> (memref<4x4xi32>)
  return
}

// -----

// CHECK-LABEL:   func.func @as_singleton(
// CHECK-SAME:                            %[[VAL_0:.*]]: memref<1xi32>,
// CHECK-SAME:                            %[[VAL_1:.*]]: index) -> i32 {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_2]]] : memref<1xi32>
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           memref.store %[[VAL_3]], %[[VAL_0]]{{\[}}%[[VAL_4]]] : memref<1xi32>
// CHECK:           return %[[VAL_3]] : i32
// CHECK:         }
func.func @as_singleton(%a : memref<i32>, %i : index) -> i32 {
  %0 = memref.load %a[] : memref<i32>
  memref.store %0, %a[] : memref<i32>
  return %0 : i32
}

// -----

// CHECK-LABEL:   func.func @dealloc_copy(
// CHECK-SAME:                            %[[VAL_0:.*]]: memref<16xi32>) -> memref<16xi32> {
// CHECK:           %[[VAL_1:.*]] = memref.alloc() : memref<16xi32>
// CHECK:           memref.copy %[[VAL_0]], %[[VAL_1]] : memref<16xi32> to memref<16xi32>
// CHECK:           memref.dealloc %[[VAL_1]] : memref<16xi32>
// CHECK:           return %[[VAL_1]] : memref<16xi32>
// CHECK:         }
func.func @dealloc_copy(%arg : memref<4x4xi32>) -> memref<4x4xi32> {
  %0 = memref.alloc() : memref<4x4xi32>
  memref.copy %arg, %0 : memref<4x4xi32> to memref<4x4xi32>
  memref.dealloc %0 : memref<4x4xi32>
  return %0 : memref<4x4xi32>
}
