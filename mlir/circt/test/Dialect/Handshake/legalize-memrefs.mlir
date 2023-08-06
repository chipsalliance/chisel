// RUN: circt-opt --handshake-legalize-memrefs %s | FileCheck %s

// CHECK-LABEL:   func.func @dealloc_copy(
// CHECK-SAME:                            %[[VAL_0:.*]]: memref<4xi32>) -> memref<4xi32> {
// CHECK:           %[[VAL_1:.*]] = memref.alloc() : memref<4xi32>
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[VAL_5:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_4]] {
// CHECK:             %[[VAL_6:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_5]]] : memref<4xi32>
// CHECK:             memref.store %[[VAL_6]], %[[VAL_1]]{{\[}}%[[VAL_5]]] : memref<4xi32>
// CHECK:           }
// CHECK:           return %[[VAL_1]] : memref<4xi32>
// CHECK:         }

func.func @dealloc_copy(%arg : memref<4xi32>) -> memref<4xi32> {
  %0 = memref.alloc() : memref<4xi32>
  memref.copy %arg, %0 : memref<4xi32> to memref<4xi32>
  memref.dealloc %0 : memref<4xi32>
  return %0 : memref<4xi32>
}
