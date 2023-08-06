// RUN: circt-opt -handshake-lower-extmem-to-hw %s | FileCheck %s

// CHECK-LABEL:   handshake.func @main(
// CHECK-SAME:          %[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: none, %[[VAL_5:.*]]: none, ...) -> (none, i4, !hw.struct<address: i4, data: i32>)
// CHECK:           %[[VAL_6:.*]]:2 = fork [2] %[[VAL_3]] : i32
// CHECK:           %[[VAL_7:.*]] = join %[[VAL_6]]#1 : i32
// CHECK:           %[[VAL_8:.*]] = arith.index_cast %[[VAL_9:.*]] : index to i4
// CHECK:           %[[VAL_10:.*]] = arith.index_cast %[[VAL_11:.*]] : index to i4
// CHECK:           %[[VAL_12:.*]] = hw.struct_create (%[[VAL_10]], %[[VAL_13:.*]]) : !hw.struct<address: i4, data: i32>
// CHECK:           %[[VAL_14:.*]]:2 = fork [2] %[[VAL_5]] : none
// CHECK:           %[[VAL_15:.*]], %[[VAL_9]] = load {{\[}}%[[VAL_0]]] %[[VAL_6]]#0, %[[VAL_14]]#0 : index, i32
// CHECK:           %[[VAL_13]], %[[VAL_11]] = store {{\[}}%[[VAL_1]]] %[[VAL_2]], %[[VAL_14]]#1 : index, i32
// CHECK:           sink %[[VAL_15]] : i32
// CHECK:           %[[VAL_16:.*]] = join %[[VAL_4]], %[[VAL_7]] : none, none
// CHECK:           return %[[VAL_16]], %[[VAL_8]], %[[VAL_12]] : none, i4, !hw.struct<address: i4, data: i32>
// CHECK:         }

handshake.func @main(%arg0: index, %arg1: index, %v: i32, %mem : memref<10xi32>, %argCtrl: none) -> none {
  %ldData, %stCtrl, %ldCtrl = handshake.extmemory[ld=1, st=1](%mem : memref<10xi32>)(%storeData, %storeAddr, %loadAddr) {id = 0 : i32} : (i32, index, index) -> (i32, none, none)
  %fCtrl:2 = fork [2] %argCtrl : none
  %loadData, %loadAddr = load [%arg0] %ldData, %fCtrl#0 : index, i32
  %storeData, %storeAddr = store [%arg1] %v, %fCtrl#1 : index, i32
  sink %loadData : i32
  %finCtrl = join %stCtrl, %ldCtrl : none, none
  return %finCtrl : none
}
