// RUN: circt-opt -lower-handshake-to-hw %s | FileCheck %s

// CHECK-LABEL:   hw.module @hw_struct_create_in_ui4_ui32_out_struct_address_ui4_data_ui32(
// CHECK-SAME:              %[[VAL_0:.*]]: !esi.channel<i4>, %[[VAL_1:.*]]: !esi.channel<i32>) -> (out0: !esi.channel<!hw.struct<address: i4, data: i32>>) {
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_4:.*]] : i4
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = esi.wrap.vr %[[VAL_9:.*]], %[[VAL_10:.*]] : !hw.struct<address: i4, data: i32>
// CHECK:           %[[VAL_10]] = comb.and %[[VAL_3]], %[[VAL_6]] : i1
// CHECK:           %[[VAL_4]] = comb.and %[[VAL_8]], %[[VAL_10]] : i1
// CHECK:           %[[VAL_9]] = hw.struct_create (%[[VAL_2]], %[[VAL_5]]) : !hw.struct<address: i4, data: i32>
// CHECK:           hw.output %[[VAL_7]] : !esi.channel<!hw.struct<address: i4, data: i32>>
// CHECK:         }

!T = !hw.struct<address: i4, data: i32>
handshake.func @main(%address : i4, %data : i32) -> (!T) {
  %0 = hw.struct_create (%address, %data) : !T
  return %0 : !T
}
