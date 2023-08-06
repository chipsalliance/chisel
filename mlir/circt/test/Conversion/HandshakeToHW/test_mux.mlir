// RUN: circt-opt -lower-handshake-to-hw -split-input-file %s | FileCheck %s

// CHECK-LABEL:   hw.module @handshake_mux_in_ui64_ui64_ui64_out_ui64(
// CHECK-SAME:                                                        %[[VAL_0:.*]]: !esi.channel<i64>, %[[VAL_1:.*]]: !esi.channel<i64>, %[[VAL_2:.*]]: !esi.channel<i64>) -> (out0: !esi.channel<i64>) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_5:.*]] : i64
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_8:.*]] : i64
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = esi.unwrap.vr %[[VAL_2]], %[[VAL_11:.*]] : i64
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = esi.wrap.vr %[[VAL_14:.*]], %[[VAL_15:.*]] : i64
// CHECK:           %[[VAL_16:.*]] = comb.extract %[[VAL_3]] from 0 : (i64) -> i1
// CHECK:           %[[VAL_17:.*]] = hw.constant false
// CHECK:           %[[VAL_18:.*]] = comb.concat %[[VAL_17]], %[[VAL_16]] : i1, i1
// CHECK:           %[[VAL_19:.*]] = hw.constant 1 : i2
// CHECK:           %[[VAL_20:.*]] = comb.shl %[[VAL_19]], %[[VAL_18]] : i2
// CHECK:           %[[VAL_21:.*]] = comb.mux %[[VAL_16]], %[[VAL_10]], %[[VAL_7]] : i1
// CHECK:           %[[VAL_15]] = comb.and %[[VAL_21]], %[[VAL_4]] : i1
// CHECK:           %[[VAL_5]] = comb.and %[[VAL_15]], %[[VAL_13]] : i1
// CHECK:           %[[VAL_22:.*]] = comb.extract %[[VAL_20]] from 0 : (i2) -> i1
// CHECK:           %[[VAL_8]] = comb.and %[[VAL_22]], %[[VAL_5]] : i1
// CHECK:           %[[VAL_23:.*]] = comb.extract %[[VAL_20]] from 1 : (i2) -> i1
// CHECK:           %[[VAL_11]] = comb.and %[[VAL_23]], %[[VAL_5]] : i1
// CHECK:           %[[VAL_14]] = comb.mux %[[VAL_16]], %[[VAL_9]], %[[VAL_6]] : i64
// CHECK:           hw.output %[[VAL_12]] : !esi.channel<i64>
// CHECK:         }

handshake.func @test_mux(%arg0: index, %arg1: index, %arg2: index, %arg3: none, ...) -> (index, none) {
  %0 = mux %arg0 [%arg1, %arg2] : index, index
  return %0, %arg3 : index, none
}

// -----

// CHECK-LABEL:   hw.module @handshake_mux_in_ui64_ui64_ui64_ui64_out_ui64(
// CHECK-SAME:                                                             %[[VAL_0:.*]]: !esi.channel<i64>, %[[VAL_1:.*]]: !esi.channel<i64>, %[[VAL_2:.*]]: !esi.channel<i64>, %[[VAL_3:.*]]: !esi.channel<i64>) -> (out0: !esi.channel<i64>) {
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_6:.*]] : i64
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_9:.*]] : i64
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = esi.unwrap.vr %[[VAL_2]], %[[VAL_12:.*]] : i64
// CHECK:           %[[VAL_13:.*]], %[[VAL_14:.*]] = esi.unwrap.vr %[[VAL_3]], %[[VAL_15:.*]] : i64
// CHECK:           %[[VAL_16:.*]], %[[VAL_17:.*]] = esi.wrap.vr %[[VAL_18:.*]], %[[VAL_19:.*]] : i64
// CHECK:           %[[VAL_20:.*]] = comb.extract %[[VAL_4]] from 0 : (i64) -> i2
// CHECK:           %[[VAL_21:.*]] = hw.constant false
// CHECK:           %[[VAL_22:.*]] = comb.concat %[[VAL_21]], %[[VAL_20]] : i1, i2
// CHECK:           %[[VAL_23:.*]] = hw.constant 1 : i3
// CHECK:           %[[VAL_24:.*]] = comb.shl %[[VAL_23]], %[[VAL_22]] : i3
// CHECK:           %[[VAL_25:.*]] = hw.array_create %[[VAL_8]], %[[VAL_11]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_26:.*]] = hw.array_get %[[VAL_25]]{{\[}}%[[VAL_20]]] : !hw.array<3xi1>
// CHECK:           %[[VAL_19]] = comb.and %[[VAL_26]], %[[VAL_5]] : i1
// CHECK:           %[[VAL_6]] = comb.and %[[VAL_19]], %[[VAL_17]] : i1
// CHECK:           %[[VAL_27:.*]] = comb.extract %[[VAL_24]] from 0 : (i3) -> i1
// CHECK:           %[[VAL_9]] = comb.and %[[VAL_27]], %[[VAL_6]] : i1
// CHECK:           %[[VAL_28:.*]] = comb.extract %[[VAL_24]] from 1 : (i3) -> i1
// CHECK:           %[[VAL_12]] = comb.and %[[VAL_28]], %[[VAL_6]] : i1
// CHECK:           %[[VAL_29:.*]] = comb.extract %[[VAL_24]] from 2 : (i3) -> i1
// CHECK:           %[[VAL_15]] = comb.and %[[VAL_29]], %[[VAL_6]] : i1
// CHECK:           %[[VAL_30:.*]] = hw.array_create %[[VAL_7]], %[[VAL_10]], %[[VAL_13]] : i64
// CHECK:           %[[VAL_18]] = hw.array_get %[[VAL_30]]{{\[}}%[[VAL_20]]] : !hw.array<3xi64>
// CHECK:           hw.output %[[VAL_16]] : !esi.channel<i64>
// CHECK:         }

handshake.func @test_mux_3way(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: none, ...) -> (index, none) {
  %0 = mux %arg0 [%arg1, %arg2, %arg3] : index, index
  return %0, %arg4 : index, none
}
