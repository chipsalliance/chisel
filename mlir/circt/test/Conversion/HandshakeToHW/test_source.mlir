// RUN: circt-opt -lower-handshake-to-hw %s | FileCheck %s

// CHECK-LABEL:   hw.module @handshake_source_0ins_1outs_ctrl() -> (out0: !esi.channel<i0>) {
// CHECK:           %[[VAL_0:.*]], %[[VAL_1:.*]] = esi.wrap.vr %[[VAL_2:.*]], %[[VAL_3:.*]] : i0
// CHECK:           %[[VAL_3]] = hw.constant true
// CHECK:           %[[VAL_2]] = hw.constant 0 : i0
// CHECK:           hw.output %[[VAL_0]] : !esi.channel<i0>
// CHECK:         }

handshake.func @test_source() -> (none) {
  %0 = source
  return %0 : none
}
