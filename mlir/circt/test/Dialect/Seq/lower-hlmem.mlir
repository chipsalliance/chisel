// RUN: circt-opt --lower-seq-hlmem %s | FileCheck %s

// CHECK-LABEL:   hw.module @d1(
// CHECK-SAME:                  %[[VAL_0:.*]]: i1,
// CHECK-SAME:                  %[[VAL_1:.*]]: i1) {
// CHECK:           %[[VAL_2:.*]] = sv.reg  : !hw.inout<uarray<4xi32>>
// CHECK:           sv.alwaysff(posedge %[[VAL_0]]) {
// CHECK:             sv.if %[[VAL_3:.*]] {
// CHECK:               %[[VAL_4:.*]] = sv.array_index_inout %[[VAL_2]]{{\[}}%[[VAL_5:.*]]] : !hw.inout<uarray<4xi32>>, i2
// CHECK:               sv.passign %[[VAL_4]], %[[VAL_6:.*]] : i32
// CHECK:             }
// CHECK:           }(syncreset : posedge %[[VAL_1]]) {
// CHECK:           }
// CHECK:           %[[VAL_7:.*]] = hw.constant 0 : i2
// CHECK:           %[[VAL_8:.*]] = hw.constant true
// CHECK:           %[[VAL_9:.*]] = hw.constant 42 : i32
// CHECK:           %[[VAL_10:.*]] = sv.array_index_inout %[[VAL_2]]{{\[}}%[[VAL_7]]] : !hw.inout<uarray<4xi32>>, i2
// CHECK:           %[[VAL_11:.*]] = sv.read_inout %[[VAL_10]] : !hw.inout<i32>
// CHECK:           %[[VAL_12:.*]] = seq.compreg sym @myMemory_rdaddr0_dly0 %[[VAL_7]], %[[VAL_0]] : i2
// CHECK:           %[[VAL_13:.*]] = sv.array_index_inout %[[VAL_2]]{{\[}}%[[VAL_12]]] : !hw.inout<uarray<4xi32>>, i2
// CHECK:           %[[VAL_14:.*]] = sv.read_inout %[[VAL_13]] : !hw.inout<i32>
// CHECK:           %[[VAL_15:.*]] = seq.compreg sym @myMemory_rd0_reg %[[VAL_14]], %[[VAL_0]] : i32
// CHECK:           hw.output
// CHECK:         }
hw.module @d1(%clk : i1, %rst : i1) -> () {
  %myMemory = seq.hlmem @myMemory %clk, %rst : <4xi32>
    seq.write %myMemory[%c0_i2] %c42_i32 wren %c1_i1 { latency = 1 } : !seq.hlmem<4xi32>

  %c0_i2 = hw.constant 0 : i2
  %c1_i1 = hw.constant 1 : i1
  %c42_i32 = hw.constant 42 : i32

  %myMemory_rdata = seq.read %myMemory[%c0_i2] rden %c1_i1 { latency = 0} : !seq.hlmem<4xi32>
  %myMemory_rdata2 = seq.read %myMemory[%c0_i2] rden %c1_i1 { latency = 2} : !seq.hlmem<4xi32>
  hw.output
}
