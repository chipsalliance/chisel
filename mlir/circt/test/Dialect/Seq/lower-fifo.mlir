// This is such a large lowering that it doesn't really make that much sense to
// inspect the test output. So this is mostly here for detecting regressions.
// Canonicalize used to remove some of the constants introduced by the lowering.
// RUN: circt-opt --lower-seq-fifo --canonicalize %s | FileCheck %s --implicit-check-not=seq.fifo


// CHECK-LABEL:   hw.module @fifo1(
// CHECK-SAME:            %[[VAL_0:.*]]: i1, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_5:.*]] = hw.constant true
// CHECK:           %[[VAL_6:.*]] = hw.constant -1 : i2
// CHECK:           %[[VAL_7:.*]] = hw.constant -2 : i2
// CHECK:           %[[VAL_8:.*]] = hw.constant 1 : i2
// CHECK:           %[[VAL_9:.*]] = hw.constant 0 : i2
// CHECK:           %[[VAL_10:.*]] = seq.compreg sym @fifo_count %[[VAL_11:.*]], %[[VAL_0]], %[[VAL_1]], %[[VAL_9]]  : i2
// CHECK:           %[[VAL_12:.*]] = seq.hlmem @fifo_mem %[[VAL_0]], %[[VAL_1]] : <3xi32>
// CHECK:           %[[VAL_13:.*]] = seq.compreg sym @fifo_rd_addr %[[VAL_14:.*]], %[[VAL_0]], %[[VAL_1]], %[[VAL_9]]  : i2
// CHECK:           %[[VAL_15:.*]] = seq.compreg sym @fifo_wr_addr %[[VAL_16:.*]], %[[VAL_0]], %[[VAL_1]], %[[VAL_9]]  : i2
// CHECK:           %[[VAL_17:.*]] = seq.read %[[VAL_12]]{{\[}}%[[VAL_13]]] rden %[[VAL_3]] {latency = 0 : i64} : !seq.hlmem<3xi32>
// CHECK:           seq.write %[[VAL_12]]{{\[}}%[[VAL_15]]] %[[VAL_2]] wren %[[VAL_4]] {latency = 1 : i64} : !seq.hlmem<3xi32>
// CHECK:           %[[VAL_18:.*]] = comb.xor %[[VAL_3]], %[[VAL_5]] : i1
// CHECK:           %[[VAL_19:.*]] = comb.xor %[[VAL_4]], %[[VAL_5]] : i1
// CHECK:           %[[VAL_20:.*]] = comb.and %[[VAL_3]], %[[VAL_19]] : i1
// CHECK:           %[[VAL_21:.*]] = comb.and %[[VAL_4]], %[[VAL_18]] : i1
// CHECK:           %[[VAL_22:.*]] = comb.icmp eq %[[VAL_10]], %[[VAL_7]] : i2
// CHECK:           %[[VAL_23:.*]] = comb.add %[[VAL_10]], %[[VAL_8]] : i2
// CHECK:           %[[VAL_24:.*]] = comb.mux %[[VAL_22]], %[[VAL_10]], %[[VAL_23]] : i2
// CHECK:           %[[VAL_25:.*]] = comb.icmp eq %[[VAL_10]], %[[VAL_9]] : i2
// CHECK:           %[[VAL_26:.*]] = comb.add %[[VAL_10]], %[[VAL_6]] : i2
// CHECK:           %[[VAL_27:.*]] = comb.xor %[[VAL_20]], %[[VAL_5]] : i1
// CHECK:           %[[VAL_28:.*]] = comb.or %[[VAL_27]], %[[VAL_25]] : i1
// CHECK:           %[[VAL_29:.*]] = comb.mux %[[VAL_28]], %[[VAL_10]], %[[VAL_26]] : i2
// CHECK:           %[[VAL_30:.*]] = comb.mux %[[VAL_21]], %[[VAL_24]], %[[VAL_29]] : i2
// CHECK:           %[[VAL_31:.*]] = comb.or %[[VAL_3]], %[[VAL_4]] : i1
// CHECK:           %[[VAL_11]] = comb.mux %[[VAL_31]], %[[VAL_30]], %[[VAL_10]] {sv.namehint = "fifo_count_next"} : i2
// CHECK:           %[[VAL_32:.*]] = comb.icmp ne %[[VAL_10]], %[[VAL_7]] : i2
// CHECK:           %[[VAL_33:.*]] = comb.and %[[VAL_4]], %[[VAL_32]] : i1
// CHECK:           %[[VAL_34:.*]] = comb.add %[[VAL_15]], %[[VAL_8]] : i2
// CHECK:           %[[VAL_16]] = comb.mux %[[VAL_33]], %[[VAL_34]], %[[VAL_15]] {sv.namehint = "fifo_wr_addr_next"} : i2
// CHECK:           %[[VAL_35:.*]] = comb.icmp ne %[[VAL_10]], %[[VAL_9]] : i2
// CHECK:           %[[VAL_36:.*]] = comb.and %[[VAL_3]], %[[VAL_35]] : i1
// CHECK:           %[[VAL_37:.*]] = comb.add %[[VAL_13]], %[[VAL_8]] : i2
// CHECK:           %[[VAL_14]] = comb.mux %[[VAL_36]], %[[VAL_37]], %[[VAL_13]] {sv.namehint = "fifo_rd_addr_next"} : i2
// CHECK:           hw.output %[[VAL_17]] : i32
// CHECK:         }
hw.module @fifo1(%clk : i1, %rst : i1, %in : i32, %rdEn : i1, %wrEn : i1) -> (out: i32) {
  %out, %full, %empty = seq.fifo depth 3 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i32
  hw.output %out : i32
}


// CHECK-LABEL:   hw.module @fifo2(
// CHECK-SAME:             %[[VAL_0:.*]]: i1, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32, empty: i1, full: i1, almost_empty: i1, almost_full: i1) {
// CHECK:           %[[VAL_5:.*]] = hw.constant 2 : i3
// CHECK:           %[[VAL_6:.*]] = hw.constant 0 : i2
// CHECK:           %[[VAL_7:.*]] = hw.constant true
// CHECK:           %[[VAL_8:.*]] = hw.constant -1 : i3
// CHECK:           %[[VAL_9:.*]] = hw.constant 3 : i3
// CHECK:           %[[VAL_10:.*]] = hw.constant 1 : i3
// CHECK:           %[[VAL_11:.*]] = hw.constant 0 : i3
// CHECK:           %[[VAL_12:.*]] = hw.constant 1 : i2
// CHECK:           %[[VAL_13:.*]] = seq.compreg sym @fifo_count %[[VAL_14:.*]], %[[VAL_0]], %[[VAL_1]], %[[VAL_11]]  : i3
// CHECK:           %[[VAL_15:.*]] = seq.hlmem @fifo_mem %[[VAL_0]], %[[VAL_1]] : <4xi32>
// CHECK:           %[[VAL_16:.*]] = seq.compreg sym @fifo_rd_addr %[[VAL_17:.*]], %[[VAL_0]], %[[VAL_1]], %[[VAL_6]]  : i2
// CHECK:           %[[VAL_18:.*]] = seq.compreg sym @fifo_wr_addr %[[VAL_19:.*]], %[[VAL_0]], %[[VAL_1]], %[[VAL_6]]  : i2
// CHECK:           %[[VAL_20:.*]] = seq.read %[[VAL_15]]{{\[}}%[[VAL_16]]] rden %[[VAL_3]] {latency = 0 : i64} : !seq.hlmem<4xi32>
// CHECK:           seq.write %[[VAL_15]]{{\[}}%[[VAL_18]]] %[[VAL_2]] wren %[[VAL_4]] {latency = 1 : i64} : !seq.hlmem<4xi32>
// CHECK:           %[[VAL_21:.*]] = comb.icmp eq %[[VAL_13]], %[[VAL_9]] {sv.namehint = "fifo_full"} : i3
// CHECK:           %[[VAL_22:.*]] = comb.icmp eq %[[VAL_13]], %[[VAL_11]] {sv.namehint = "fifo_empty"} : i3
// CHECK:           %[[VAL_23:.*]] = comb.xor %[[VAL_3]], %[[VAL_7]] : i1
// CHECK:           %[[VAL_24:.*]] = comb.xor %[[VAL_4]], %[[VAL_7]] : i1
// CHECK:           %[[VAL_25:.*]] = comb.and %[[VAL_3]], %[[VAL_24]] : i1
// CHECK:           %[[VAL_26:.*]] = comb.and %[[VAL_4]], %[[VAL_23]] : i1
// CHECK:           %[[VAL_27:.*]] = comb.icmp eq %[[VAL_13]], %[[VAL_9]] : i3
// CHECK:           %[[VAL_28:.*]] = comb.add %[[VAL_13]], %[[VAL_10]] : i3
// CHECK:           %[[VAL_29:.*]] = comb.mux %[[VAL_27]], %[[VAL_13]], %[[VAL_28]] : i3
// CHECK:           %[[VAL_30:.*]] = comb.icmp eq %[[VAL_13]], %[[VAL_11]] : i3
// CHECK:           %[[VAL_31:.*]] = comb.add %[[VAL_13]], %[[VAL_8]] : i3
// CHECK:           %[[VAL_32:.*]] = comb.xor %[[VAL_25]], %[[VAL_7]] : i1
// CHECK:           %[[VAL_33:.*]] = comb.or %[[VAL_32]], %[[VAL_30]] : i1
// CHECK:           %[[VAL_34:.*]] = comb.mux %[[VAL_33]], %[[VAL_13]], %[[VAL_31]] : i3
// CHECK:           %[[VAL_35:.*]] = comb.mux %[[VAL_26]], %[[VAL_29]], %[[VAL_34]] : i3
// CHECK:           %[[VAL_36:.*]] = comb.or %[[VAL_3]], %[[VAL_4]] : i1
// CHECK:           %[[VAL_14]] = comb.mux %[[VAL_36]], %[[VAL_35]], %[[VAL_13]] {sv.namehint = "fifo_count_next"} : i3
// CHECK:           %[[VAL_37:.*]] = comb.xor %[[VAL_21]], %[[VAL_7]] : i1
// CHECK:           %[[VAL_38:.*]] = comb.and %[[VAL_4]], %[[VAL_37]] : i1
// CHECK:           %[[VAL_39:.*]] = comb.add %[[VAL_18]], %[[VAL_12]] : i2
// CHECK:           %[[VAL_19]] = comb.mux %[[VAL_38]], %[[VAL_39]], %[[VAL_18]] {sv.namehint = "fifo_wr_addr_next"} : i2
// CHECK:           %[[VAL_40:.*]] = comb.xor %[[VAL_22]], %[[VAL_7]] : i1
// CHECK:           %[[VAL_41:.*]] = comb.and %[[VAL_3]], %[[VAL_40]] : i1
// CHECK:           %[[VAL_42:.*]] = comb.add %[[VAL_16]], %[[VAL_12]] : i2
// CHECK:           %[[VAL_17]] = comb.mux %[[VAL_41]], %[[VAL_42]], %[[VAL_16]] {sv.namehint = "fifo_rd_addr_next"} : i2
// CHECK:           %[[VAL_43:.*]] = comb.extract %[[VAL_13]] from 1 : (i3) -> i2
// CHECK:           %[[VAL_44:.*]] = comb.icmp ne %[[VAL_43]], %[[VAL_6]] {sv.namehint = "fifo_almost_full"} : i2
// CHECK:           %[[VAL_45:.*]] = comb.icmp ult %[[VAL_13]], %[[VAL_5]] {sv.namehint = "fifo_almost_empty"} : i3
// CHECK:           hw.output %[[VAL_20]], %[[VAL_22]], %[[VAL_21]], %[[VAL_45]], %[[VAL_44]] : i32, i1, i1, i1, i1
// CHECK:         }
hw.module @fifo2(%clk : i1, %rst : i1, %in : i32, %rdEn : i1, %wrEn : i1) -> (out: i32, empty: i1, full: i1, almost_empty : i1, almost_full : i1) {
  %out, %full, %empty, %almostFull, %almostEmpty = seq.fifo depth 4 almost_full 2 almost_empty 1 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i32
  hw.output %out, %empty, %full, %almostEmpty, %almostFull : i32, i1, i1, i1, i1
}
