// RUN: circt-opt --lower-dc-to-hw %s | FileCheck %s

// CHECK-LABEL:   hw.module @simple(
// CHECK-SAME:             %[[VAL_0:.*]]: !esi.channel<i0>, %[[VAL_1:.*]]: !esi.channel<i64>, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: !esi.channel<i1>) -> (out0: !esi.channel<i0>, out1: !esi.channel<i64>, out2: i1, out3: !esi.channel<i1>) attributes {argNames = ["", "", "", ""]} {
// CHECK:           hw.output %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : !esi.channel<i0>, !esi.channel<i64>, i1, !esi.channel<i1>
// CHECK:         }
hw.module @simple(%0 : !dc.token, %1 : !dc.value<i64>, %2 : i1, %3 : !dc.value<i1>)
        -> (out0: !dc.token, out1: !dc.value<i64>, out2: i1, out3: !dc.value<i1>) {
    hw.output %0, %1, %2, %3 : !dc.token, !dc.value<i64>, i1, !dc.value<i1>
}

// CHECK-LABEL:   hw.module @pack(
// CHECK-SAME:                    %[[VAL_0:.*]]: !esi.channel<i0>, %[[VAL_1:.*]]: i64) -> (out0: !esi.channel<i64>) {
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_4:.*]] : i0
// CHECK:           %[[VAL_5:.*]], %[[VAL_4]] = esi.wrap.vr %[[VAL_1]], %[[VAL_3]] : i64
// CHECK:           hw.output %[[VAL_5]] : !esi.channel<i64>
hw.module @pack(%token : !dc.token, %v1 : i64) -> (out0: !dc.value<i64>) {
    %out = dc.pack %token, %v1 : i64
    hw.output %out : !dc.value<i64>
}

// CHECK-LABEL:   hw.module @unpack(
// CHECK-SAME:                      %[[VAL_0:.*]]: !esi.channel<i64>) -> (out0: !esi.channel<i0>, out1: i64) {
// CHECK:           %[[VAL_1:.*]], %[[VAL_2:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_3:.*]] : i64
// CHECK:           %[[VAL_4:.*]] = hw.constant 0 : i0
// CHECK:           %[[VAL_5:.*]], %[[VAL_3]] = esi.wrap.vr %[[VAL_4]], %[[VAL_2]] : i0
// CHECK:           hw.output %[[VAL_5]], %[[VAL_1]] : !esi.channel<i0>, i64
hw.module @unpack(%v : !dc.value<i64>) -> (out0: !dc.token, out1: i64) {
    %out:2 = dc.unpack %v : !dc.value<i64>
    hw.output %out#0, %out#1 : !dc.token, i64
}

// CHECK-LABEL:   hw.module @join(
// CHECK-SAME:            %[[VAL_0:.*]]: !esi.channel<i0>, %[[VAL_1:.*]]: !esi.channel<i0>) -> (out0: !esi.channel<i0>) {
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_4:.*]] : i0
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_4]] : i0
// CHECK:           %[[VAL_7:.*]] = hw.constant 0 : i0
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = esi.wrap.vr %[[VAL_7]], %[[VAL_10:.*]] : i0
// CHECK:           %[[VAL_10]] = comb.and %[[VAL_3]], %[[VAL_6]] : i1
// CHECK:           %[[VAL_4]] = comb.and %[[VAL_9]], %[[VAL_10]] : i1
// CHECK:           hw.output %[[VAL_8]] : !esi.channel<i0>
// CHECK:         }
hw.module @join(%t1 : !dc.token, %t2 : !dc.token) -> (out0: !dc.token) {
    %out = dc.join %t1, %t2
    hw.output %out : !dc.token
}

// CHECK-LABEL:   hw.module @fork(
// CHECK-SAME:           %[[VAL_0:.*]]: !esi.channel<i0>, %[[VAL_1:.*]]: i1 {dc.clock}, %[[VAL_2:.*]]: i1 {dc.reset}) -> (out0: !esi.channel<i0>, out1: !esi.channel<i0>) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_5:.*]] : i0
// CHECK:           %[[VAL_6:.*]] = hw.constant 0 : i0
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = esi.wrap.vr %[[VAL_6]], %[[VAL_9:.*]] : i0
// CHECK:           %[[VAL_10:.*]] = hw.constant 0 : i0
// CHECK:           %[[VAL_11:.*]], %[[VAL_12:.*]] = esi.wrap.vr %[[VAL_10]], %[[VAL_13:.*]] : i0
// CHECK:           %[[VAL_14:.*]] = hw.constant false
// CHECK:           %[[VAL_15:.*]] = hw.constant true
// CHECK:           %[[VAL_16:.*]] = comb.xor %[[VAL_5]], %[[VAL_15]] : i1
// CHECK:           %[[VAL_17:.*]] = comb.and %[[VAL_18:.*]], %[[VAL_16]] : i1
// CHECK:           %[[VAL_19:.*]] = seq.compreg %[[VAL_17]], %[[VAL_1]], %[[VAL_2]], %[[VAL_14]]  : i1
// CHECK:           %[[VAL_20:.*]] = comb.xor %[[VAL_19]], %[[VAL_15]] : i1
// CHECK:           %[[VAL_9]] = comb.and %[[VAL_20]], %[[VAL_4]] : i1
// CHECK:           %[[VAL_21:.*]] = comb.and %[[VAL_8]], %[[VAL_9]] : i1
// CHECK:           %[[VAL_18]] = comb.or %[[VAL_21]], %[[VAL_19]] {sv.namehint = "done0"} : i1
// CHECK:           %[[VAL_22:.*]] = comb.xor %[[VAL_5]], %[[VAL_15]] : i1
// CHECK:           %[[VAL_23:.*]] = comb.and %[[VAL_24:.*]], %[[VAL_22]] : i1
// CHECK:           %[[VAL_25:.*]] = seq.compreg %[[VAL_23]], %[[VAL_1]], %[[VAL_2]], %[[VAL_14]]  : i1
// CHECK:           %[[VAL_26:.*]] = comb.xor %[[VAL_25]], %[[VAL_15]] : i1
// CHECK:           %[[VAL_13]] = comb.and %[[VAL_26]], %[[VAL_4]] : i1
// CHECK:           %[[VAL_27:.*]] = comb.and %[[VAL_12]], %[[VAL_13]] : i1
// CHECK:           %[[VAL_24]] = comb.or %[[VAL_27]], %[[VAL_25]] {sv.namehint = "done1"} : i1
// CHECK:           %[[VAL_5]] = comb.and %[[VAL_18]], %[[VAL_24]] {sv.namehint = "allDone"} : i1
// CHECK:           hw.output %[[VAL_7]], %[[VAL_11]] : !esi.channel<i0>, !esi.channel<i0>
// CHECK:         }
hw.module @fork(%t : !dc.token, %clk : i1 {"dc.clock"}, %rst : i1 {"dc.reset"}) -> (out0: !dc.token, out1: !dc.token) {
    %out:2 = dc.fork [2] %t
    hw.output %out#0, %out#1 : !dc.token, !dc.token
}

// CHECK-LABEL:   hw.module @bufferToken(
// CHECK-SAME:              %[[VAL_0:.*]]: !esi.channel<i0>, %[[VAL_1:.*]]: i1 {dc.clock}, %[[VAL_2:.*]]: i1 {dc.reset}) -> (out0: !esi.channel<i0>) {
// CHECK:           %[[VAL_3:.*]] = esi.buffer %[[VAL_1]], %[[VAL_2]], %[[VAL_0]] {stages = 2 : i64} : i0
// CHECK:           hw.output %[[VAL_3]] : !esi.channel<i0>
// CHECK:         }
hw.module @bufferToken(%t1 : !dc.token, %clk : i1 {"dc.clock"}, %rst : i1 {"dc.reset"}) -> (out0: !dc.token) {
    %out = dc.buffer [2] %t1 : !dc.token
    hw.output %out : !dc.token
}

// CHECK-LABEL:   hw.module @bufferValue(
// CHECK-SAME:              %[[VAL_0:.*]]: !esi.channel<i64>, %[[VAL_1:.*]]: i1 {dc.clock}, %[[VAL_2:.*]]: i1 {dc.reset}) -> (out0: !esi.channel<i64>) {
// CHECK:           %[[VAL_3:.*]] = esi.buffer %[[VAL_1]], %[[VAL_2]], %[[VAL_0]] {stages = 2 : i64} : i64
// CHECK:           hw.output %[[VAL_3]] : !esi.channel<i64>
// CHECK:         }
hw.module @bufferValue(%v1 : !dc.value<i64>, %clk : i1 {"dc.clock"}, %rst : i1 {"dc.reset"}) -> (out0: !dc.value<i64>) {
    %out = dc.buffer [2] %v1 : !dc.value<i64>
    hw.output %out : !dc.value<i64>
}

// CHECK-LABEL:   hw.module @branch(
// CHECK-SAME:                      %[[VAL_0:.*]]: !esi.channel<i1>) -> (out0: !esi.channel<i0>, out1: !esi.channel<i0>) {
// CHECK:           %[[VAL_1:.*]], %[[VAL_2:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_3:.*]] : i1
// CHECK:           %[[VAL_4:.*]] = hw.constant 0 : i0
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = esi.wrap.vr %[[VAL_4]], %[[VAL_7:.*]] : i0
// CHECK:           %[[VAL_8:.*]] = hw.constant 0 : i0
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = esi.wrap.vr %[[VAL_8]], %[[VAL_11:.*]] : i0
// CHECK:           %[[VAL_7]] = comb.and %[[VAL_1]], %[[VAL_2]] : i1
// CHECK:           %[[VAL_12:.*]] = hw.constant true
// CHECK:           %[[VAL_13:.*]] = comb.xor %[[VAL_1]], %[[VAL_12]] : i1
// CHECK:           %[[VAL_11]] = comb.and %[[VAL_13]], %[[VAL_2]] : i1
// CHECK:           %[[VAL_14:.*]] = comb.mux %[[VAL_1]], %[[VAL_6]], %[[VAL_10]] : i1
// CHECK:           %[[VAL_3]] = comb.and %[[VAL_14]], %[[VAL_2]] : i1
// CHECK:           hw.output %[[VAL_5]], %[[VAL_9]] : !esi.channel<i0>, !esi.channel<i0>
// CHECK:         }
hw.module @branch(%sel : !dc.value<i1>) -> (out0: !dc.token, out1: !dc.token) {
    %true, %false = dc.branch %sel
    hw.output %true, %false : !dc.token, !dc.token
}

// CHECK-LABEL:   hw.module @select(
// CHECK-SAME:               %[[VAL_0:.*]]: !esi.channel<i1>, %[[VAL_1:.*]]: !esi.channel<i0>, %[[VAL_2:.*]]: !esi.channel<i0>) -> (out0: !esi.channel<i0>) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = esi.unwrap.vr %[[VAL_0]], %[[VAL_5:.*]] : i1
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = esi.unwrap.vr %[[VAL_1]], %[[VAL_8:.*]] : i0
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = esi.unwrap.vr %[[VAL_2]], %[[VAL_11:.*]] : i0
// CHECK:           %[[VAL_12:.*]] = hw.constant 0 : i0
// CHECK:           %[[VAL_13:.*]], %[[VAL_14:.*]] = esi.wrap.vr %[[VAL_12]], %[[VAL_15:.*]] : i0
// CHECK:           %[[VAL_16:.*]] = hw.constant false
// CHECK:           %[[VAL_17:.*]] = comb.concat %[[VAL_16]], %[[VAL_3]] : i1, i1
// CHECK:           %[[VAL_18:.*]] = hw.constant 1 : i2
// CHECK:           %[[VAL_19:.*]] = comb.shl %[[VAL_18]], %[[VAL_17]] : i2
// CHECK:           %[[VAL_20:.*]] = comb.mux %[[VAL_3]], %[[VAL_10]], %[[VAL_7]] : i1
// CHECK:           %[[VAL_15]] = comb.and %[[VAL_20]], %[[VAL_4]] : i1
// CHECK:           %[[VAL_5]] = comb.and %[[VAL_15]], %[[VAL_14]] : i1
// CHECK:           %[[VAL_21:.*]] = comb.extract %[[VAL_19]] from 0 : (i2) -> i1
// CHECK:           %[[VAL_8]] = comb.and %[[VAL_21]], %[[VAL_5]] : i1
// CHECK:           %[[VAL_22:.*]] = comb.extract %[[VAL_19]] from 1 : (i2) -> i1
// CHECK:           %[[VAL_11]] = comb.and %[[VAL_22]], %[[VAL_5]] : i1
// CHECK:           hw.output %[[VAL_13]] : !esi.channel<i0>
// CHECK:         }
hw.module @select(%sel : !dc.value<i1>, %true : !dc.token, %false : !dc.token) -> (out0: !dc.token) {
    %0 = dc.select %sel, %true, %false
    hw.output %0 : !dc.token
}

// CHECK-LABEL:   hw.module @to_from_esi_noop(
// CHECK-SAME:               %[[VAL_0:.*]]: !esi.channel<i0>, %[[VAL_1:.*]]: !esi.channel<i1>) -> (token: !esi.channel<i0>, value: !esi.channel<i1>) {
// CHECK-NEXT:           hw.output %[[VAL_0]], %[[VAL_1]] : !esi.channel<i0>, !esi.channel<i1>
// CHECK-NEXT:         }
hw.module @to_from_esi_noop(%token : !esi.channel<i0>, %value : !esi.channel<i1>) ->
    (token : !esi.channel<i0>, value : !esi.channel<i1>) {
    %token_dc = dc.from_esi %token : !esi.channel<i0>
    %value_dc = dc.from_esi %value : !esi.channel<i1>
    %token_esi = dc.to_esi %token_dc : !dc.token
    %value_esi = dc.to_esi %value_dc : !dc.value<i1>
    hw.output %token_esi, %value_esi : !esi.channel<i0>, !esi.channel<i1>
}
