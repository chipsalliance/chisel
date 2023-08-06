// RUN: circt-opt --hw-eliminate-inout-ports %s | FileCheck %s

// CHECK-LABEL:   hw.module @read(
// CHECK-SAME:                    %[[VAL_0:.*]]: i42) -> (out: i42) {
// CHECK:           hw.output %[[VAL_0]] : i42
// CHECK:         }
hw.module @read(%a: !hw.inout<i42>) -> (out: i42) {
  %aget = sv.read_inout %a: !hw.inout<i42>
  hw.output %aget : i42
}

// CHECK-LABEL:   hw.module @write() -> (a_wr: i42) {
// CHECK:           %[[VAL_0:.*]] = hw.constant 0 : i42
// CHECK:           hw.output %[[VAL_0]] : i42
// CHECK:         }
hw.module @write(%a: !hw.inout<i42>) {
  %0 = hw.constant 0 : i42
  sv.assign %a, %0 : i42
}

// CHECK-LABEL:   hw.module @read_write(
// CHECK-SAME:                          %[[VAL_0:.*]]: i42) -> (a_wr: i42, out: i42) {
// CHECK:           hw.output %[[VAL_0]], %[[VAL_0]] : i42, i42
// CHECK:         }
hw.module @read_write(%a: !hw.inout<i42>) -> (out: i42) {
  %aget = sv.read_inout %a: !hw.inout<i42>
  sv.assign %a, %aget : i42
  hw.output %aget : i42
}

// CHECK-LABEL:   hw.module @oneLevel() {
// CHECK:           %[[VAL_0:.*]] = sv.wire : !hw.inout<i42>
// CHECK:           %[[VAL_1:.*]] = sv.read_inout %[[VAL_0]] : !hw.inout<i42>
// CHECK:           %[[VAL_2:.*]] = hw.instance "read" @read(a_rd: %[[VAL_1]]: i42) -> (out: i42)
// CHECK:           sv.assign %[[VAL_0]], %[[VAL_3:.*]] : i42
// CHECK:           %[[VAL_3]] = hw.instance "write" @write() -> (a_wr: i42)
// CHECK:           %[[VAL_4:.*]] = sv.read_inout %[[VAL_0]] : !hw.inout<i42>
// CHECK:           sv.assign %[[VAL_0]], %[[VAL_5:.*]] : i42
// CHECK:           %[[VAL_5]], %[[VAL_6:.*]] = hw.instance "readWrite" @read_write(a_rd: %[[VAL_4]]: i42) -> (a_wr: i42, out: i42)
// CHECK:           hw.output
// CHECK:         }
hw.module @oneLevel() {
  // No error here, even though the inout is written in two places. The
  // pass will only error upon the recursive case when it inspects a module
  // and sees that there are multiple writers to an inout *port*.
  %0 = sv.wire : !hw.inout<i42>
  %read = hw.instance "read" @read(a : %0 : !hw.inout<i42>) -> (out: i42)
  hw.instance "write" @write(a : %0 : !hw.inout<i42>) -> ()
  %read_write = hw.instance "readWrite" @read_write(a : %0 : !hw.inout<i42>) -> (out: i42)
}


// CHECK-LABEL:   hw.module @passthrough() -> (a_wr: i42) {
// CHECK:           %[[VAL_0:.*]] = hw.instance "write" @write() -> (a_wr: i42)
// CHECK:           hw.output %[[VAL_0]] : i42
// CHECK:         }
hw.module @passthrough(%a : !hw.inout<i42>) -> () {
  hw.instance "write" @write(a : %a : !hw.inout<i42>) -> ()
}

// CHECK-LABEL:   hw.module @passthroughTwoLevels() {
// CHECK:           %[[VAL_0:.*]] = sv.wire : !hw.inout<i42>
// CHECK:           sv.assign %[[VAL_0]], %[[VAL_1:.*]] : i42
// CHECK:           %[[VAL_1]] = hw.instance "passthrough" @passthrough() -> (a_wr: i42)
// CHECK:           hw.output
// CHECK:         }
hw.module @passthroughTwoLevels() {
  %0 = sv.wire : !hw.inout<i42>
  hw.instance "passthrough" @passthrough(a : %0 : !hw.inout<i42>) -> ()
}


// For now, we don't support/touch inout ports. We add a test here to add
// an early detection signal if something changes in the inout port handling
// logic in CIRCT.

// CHECK-LABEL:   hw.module @outputInout() -> (out: !hw.inout<i42>) {
// CHECK:           %[[VAL_0:.*]] = sv.wire : !hw.inout<i42>
// CHECK:           hw.output %[[VAL_0]] : !hw.inout<i42>
// CHECK:         }
hw.module @outputInout() -> (out : !hw.inout<i42>) {
  %0 = sv.wire : !hw.inout<i42>
  hw.output %0 : !hw.inout<i42>
}

// CHECK-LABEL:   hw.module @outputInoutDriver() {
// CHECK:           %[[VAL_0:.*]] = hw.instance "outputInout" @outputInout() -> (out: !hw.inout<i42>)
// CHECK:           sv.assign %[[VAL_0]], %[[VAL_1:.*]] : i42
// CHECK:           %[[VAL_1]] = hw.instance "write" @write() -> (a_wr: i42)
// CHECK:           hw.output
// CHECK:         }
hw.module @outputInoutDriver() {
  %0 = hw.instance "outputInout" @outputInout() -> (out : !hw.inout<i42>)
  hw.instance "write" @write(a : %0 : !hw.inout<i42>) -> ()
}
