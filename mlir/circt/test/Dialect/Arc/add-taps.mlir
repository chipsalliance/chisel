// RUN: circt-opt %s --arc-add-taps | FileCheck %s

// CHECK-LABEL: hw.module @ObservePorts
hw.module @ObservePorts(%x: i4, %y: i4) -> (u: i4, v: i4) {
  // CHECK-NEXT: arc.tap %x {name = "x"} : i4
  // CHECK-NEXT: arc.tap %y {name = "y"} : i4
  // CHECK-NEXT: %0 = comb.add
  // CHECK-NEXT: %1 = comb.sub
  %0 = comb.add %x, %y : i4
  %1 = comb.sub %x, %y : i4
  // CHECK-NEXT: arc.tap %0 {name = "u"} : i4
  // CHECK-NEXT: arc.tap %1 {name = "v"} : i4
  // CHECK-NEXT: hw.output
  hw.output %0, %1 : i4, i4
}
// CHECK-NEXT: }


// CHECK-LABEL: hw.module @ObserveWires
hw.module @ObserveWires() {
  // CHECK-NEXT: arc.tap [[RD:%.+]] {name = "x"} : i4
  // CHECK-NEXT: %x = sv.wire
  // CHECK-NEXT: [[RD]] = sv.read_inout %x
  %x = sv.wire : !hw.inout<i4>
  %0 = sv.read_inout %x : !hw.inout<i4>

  // CHECK-NEXT: [[RD:%.+]] = sv.read_inout %y
  // CHECK-NEXT: arc.tap [[RD]] {name = "y"} : i4
  // CHECK-NEXT: %y = sv.wire
  %y = sv.wire : !hw.inout<i4>

  // CHECK-NEXT: hw.constant
  // CHECK-NEXT: arc.tap %z {name = "z"} : i4
  // CHECK-NEXT: %z = hw.wire
  %c0_i4 = hw.constant 0 : i4
  %z = hw.wire %c0_i4 : i4

  // CHECK-NEXT: hw.output
}
// CHECK-NEXT: }

