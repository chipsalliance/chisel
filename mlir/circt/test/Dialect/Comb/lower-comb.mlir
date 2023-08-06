// RUN: circt-opt %s --lower-comb | FileCheck %s

// CHECK-LABEL: hw.module @ex2in
// CHECK:         [[R0:%.+]] = comb.mux %b, %true, %false : i1
// CHECK:         [[R1:%.+]] = comb.mux %b, %false, %true : i1
// CHECK:         [[R2:%.+]] = comb.mux %a, [[R0]], [[R1]] {sv.namehint = "lut1"} : i1
hw.module @ex2in(%a: i1, %b: i1) -> (x: i1) {
  %0 = comb.truth_table %a, %b -> [true, false, false, true] {sv.namehint="lut1"}
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @ex0in
// CHECK-NEXT:    [[R0:%.+]] = hw.constant true {sv.namehint = "lut1"}
// CHECK-NEXT:    [[R1:%.+]] = hw.constant false
// CHECK-NEXT:    hw.output [[R0]] : i1
hw.module @ex0in() -> (x: i1) {
  %0 = comb.truth_table -> [true] {sv.namehint="lut1"}
  hw.output %0 : i1
}
