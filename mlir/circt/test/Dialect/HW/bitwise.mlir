// RUN: circt-opt %s | FileCheck %s

// CHECK-LABEL: hw.module @bitwise(%arg0: i7, %arg1: i7) -> (r: i21) {
hw.module @bitwise(%arg0: i7, %arg1: i7) -> (r: i21) {
// CHECK-NEXT:    [[RES0:%[0-9]+]] = comb.xor %arg0 : i7
// CHECK-NEXT:    [[RES1:%[0-9]+]] = comb.or  %arg0, %arg1 : i7
// CHECK-NEXT:    [[RES2:%[0-9]+]] = comb.and %arg0, %arg1, %arg0 : i7
  %and1 = comb.xor %arg0 : i7
  %or1  = comb.or  %arg0, %arg1 : i7
  %xor1 = comb.and %arg0, %arg1, %arg0 : i7

// CHECK-NEXT:    [[RESULT:%[0-9]+]] = comb.concat [[RES0]], [[RES1]], [[RES2]] : i7, i7, i7
// CHECK-NEXT:    hw.output [[RESULT]] : i21
  %result = comb.concat %and1, %or1, %xor1 : i7, i7, i7
  hw.output %result : i21
}

// CHECK-LABEL: hw.module @shl_op(%a: i7, %b: i7) -> (r: i7) {
hw.module @shl_op(%a: i7, %b: i7) -> (r: i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.shl  %a, %b : i7
  %0  = comb.shl  %a, %b : i7
// CHECK-NEXT:    hw.output [[RES]]
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @shr_op(%a: i7, %b: i7) -> (r0: i7) {
hw.module @shr_op(%a: i7, %b: i7) -> (r0: i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.shru %a, %b : i7
  %0  = comb.shru %a, %b : i7
// CHECK-NEXT:    hw.output [[RES]]
  hw.output %0 : i7
}

// CHECK-LABEL: hw.module @casts(%in1: i7) -> (x: !hw.struct<int: i7>)
hw.module @casts(%in1: i7) -> (x: !hw.struct<int: i7>) {
  // CHECK-NEXT: %0 = hw.bitcast %in1 : (i7) -> !hw.array<7xi1>
  // CHECK-NEXT: %1 = hw.bitcast %0 : (!hw.array<7xi1>) -> !hw.struct<int: i7>
  %bits = hw.bitcast %in1 : (i7) -> !hw.array<7xi1>
  %backToInt = hw.bitcast %bits : (!hw.array<7xi1>) -> !hw.struct<int: i7>
  hw.output %backToInt : !hw.struct<int: i7>
}
