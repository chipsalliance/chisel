// RUN: circt-opt %s --arc-lower-lut | FileCheck %s

// CHECK-LABEL: arc.define @checkEndianess(%arg0: i1, %arg1: i1, %arg2: i1) -> i3 {
// CHECK-NEXT:   %c-342392_i24 = hw.constant -342392 : i24
// CHECK-NEXT:   %c0_i21 = hw.constant 0 : i21
// CHECK-NEXT:   %0 = comb.concat %c0_i21, %arg0, %arg1, %arg2 : i21, i1, i1, i1
// CHECK-NEXT:   %c3_i24 = hw.constant 3 : i24
// CHECK-NEXT:   %1 = comb.mul %0, %c3_i24 : i24
// CHECK-NEXT:   %2 = comb.shru %c-342392_i24, %1 : i24
// CHECK-NEXT:   %3 = comb.extract %2 from 0 : (i24) -> i3
// CHECK-NEXT:   arc.output %3 : i3
// CHECK-NEXT: }
arc.define @checkEndianess(%arg0: i1, %arg1: i1, %arg2: i1) -> i3 {
  %0 = arc.lut(%arg0, %arg1, %arg2) : (i1, i1, i1) -> i3 {
  ^bb0(%arg3: i1, %arg4: i1, %arg5: i1):
    %1 = comb.concat %arg3, %arg4, %arg5 : i1, i1, i1
    // Desired LUT: 111_110_101_100_011_010_001_000 (underscores are just for readability)
    // This is -342392 in decimal
    arc.output %1 : i3
  }
  arc.output %0 : i3
}

// CHECK-LABEL:  arc.define @integerLut(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1, %arg4: i1) -> i1 {
// CHECK-NEXT:    %c15921664_i32 = hw.constant 15921664 : i32
// CHECK-NEXT:    %c0_i27 = hw.constant 0 : i27
// CHECK-NEXT:    %0 = comb.concat %c0_i27, %arg0, %arg1, %arg4, %arg2, %arg3 : i27, i1, i1, i1, i1, i1
// CHECK-NEXT:    %c1_i32 = hw.constant 1 : i32
// CHECK-NEXT:    %1 = comb.mul %0, %c1_i32 : i32
// CHECK-NEXT:    %2 = comb.shru %c15921664_i32, %1 : i32
// CHECK-NEXT:    %3 = comb.extract %2 from 0 : (i32) -> i1
// CHECK-NEXT:    arc.output %3 : i1
// CHECK-NEXT:  }
arc.define @integerLut(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1, %arg4: i1) -> i1 {
  %0 = arc.lut(%arg0, %arg1, %arg4, %arg2, %arg3) : (i1, i1, i1, i1, i1) -> i1 {
  ^bb0(%arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1, %arg9: i1):
    %1 = comb.and %arg8, %arg9 : i1
    %2 = comb.icmp ne %arg7, %1 : i1
    %3 = comb.mux %2, %arg7, %arg9 : i1
    %4 = comb.xor %arg5, %arg6 : i1
    %5 = comb.and %4, %3 : i1
    arc.output %5 : i1
  }
  arc.output %0 : i1
}

// CHECK-NEXT: arc.define @twoLutInSequence(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1, %arg4: i1, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1) -> i1 {
// CHECK-NEXT:   %c-1565873536_i32 = hw.constant -1565873536 : i32
// CHECK-NEXT:   %c0_i27 = hw.constant 0 : i27
// CHECK-NEXT:   %0 = comb.concat %c0_i27, %arg3, %arg4, %arg5, %arg6, %arg7 : i27, i1, i1, i1, i1, i1
// CHECK-NEXT:   %c1_i32 = hw.constant 1 : i32
// CHECK-NEXT:   %1 = comb.mul %0, %c1_i32 : i32
// CHECK-NEXT:   %2 = comb.shru %c-1565873536_i32, %1 : i32
// CHECK-NEXT:   %3 = comb.extract %2 from 0 : (i32) -> i1
// CHECK-NEXT:   %c-1613786944_i32 = hw.constant -1613786944 : i32
// CHECK-NEXT:   %c0_i27_0 = hw.constant 0 : i27
// CHECK-NEXT:   %4 = comb.concat %c0_i27_0, %3, %arg0, %arg1, %arg2, %arg8 : i27, i1, i1, i1, i1, i1
// CHECK-NEXT:   %c1_i32_1 = hw.constant 1 : i32
// CHECK-NEXT:   %5 = comb.mul %4, %c1_i32_1 : i32
// CHECK-NEXT:   %6 = comb.shru %c-1613786944_i32, %5 : i32
// CHECK-NEXT:   %7 = comb.extract %6 from 0 : (i32) -> i1
// CHECK-NEXT:   arc.output %7 : i1
// CHECK-NEXT: }
arc.define @twoLutInSequence(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1, %arg4: i1, %arg5: i1, %arg6: i1, %arg7: i1, %arg8: i1) -> i1 {
  %0 = arc.lut(%arg3, %arg4, %arg5, %arg6, %arg7) : (i1, i1, i1, i1, i1) -> i1 {
  ^bb0(%arg9: i1, %arg10: i1, %arg12: i1, %arg11: i1, %arg13: i1):
    %2 = comb.xor %arg12, %arg10 : i1
    %3 = comb.and %arg11, %2 : i1
    %4 = comb.xor %3, %arg10 : i1
    %5 = comb.xor %arg9, %arg10 : i1
    %6 = comb.or %5, %4 : i1
    %7 = comb.and %6, %arg13 : i1
    arc.output %7 : i1
  }
  %1 = arc.lut(%0, %arg0, %arg1, %arg2, %arg8) : (i1, i1, i1, i1, i1) -> i1 {
  ^bb0(%arg13: i1, %arg12: i1, %arg9: i1, %arg11: i1, %arg10: i1):
    %2 = comb.icmp ne %arg11, %arg12 : i1
    %3 = comb.mux %arg10, %arg11, %2 : i1
    %4 = comb.mux %arg9, %3, %arg13 : i1
    arc.output %4 : i1
  }
  arc.output %1 : i1
}

// CHECK-LABEL: arc.define @arrayLut(%arg0: i1, %arg1: i1, %arg2: i2, %arg3: i2, %arg4: i2) -> i2 {
// CHECK-NEXT:   %0 = hw.aggregate_constant [-2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, -2 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, -2 : i2, 0 : i2, 0 : i2, -2 : i2, -2 : i2, 0 : i2, 0 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -1 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, 1 : i2, 1 : i2, 1 : i2, 1 : i2, 1 : i2, 1 : i2, 1 : i2, 1 : i2, -1 : i2, 1 : i2, 1 : i2, 1 : i2, 1 : i2, 1 : i2, 1 : i2, 1 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 1 : i2, 0 : i2, 0 : i2, -2 : i2, 0 : i2, 0 : i2, -1 : i2, -2 : i2, 1 : i2, 0 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -1 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -1 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, 1 : i2, 1 : i2, 1 : i2, 1 : i2, 1 : i2, 1 : i2, 1 : i2, 1 : i2, -1 : i2, 1 : i2, 1 : i2, 1 : i2, 1 : i2, 1 : i2, 1 : i2, 1 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 1 : i2, 0 : i2, 0 : i2, -2 : i2, 0 : i2, 0 : i2, -1 : i2, -2 : i2, 1 : i2, 0 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, -2 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, -2 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, 0 : i2, -2 : i2, 0 : i2, 0 : i2, -2 : i2, -2 : i2, 0 : i2, 0 : i2] : !hw.array<256xi2>
// CHECK-NEXT:   %1 = comb.concat %arg0, %arg1, %arg4, %arg2, %arg3 : i1, i1, i2, i2, i2
// CHECK-NEXT:   %2 = hw.array_get %0[%1] : !hw.array<256xi2>, i8
// CHECK-NEXT:   arc.output %2 : i2
// CHECK-NEXT: }
arc.define @arrayLut(%arg0: i1, %arg1: i1, %arg2: i2, %arg3: i2, %arg4: i2) -> i2 {
  %0 = arc.lut(%arg0, %arg1, %arg4, %arg2, %arg3) : (i1, i1, i2, i2, i2) -> i2 {
  ^bb0(%arg5: i1, %arg6: i1, %arg7: i2, %arg8: i2, %arg9: i2):
    %true = hw.constant true
    %1 = comb.and %arg8, %arg9 : i2
    %2 = comb.icmp ne %arg7, %1 : i2
    %3 = comb.mux %2, %arg7, %arg9 : i2
    %4 = comb.xor %arg5, %arg6 : i1
    %5 = comb.concat %true, %4 : i1, i1
    %6 = comb.and %5, %3 : i2
    arc.output %6 : i2
  }
  arc.output %0 : i2
}
