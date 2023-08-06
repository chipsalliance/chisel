// RUN: circt-opt %s --arc-make-tables | FileCheck %s

// CHECK-LABEL: arc.define @Simple
arc.define @Simple(%arg0: i4) -> i4 {
  // CHECK-NEXT: %0 = hw.aggregate_constant [7 : i4, -2 : i4, -3 : i4, -4 : i4, -5 : i4, -6 : i4, -7 : i4, -8 : i4, 7 : i4, 6 : i4, 5 : i4, 4 : i4, 3 : i4, 2 : i4, 1 : i4, 0 : i4] : !hw.array<16xi4>
  // CHECK-NEXT: hw.array_get %0[%arg0]
  // CHECK-NEXT: arc.output %1
  %c0_i3 = hw.constant 0 : i3
  %c0_i2 = hw.constant 0 : i2
  %false = hw.constant false
  %0 = comb.concat %c0_i3, %arg0 : i3, i4
  %1 = comb.concat %c0_i2, %arg0 : i2, i4
  %2 = comb.concat %false, %arg0 : i1, i4
  %3 = comb.concat %arg0, %false : i4, i1
  %4 = comb.and bin %2, %3 : i5
  %5 = comb.concat %false, %4 : i1, i5
  %6 = comb.add bin %1, %5 : i6
  %7 = comb.and bin %1, %6 : i6
  %8 = comb.concat %false, %7 : i1, i6
  %9 = comb.add bin %0, %8 : i7
  %10 = comb.extract %9 from 0 : (i7) -> i4
  %11 = comb.and %arg0, %10 : i4
  %12 = comb.add %arg0, %11 : i4
  %13 = comb.and %arg0, %12 : i4
  %14 = comb.add %arg0, %13 : i4
  %15 = comb.and %arg0, %14 : i4
  %16 = comb.add %arg0, %15 : i4
  %17 = comb.and %arg0, %16 : i4
  %18 = comb.add %arg0, %17 : i4
  %19 = comb.and %arg0, %18 : i4
  %20 = comb.add %arg0, %19 : i4
  %21 = comb.and %arg0, %20 : i4
  arc.output %21 : i4
}
// CHECK-NEXT: }

// CHECK-LABEL: arc.define @TooManyBits
arc.define @TooManyBits(%arg0: i30) -> i30 {
  // CHECK-NOT: hw.aggregate_constant
  %0 = comb.and %arg0, %arg0 : i30
  %1 = comb.add %arg0, %0 : i30
  %2 = comb.and %arg0, %1 : i30
  %3 = comb.add %arg0, %2 : i30
  %4 = comb.and %arg0, %3 : i30
  %5 = comb.add %arg0, %4 : i30
  %6 = comb.and %arg0, %5 : i30
  %7 = comb.add %arg0, %6 : i30
  %8 = comb.and %arg0, %7 : i30
  %9 = comb.add %arg0, %8 : i30
  %10 = comb.and %arg0, %9 : i30
  %11 = comb.add %arg0, %10 : i30
  %12 = comb.and %arg0, %11 : i30
  %13 = comb.add %arg0, %12 : i30
  %14 = comb.and %arg0, %13 : i30
  %15 = comb.add %arg0, %14 : i30
  %16 = comb.and %arg0, %15 : i30
  %17 = comb.add %arg0, %16 : i30
  %18 = comb.and %arg0, %17 : i30
  %19 = comb.add %arg0, %18 : i30
  %20 = comb.and %arg0, %19 : i30
  // CHECK: arc.output
  arc.output %20 : i30
}
// CHECK-NEXT: }
