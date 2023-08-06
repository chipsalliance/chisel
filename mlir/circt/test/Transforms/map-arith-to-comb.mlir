
// RUN: circt-opt -split-input-file -verify-diagnostics --pass-pipeline='builtin.module(any(map-arith-to-comb))' %s | FileCheck %s

// CHECK-LABEL:   hw.module @foo(
// CHECK-SAME:          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1) {
// CHECK:           %[[VAL_3:.*]] = comb.add %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_4:.*]] = comb.sub %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_5:.*]] = comb.mul %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = comb.divs %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_7:.*]] = comb.divu %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_8:.*]] = comb.mods %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_9:.*]] = comb.modu %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_10:.*]] = comb.and %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_11:.*]] = comb.or %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_12:.*]] = comb.xor %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_13:.*]] = comb.shl %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_14:.*]] = comb.shrs %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_15:.*]] = comb.shru %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_16:.*]] = comb.mux %[[VAL_2]], %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_17:.*]] = comb.extract %[[VAL_1]] from 31 : (i32) -> i1
// CHECK:           %[[VAL_18:.*]] = comb.concat %[[VAL_17]], %[[VAL_1]] : i1, i32
// CHECK:           %[[VAL_19:.*]] = hw.constant 0 : i3
// CHECK:           %[[VAL_20:.*]] = comb.concat %[[VAL_19]], %[[VAL_1]] : i3, i32
// CHECK:           %[[VAL_21:.*]] = comb.extract %[[VAL_1]] from 0 : (i32) -> i16
// CHECK:           hw.output
// CHECK:         }

hw.module @foo(%arg0 : i32, %arg1 : i32, %arg2 : i1) -> () {
    %0 = arith.addi %arg0, %arg1 : i32
    %1 = arith.subi %arg0, %arg1 : i32
    %2 = arith.muli %arg0, %arg1 : i32
    %3 = arith.divsi %arg0, %arg1 : i32
    %4 = arith.divui %arg0, %arg1 : i32
    %5 = arith.remsi %arg0, %arg1 : i32
    %6 = arith.remui %arg0, %arg1 : i32
    %7 = arith.andi  %arg0, %arg1 : i32
    %8 = arith.ori   %arg0, %arg1 : i32
    %9 = arith.xori  %arg0, %arg1 : i32
    %10 = arith.shli %arg0, %arg1 : i32
    %11 = arith.shrsi %arg0, %arg1 : i32
    %12 = arith.shrui %arg0, %arg1 : i32
    %13 = arith.select %arg2, %arg0, %arg1 : i32
    %14 = arith.extsi %arg1 : i32 to i33
    %15 = arith.extui %arg1 : i32 to i35
    %16 = arith.trunci %arg1 : i32 to i16
    %17 = arith.cmpi slt, %arg0, %arg1 : i32
}

// -----

hw.module @invalidVector(%arg0 : vector<4xi32>) -> () {
  // expected-error @+1 {{failed to legalize operation 'arith.extsi' that was explicitly marked illegal}}
    %0 = arith.extsi %arg0 : vector<4xi32> to vector<4xi33>
}
