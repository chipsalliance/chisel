// RUN: circt-opt %s --split-input-file --hw-specialize | FileCheck %s

// Test two different ways of instantiating a generic module.

module {

// CHECK-LABEL:   hw.module @addToFirst_N_4_X_8(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !hw.array<4xi8>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: i8) -> (out: i8) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_3:.*]] = comb.extract %[[VAL_2]] from 0 : (i64) -> i2
// CHECK:           %[[VAL_4:.*]] = hw.array_get %[[VAL_0]]{{\[}}%[[VAL_3]]] : !hw.array<4xi8>
// CHECK:           %[[VAL_5:.*]] = comb.add %[[VAL_4]], %[[VAL_1]] : i8
// CHECK:           hw.output %[[VAL_5]] : i8
// CHECK:         }

// CHECK-LABEL:   hw.module @addToFirst_N_5_X_9(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !hw.array<5xi9>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: i9) -> (out: i9) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_3:.*]] = comb.extract %[[VAL_2]] from 0 : (i64) -> i3
// CHECK:           %[[VAL_4:.*]] = hw.array_get %[[VAL_0]]{{\[}}%[[VAL_3]]] : !hw.array<5xi9>
// CHECK:           %[[VAL_5:.*]] = comb.add %[[VAL_4]], %[[VAL_1]] : i9
// CHECK:           hw.output %[[VAL_5]] : i9
// CHECK:         }
  hw.module @addToFirst<N: i32, X: i32>(
      %vec : !hw.array<#hw.param.decl.ref<"N"> x !hw.int<#hw.param.decl.ref<"X">>>,
      %a : !hw.int<#hw.param.decl.ref<"X">>) -> (out: !hw.int<#hw.param.decl.ref<"X">>) {
    %c0 = arith.constant 0 : i64
    %first = hw.array_get %vec[%c0] : !hw.array<#hw.param.decl.ref<"N"> x !hw.int<#hw.param.decl.ref<"X">>>, i64
    %0 = comb.add %first, %a : !hw.int<#hw.param.decl.ref<"X">>
    hw.output %0 : !hw.int<#hw.param.decl.ref<"X">>
  }

// CHECK-LABEL:   hw.module @top(
// CHECK-SAME:        %[[VAL_0:.*]]: !hw.array<4xi8>, %[[VAL_1:.*]]: i8, %[[VAL_2:.*]]: !hw.array<5xi9>, %[[VAL_3:.*]]: i9) -> (out1: i8, out2: i9) {
// CHECK:           %[[VAL_4:.*]] = hw.instance "inst1" @addToFirst_N_4_X_8(vec: %[[VAL_0]]: !hw.array<4xi8>, a: %[[VAL_1]]: i8) -> (out: i8)
// CHECK:           %[[VAL_5:.*]] = hw.instance "inst2" @addToFirst_N_5_X_9(vec: %[[VAL_2]]: !hw.array<5xi9>, a: %[[VAL_3]]: i9) -> (out: i9)
// CHECK:           hw.output %[[VAL_4]], %[[VAL_5]] : i8, i9
// CHECK:         }
  hw.module @top(
      %vec1 : !hw.array<4 x !hw.int<8>>, %a1 : !hw.int<8>,
      %vec2 : !hw.array<5 x !hw.int<9>>, %a2 : !hw.int<9>) ->
      (out1: !hw.int<8>, out2: !hw.int<9>) {
    %0 = hw.instance "inst1" @addToFirst<N: i32 = 4, X: i32 = 8>
      (vec: %vec1 : !hw.array<4 x !hw.int<8>>, a: %a1 : !hw.int<8>) -> (out: !hw.int<8>)
    %1 = hw.instance "inst2" @addToFirst<N: i32 = 5, X: i32 = 9>
      (vec: %vec2 : !hw.array<5 x !hw.int<9>>, a: %a2 : !hw.int<9>) -> (out: !hw.int<9>)
    hw.output %0, %1 : i8, i9
  }
}

// -----

// Test hw.param.value.

module {

// CHECK-LABEL:   hw.module @constantGen_V_8() -> (out: i64) {
// CHECK:           %[[VAL_0:.*]] = hw.constant 8 : i64
// CHECK:           hw.output %[[VAL_0]] : i64
// CHECK:         }

// CHECK-LABEL:   hw.module @constantGen_V_9() -> (out: i64) {
// CHECK:           %[[VAL_0:.*]] = hw.constant 9 : i64
// CHECK:           hw.output %[[VAL_0]] : i64
// CHECK:         }
  hw.module @constantGen<V: i64>() -> (out: i64) {
    %0 = hw.param.value i64 = #hw.param.decl.ref<"V">
    hw.output %0 :i64
  }

// CHECK-LABEL:  hw.module @top() -> (out1: i64, out2: i64) {
// CHECK:          %inst1.out = hw.instance "inst1" @constantGen_V_8() -> (out: i64)
// CHECK:          %inst2.out = hw.instance "inst2" @constantGen_V_9() -> (out: i64)
// CHECK:          hw.output %inst1.out, %inst2.out : i64, i64
// CHECK:        }
  hw.module @top() -> (out1: i64, out2: i64) {
    %0 = hw.instance "inst1" @constantGen<V: i64 = 8> () -> (out: i64)
    %1 = hw.instance "inst2" @constantGen<V: i64 = 9> () -> (out: i64)
    hw.output %0, %1 : i64, i64
  }
}

// -----

// Test two identical instances of the same module.

module {

  hw.module @constantGen<V: i64>() -> (out: i64) {
    %0 = hw.param.value i64 = #hw.param.decl.ref<"V">
    hw.output %0 :i64
  }

// CHECK-LABEL:   hw.module @top() -> (out1: i64, out2: i64) {
// CHECK:           %[[VAL_0:.*]] = hw.instance "inst1" @constantGen_V_8() -> (out: i64)
// CHECK:           %[[VAL_1:.*]] = hw.instance "inst2" @constantGen_V_8() -> (out: i64)
// CHECK:           hw.output %[[VAL_0]], %[[VAL_1]] : i64, i64
// CHECK:         }
  hw.module @top() -> (out1: i64, out2: i64) {
    %0 = hw.instance "inst1" @constantGen<V: i64 = 8> () -> (out: i64)
    %1 = hw.instance "inst2" @constantGen<V: i64 = 8> () -> (out: i64)
    hw.output %0, %1 : i64, i64
  }
}

// -----

// Test unique, non-aliasing module name generation.

module {

// CHECK-LABEL:   hw.module @constantGen_V_5_1() -> (out: i64) {
  hw.module @constantGen<V: i64>() -> (out: i64) {
    %0 = hw.param.value i64 = #hw.param.decl.ref<"V">
    hw.output %0 :i64
  }

  hw.module.extern @constantGen_V_5<V: i64>() -> (out: i64)
  hw.module.extern @constantGen_V_5_0<V: i64>() -> (out: i64)

  hw.module @top() -> (out1: i64) {
// CHECK:           %[[VAL_0:.*]] = hw.instance "inst1" @constantGen_V_5_1() -> (out: i64)
    %0 = hw.instance "inst1" @constantGen<V: i64 = 5> () -> (out: i64)
    hw.output %0 : i64
  }
}

// -----

// Test a parent module instantiating parametric modules using its parameters

module {
  hw.module @constantGen<V: i32>() -> (out: i32) {
    %0 = hw.param.value i32 = #hw.param.decl.ref<"V">
    hw.output %0 : i32
  }

  hw.module @takesParametericWidth<width: i32>(
    %in: !hw.int<#hw.param.decl.ref<"width">>) {}

  hw.module @usesConstantGen<V: i32>(
    %in: !hw.int<#hw.param.decl.ref<"V">>) -> (out: i32) {
    %0 = hw.instance "inst1" @constantGen<V: i32 = #hw.param.decl.ref<"V">> () -> (out: i32)
    hw.instance "inst2" @takesParametericWidth<width: i32 = #hw.param.decl.ref<"V">>(in: %in : !hw.int<#hw.param.decl.ref<"V">>) -> ()
    hw.output %0 : i32
  }

  // CHECK-LABEL: hw.module @usesConstantGen_V_8(%in: i8) -> (out: i32) {
  // CHECK:         %[[VAL_0:.*]] = hw.instance "inst1" @constantGen_V_8() -> (out: i32)
  // CHECK:         hw.instance "inst2" @takesParametericWidth_width_8(in: %in: i8) -> ()
  // CHECK:         hw.output %[[VAL_0]] : i32
  // CHECK:       }

  // CHECK-LABEL: hw.module @constantGen_V_8() -> (out: i32) {
  // CHECK:         %[[VAL_0:.*]] = hw.constant 8 : i32
  // CHECK:         hw.output %[[VAL_0]] : i32
  // CHECK:       }

  // CHECK-LABEL: hw.module @takesParametericWidth_width_8(%in: i8) {
  // CHECK:          hw.output
  // CHECK:       }

  hw.module @top() -> (out: i32) {
    %in = hw.constant 1 : i8
    %0 = hw.instance "inst" @usesConstantGen<V: i32 = 8> (in: %in: i8) -> (out: i32)
    hw.output %0 : i32
  }
}
