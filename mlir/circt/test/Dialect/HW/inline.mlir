// RUN: circt-opt -inline %s | FileCheck %s

// Test whether the _pure_ ops in the `hw` dialect can be inlined by the
// upstream pass. See: `HWInlinerInterface::isLegalToInline`

!struct = !hw.struct<a: i4, b: i4>
!enum = !hw.enum<A, B>
!array = !hw.array<4xi1>
!double_array = !hw.array<8xi1>
!union = !hw.union<a: i4, b: i8>

func.func @constant() -> i4 {
  %0 = hw.constant 1 : i4
  return %0 : i4
}

func.func @aggregate_constant() -> !struct {
  %0 = hw.aggregate_constant [0 : i4, 1 : i4] : !struct
  return %0 : !struct
}

func.func @enum_constant() -> !enum {
  %0 = hw.enum.constant A : !enum
  return %0 : !enum
}

func.func @bitcast(%arg : i4) -> !array {
  %0 = hw.bitcast %arg : (i4) -> !array
  return %0 : !array
}

func.func @array_create(%arg : i1) -> !array {
  %0 = hw.array_create %arg, %arg, %arg, %arg : i1
  return %0 : !array
}

func.func @array_concat(%arg : !array) -> !double_array {
  %0 = hw.array_concat %arg, %arg : !array, !array
  return %0 : !double_array
}

func.func @array_slice(%arg : !double_array) -> !array {
  %0 = hw.constant 2 : i3
  %1 = hw.array_slice %arg[%0] : (!double_array) -> !array
  return %1 : !array
}

func.func @array_get(%arg : !array) -> i1 {
  %0 = hw.constant 0 : i2
  %1 = hw.array_get %arg[%0] : !array, i2
  return %1 : i1
}

func.func @struct_create(%arg : i4) -> !struct {
  %0 = hw.struct_create(%arg, %arg) : !struct
  return %0 : !struct
}

func.func @struct_explode(%arg : !struct) -> i4 {
  %0:2 = hw.struct_explode %arg : !struct
  return %0#1 : i4
}

func.func @struct_extract(%arg : !struct) -> i4 {
  %0 = hw.struct_extract %arg["a"] : !struct
  return %0 : i4
}

func.func @struct_inject(%arg0 : !struct, %arg1 : i4) -> !struct {
  %0 = hw.struct_inject %arg0["b"], %arg1 : !struct
  return %0 : !struct
}

func.func @union_create(%arg : i4) -> !union {
  %0 = hw.union_create "a", %arg : !union
  return %0 : !union
}

func.func @union_extract(%arg : !union) -> i4 {
  %0 = hw.union_extract %arg["a"] : !union
  return %0 : i4
}

// CHECK-LABEL: @test_inliner
func.func @test_inliner() ->
    (i4, !struct, !enum, !array, !array, !double_array, !array, i1, !struct,
     i4, i4, !struct, !union, i4) {
  // CHECK-NOT: {{.*}} = call
  %true = hw.constant true
  %0 = call @constant() : () -> i4
  %1 = call @aggregate_constant() : () -> !struct
  %2 = call @enum_constant() : () -> !enum
  %3 = call @bitcast(%0) : (i4) -> !array
  %4 = call @array_create(%true) : (i1) -> !array
  %5 = call @array_concat(%4) : (!array) -> !double_array
  %6 = call @array_slice(%5) : (!double_array) -> !array
  %7 = call @array_get(%6) : (!array) -> i1
  %8 = call @struct_create(%0) : (i4) -> !struct
  %9 = call @struct_explode(%8) : (!struct) -> i4
  %10 = call @struct_extract(%8) : (!struct) -> i4
  %11 = call @struct_inject(%8, %9) : (!struct, i4) -> !struct
  %12 = call @union_create(%0) : (i4) -> !union
  %13 = call @union_extract(%12) : (!union) -> i4
  return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13 :
      i4, !struct, !enum, !array, !array, !double_array, !array, i1, !struct,
      i4, i4, !struct, !union, i4
}
