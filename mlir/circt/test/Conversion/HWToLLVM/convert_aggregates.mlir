// RUN: circt-opt %s --convert-hw-to-llvm | FileCheck %s

// CHECK-LABEL: @convertBitcast
func.func @convertBitcast(%arg0 : i32, %arg1: !hw.array<2xi32>, %arg2: !hw.struct<foo: i32, bar: i32>) {
  // CHECK-NEXT: %[[AARG1:.*]] = builtin.unrealized_conversion_cast %arg1 : !hw.array<2xi32> to !llvm.array<2 x i32>
  // CHECK-NEXT: %[[AARG2:.*]] = builtin.unrealized_conversion_cast %arg2 : !hw.struct<foo: i32, bar: i32> to !llvm.struct<(i32, i32)>

  // CHECK-NEXT: %[[ONE1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[A1:.*]] = llvm.alloca %[[ONE1]] x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr<i32>
  // CHECK-NEXT: llvm.store %arg0, %[[A1]] : !llvm.ptr<i32>
  // CHECK-NEXT: %[[B1:.*]] = llvm.bitcast %[[A1]] : !llvm.ptr<i32> to !llvm.ptr<array<4 x i8>>
  // CHECK-NEXT: llvm.load %[[B1]] : !llvm.ptr<array<4 x i8>>
  %0 = hw.bitcast %arg0 : (i32) -> !hw.array<4xi8>

  // CHECK-NEXT: %[[ONE2:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[A2:.*]] = llvm.alloca %[[ONE2]] x !llvm.array<2 x i32> {alignment = 4 : i64} : (i32) -> !llvm.ptr<array<2 x i32>>
  // CHECK-NEXT: llvm.store %[[AARG1]], %[[A2]] : !llvm.ptr<array<2 x i32>>
  // CHECK-NEXT: %[[B2:.*]] = llvm.bitcast %[[A2]] : !llvm.ptr<array<2 x i32>> to !llvm.ptr<i64>
  // CHECK-NEXT: llvm.load %[[B2]] : !llvm.ptr<i64>
  %1 = hw.bitcast %arg1 : (!hw.array<2xi32>) -> i64

  // CHECK-NEXT: %[[ONE3:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[A3:.*]] = llvm.alloca %[[ONE3]] x !llvm.struct<(i32, i32)> {alignment = 4 : i64} : (i32) -> !llvm.ptr<struct<(i32, i32)>>
  // CHECK-NEXT: llvm.store %[[AARG2]], %[[A3]] : !llvm.ptr<struct<(i32, i32)>>
  // CHECK-NEXT: %[[B3:.*]] = llvm.bitcast %[[A3]] : !llvm.ptr<struct<(i32, i32)>> to !llvm.ptr<i64>
  // CHECK-NEXT: llvm.load %[[B3]] : !llvm.ptr<i64>
  %2 = hw.bitcast %arg2 : (!hw.struct<foo: i32, bar: i32>) -> i64

  return
}

// CHECK-LABEL: @convertArray
func.func @convertArray(%arg0 : i1, %arg1: !hw.array<2xi32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  // CHECK-NEXT: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %arg1 : !hw.array<2xi32> to !llvm.array<2 x i32>

  // CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[ALLOCA:.*]] = llvm.alloca %[[ONE]] x !llvm.array<2 x i32> {alignment = 4 : i64} : (i32) -> !llvm.ptr<array<2 x i32>>
  // CHECK-NEXT: llvm.store %[[CAST0]], %[[ALLOCA]] : !llvm.ptr<array<2 x i32>>
  // CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[ZEXT:.*]] = llvm.zext %arg0 : i1 to i2
  // CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][%[[ZERO]], %[[ZEXT]]] : (!llvm.ptr<array<2 x i32>>, i32, i2) -> !llvm.ptr<i32>
  // CHECK-NEXT: llvm.load %[[GEP]] : !llvm.ptr<i32>
  %0 = hw.array_get %arg1[%arg0] : !hw.array<2xi32>, i1

  // CHECK-NEXT: %[[ZERO1:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[ONE4:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[ALLOCA1:.*]] = llvm.alloca %[[ONE4]] x !llvm.array<2 x i32> {alignment = 4 : i64} : (i32) -> !llvm.ptr<array<2 x i32>>
  // CHECK-NEXT: llvm.store %[[CAST0]], %[[ALLOCA1]] : !llvm.ptr<array<2 x i32>>
  // CHECK-NEXT: %[[ZEXT1:.*]] = llvm.zext %arg0 : i1 to i2
  // CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ALLOCA1]][%[[ZERO1]], %[[ZEXT1]]] : (!llvm.ptr<array<2 x i32>>, i32, i2) -> !llvm.ptr<i32>
  // CHECK-NEXT: %[[BITCAST:.*]] = llvm.bitcast %[[GEP1]] : !llvm.ptr<i32> to !llvm.ptr<array<1 x i32>>
  // CHECK-NEXT: llvm.load %[[LD:.*]] : !llvm.ptr<array<1 x i32>>
  %1 = hw.array_slice %arg1[%arg0] : (!hw.array<2xi32>) -> !hw.array<1xi32>

  // CHECK-NEXT: %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.array<4 x i32>
  // CHECK-NEXT: %[[E1:.*]] = llvm.extractvalue %[[CAST0]][0] : !llvm.array<2 x i32>
  // CHECK-NEXT: %[[I1:.*]] = llvm.insertvalue %[[E1]], %[[UNDEF]][0] : !llvm.array<4 x i32>
  // CHECK-NEXT: %[[E2:.*]] = llvm.extractvalue %[[CAST0]][1] : !llvm.array<2 x i32>
  // CHECK-NEXT: %[[I2:.*]] = llvm.insertvalue %[[E2]], %[[I1]][1] : !llvm.array<4 x i32>
  // CHECK-NEXT: %[[E3:.*]] = llvm.extractvalue %[[CAST0]][0] : !llvm.array<2 x i32>
  // CHECK-NEXT: %[[I3:.*]] = llvm.insertvalue %[[E3]], %[[I2]][2] : !llvm.array<4 x i32>
  // CHECK: %[[E4:.*]] = llvm.extractvalue %[[CAST0]][1] : !llvm.array<2 x i32>
  // CHECK-NEXT: llvm.insertvalue %[[E4]], %[[I3]][3] : !llvm.array<4 x i32>
  %2 = hw.array_concat %arg1, %arg1 : !hw.array<2xi32>, !hw.array<2xi32>


  // CHECK-NEXT: [[V6:%.*]] = llvm.mlir.undef : !llvm.array<4 x i32>
  // CHECK-NEXT: [[V7:%.*]] = llvm.insertvalue %arg5, [[V6]][0] : !llvm.array<4 x i32>
  // CHECK-NEXT: [[V8:%.*]] = llvm.insertvalue %arg4, [[V7]][1] : !llvm.array<4 x i32>
  // CHECK-NEXT: [[V9:%.*]] = llvm.insertvalue %arg3, [[V8]][2] : !llvm.array<4 x i32>
  // CHECK-NEXT: [[V10:%.*]] = llvm.insertvalue %arg2, [[V9]][3] : !llvm.array<4 x i32>
  %3 = hw.array_create %arg2, %arg3, %arg4, %arg5 : i32

  return
}

// CHECK: llvm.mlir.global internal constant @[[GLOB1:.+]](dense<[1, 0]> : tensor<2xi32>) {addr_space = 0 : i32} : !llvm.array<2 x i32>

// CHECK: llvm.mlir.global internal constant @[[GLOB2:.+]](dense<{{[[][[]}}3, 2], [1, 0{{[]][]]}}> : tensor<2x2xi32>) {addr_space = 0 : i32} : !llvm.array<2 x array<2 x i32>>

// CHECK: llvm.mlir.global internal @[[GLOB3:.+]]() {addr_space = 0 : i32} : !llvm.array<2 x struct<(i1, i32)>> {
// CHECK-NEXT:   [[V0:%.+]] = llvm.mlir.undef : !llvm.array<2 x struct<(i1, i32)>>
// CHECK-NEXT:   [[V1:%.+]] = llvm.mlir.undef : !llvm.struct<(i1, i32)>
// CHECK-NEXT:   [[V2:%.+]] = llvm.mlir.constant(false) : i1
// CHECK-NEXT:   [[V3:%.+]] = llvm.insertvalue [[V2]], [[V1]][0] : !llvm.struct<(i1, i32)>
// CHECK-NEXT:   [[V4:%.+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:   [[V5:%.+]] = llvm.insertvalue [[V4]], [[V3]][1] : !llvm.struct<(i1, i32)>
// CHECK-NEXT:   [[V6:%.+]] = llvm.insertvalue [[V5]], [[V0]][0] : !llvm.array<2 x struct<(i1, i32)>>
// CHECK-NEXT:   [[V7:%.+]] = llvm.mlir.undef : !llvm.struct<(i1, i32)>
// CHECK-NEXT:   [[V8:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-NEXT:   [[V9:%.+]] = llvm.insertvalue [[V8]], [[V7]][0] : !llvm.struct<(i1, i32)>
// CHECK-NEXT:   [[V10:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:   [[V11:%.+]] = llvm.insertvalue [[V10]], [[V9]][1] : !llvm.struct<(i1, i32)>
// CHECK-NEXT:   [[V12:%.+]] = llvm.insertvalue [[V11]], [[V6]][1] : !llvm.array<2 x struct<(i1, i32)>>
// CHECK-NEXT:   llvm.return [[V12]] : !llvm.array<2 x struct<(i1, i32)>>
// CHECK-NEXT: }

// CHECK: @convertConstArray
func.func @convertConstArray(%arg0 : i1) {
  // COM: Test: simple constant array converted to constant global
  // CHECK: %[[VAL_2:.*]] = llvm.mlir.addressof @[[GLOB1]] : !llvm.ptr<array<2 x i32>>
  // CHECK-NEXT: %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<array<2 x i32>>
  %0 = hw.aggregate_constant [0 : i32, 1 : i32] : !hw.array<2xi32>

  // COM: Test: when the array argument is already a load from a pointer,
  // COM: then don't allocate on the stack again but take that pointer directly as a shortcut
  // CHECK-NEXT: %[[VAL_4:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[VAL_5:.*]] = llvm.zext %arg0 : i1 to i2
  // CHECK-NEXT: %[[VAL_6:.*]] = llvm.getelementptr %[[VAL_2]][%[[VAL_4]], %[[VAL_5]]] : (!llvm.ptr<array<2 x i32>>, i32, i2) -> !llvm.ptr<i32>
  // CHECK-NEXT: %{{.+}} = llvm.load %[[VAL_6]] : !llvm.ptr<i32>
  %1 = hw.array_get %0[%arg0] : !hw.array<2xi32>, i1

  // COM: Test: nested constant array can also converted to a constant global
  // CHECK: %[[VAL_7:.*]] = llvm.mlir.addressof @[[GLOB2]] : !llvm.ptr<array<2 x array<2 x i32>>>
  // CHECK-NEXT: %{{.+}} = llvm.load %[[VAL_7]] : !llvm.ptr<array<2 x array<2 x i32>>>
  %2 = hw.aggregate_constant [[0 : i32, 1 : i32], [2 : i32, 3 : i32]] : !hw.array<2x!hw.array<2xi32>>

  // COM: the same constants only create one global (note: even if they are in different functions).
  // CHECK: %[[VAL_8:.*]] = llvm.mlir.addressof @[[GLOB2]] : !llvm.ptr<array<2 x array<2 x i32>>>
  // CHECK-NEXT: %{{.+}} = llvm.load %[[VAL_8]] : !llvm.ptr<array<2 x array<2 x i32>>>
  %3 = hw.aggregate_constant [[0 : i32, 1 : i32], [2 : i32, 3 : i32]] : !hw.array<2x!hw.array<2xi32>>

  // CHECK: %[[VAL_9:.+]] = llvm.mlir.addressof @[[GLOB3]] : !llvm.ptr<array<2 x struct<(i1, i32)>>>
  // CHECK-NEXT: {{%.+}} = llvm.load %[[VAL_9]] : !llvm.ptr<array<2 x struct<(i1, i32)>>>
  %4 = hw.aggregate_constant [[0 : i32, 1 : i1], [2 : i32, 0 : i1]] : !hw.array<2x!hw.struct<a: i32, b: i1>>

  return
}

// CHECK: llvm.mlir.global internal @[[GLOB4:.+]]() {addr_space = 0 : i32} : !llvm.struct<(i2, i32)> {
// CHECK-NEXT:    [[V0:%.+]] = llvm.mlir.undef : !llvm.struct<(i2, i32)>
// CHECK-NEXT:    [[V1:%.+]] = llvm.mlir.constant(1 : i2) : i2
// CHECK-NEXT:    [[V2:%.+]] = llvm.insertvalue [[V1]], [[V0]][0] : !llvm.struct<(i2, i32)>
// CHECK-NEXT:    [[V3:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    [[V4:%.+]] = llvm.insertvalue [[V3]], [[V2]][1] : !llvm.struct<(i2, i32)>
// CHECK-NEXT:    llvm.return [[V4]] : !llvm.struct<(i2, i32)>
// CHECK-NEXT:  }

// CHECK: @convertConstStruct
func.func @convertConstStruct() {
  // CHECK-NEXT: [[V0:%.+]] = llvm.mlir.addressof @[[GLOB4]] : !llvm.ptr<struct<(i2, i32)>>
  // CHECK-NEXT: [[V1:%.+]] = llvm.load [[V0]] : !llvm.ptr<struct<(i2, i32)>>
  %0 = hw.aggregate_constant [0 : i32, 1 : i2] : !hw.struct<a: i32, b: i2>
  // CHECK-NEXT: {{%.+}} = llvm.extractvalue [[V1]][1] : !llvm.struct<(i2, i32)>
  %1 = hw.struct_extract %0["a"] : !hw.struct<a: i32, b: i2>
  return
}

// CHECK-LABEL: @convertStruct
func.func @convertStruct(%arg0 : i32, %arg1: !hw.struct<foo: i32, bar: i8>, %arg2: !hw.struct<>, %arg3 : i1, %arg4 : i2) {
  // CHECK-NEXT: [[ARG1:%.+]] = builtin.unrealized_conversion_cast %arg1 : !hw.struct<foo: i32, bar: i8> to !llvm.struct<(i8, i32)>
  // CHECK-NEXT: llvm.extractvalue [[ARG1]][1] : !llvm.struct<(i8, i32)>
  %0 = hw.struct_extract %arg1["foo"] : !hw.struct<foo: i32, bar: i8>

  // CHECK: llvm.insertvalue %arg0, [[ARG1]][1] : !llvm.struct<(i8, i32)>
  %1 = hw.struct_inject %arg1["foo"], %arg0 : !hw.struct<foo: i32, bar: i8>

  // CHECK: llvm.extractvalue [[ARG1]][1] : !llvm.struct<(i8, i32)>
  // CHECK: llvm.extractvalue [[ARG1]][0] : !llvm.struct<(i8, i32)>
  %2:2 = hw.struct_explode %arg1 : !hw.struct<foo: i32, bar: i8>

  hw.struct_explode %arg2 : !hw.struct<>

  // CHECK-NEXT: [[V3:%.*]] = llvm.mlir.undef : !llvm.struct<(i32, i2, i1)>
  // CHECK-NEXT: [[V4:%.*]] = llvm.insertvalue %arg0, [[V3]][0] : !llvm.struct<(i32, i2, i1)>
  // CHECK-NEXT: [[V5:%.*]] = llvm.insertvalue %arg4, [[V4]][1] : !llvm.struct<(i32, i2, i1)>
  // CHECK-NEXT: [[V6:%.*]] = llvm.insertvalue %arg3, [[V5]][2] : !llvm.struct<(i32, i2, i1)>
  %3 = hw.struct_create (%arg3, %arg4, %arg0) : !hw.struct<foo: i1, bar: i2, baz: i32>

  return
}
