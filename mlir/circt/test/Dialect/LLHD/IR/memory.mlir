// RUN: circt-opt %s -mlir-print-op-generic -split-input-file -verify-diagnostics | circt-opt | circt-opt | FileCheck %s

// CHECK-LABEL: @check_var
// CHECK-SAME: %[[INT:.*]]: i32
// CHECK-SAME: %[[ARRAY:.*]]: !hw.array<3xi1>
// CHECK-SAME: %[[TUP:.*]]: !hw.struct<foo: i1, bar: i2, baz: i3>
func.func @check_var(%int : i32, %array : !hw.array<3xi1>, %tup : !hw.struct<foo: i1, bar: i2, baz: i3>) {
  // CHECK-NEXT: %{{.*}} = llhd.var %[[INT]] : i32
  %0 = llhd.var %int : i32
  // CHECK-NEXT: %{{.*}} = llhd.var %[[ARRAY]] : !hw.array<3xi1>
  %1 = llhd.var %array : !hw.array<3xi1>
  // CHECK-NEXT: %{{.*}} = llhd.var %[[TUP]] : !hw.struct<foo: i1, bar: i2, baz: i3>
  %2 = llhd.var %tup : !hw.struct<foo: i1, bar: i2, baz: i3>

  return
}

// CHECK-LABEL: @check_load
// CHECK-SAME: %[[INT:.*]]: !llhd.ptr<i32>
// CHECK-SAME: %[[ARRAY:.*]]: !llhd.ptr<!hw.array<3xi1>>
// CHECK-SAME: %[[TUP:.*]]: !llhd.ptr<!hw.struct<foo: i1, bar: i2, baz: i3>>
func.func @check_load(%int : !llhd.ptr<i32>, %array : !llhd.ptr<!hw.array<3xi1>>, %tup : !llhd.ptr<!hw.struct<foo: i1, bar: i2, baz: i3>>) {
  // CHECK-NEXT: %{{.*}} = llhd.load %[[INT]] : !llhd.ptr<i32>
  %0 = llhd.load %int : !llhd.ptr<i32>
  // CHECK-NEXT: %{{.*}} = llhd.load %[[ARRAY]] : !llhd.ptr<!hw.array<3xi1>>
  %1 = llhd.load %array : !llhd.ptr<!hw.array<3xi1>>
  // CHECK-NEXT: %{{.*}} = llhd.load %[[TUP]] : !llhd.ptr<!hw.struct<foo: i1, bar: i2, baz: i3>>
  %2 = llhd.load %tup : !llhd.ptr<!hw.struct<foo: i1, bar: i2, baz: i3>>

  return

}
// CHECK-LABEL: @check_store
// CHECK-SAME: %[[INT:.*]]: !llhd.ptr<i32>
// CHECK-SAME: %[[INTC:.*]]: i32
// CHECK-SAME: %[[ARRAY:.*]]: !llhd.ptr<!hw.array<3xi1>>
// CHECK-SAME: %[[ARRAYC:.*]]: !hw.array<3xi1>
// CHECK-SAME: %[[TUP:.*]]: !llhd.ptr<!hw.struct<foo: i1, bar: i2, baz: i3>>
// CHECK-SAME: %[[TUPC:.*]]: !hw.struct<foo: i1, bar: i2, baz: i3>
func.func @check_store(%int : !llhd.ptr<i32>, %intC : i32 , %array : !llhd.ptr<!hw.array<3xi1>>, %arrayC : !hw.array<3xi1>, %tup : !llhd.ptr<!hw.struct<foo: i1, bar: i2, baz: i3>>, %tupC : !hw.struct<foo: i1, bar: i2, baz: i3>) {
  // CHECK-NEXT: llhd.store %[[INT]], %[[INTC]] : !llhd.ptr<i32>
  llhd.store %int, %intC : !llhd.ptr<i32>
  // CHECK-NEXT: llhd.store %[[ARRAY]], %[[ARRAYC]] : !llhd.ptr<!hw.array<3xi1>>
  llhd.store %array, %arrayC : !llhd.ptr<!hw.array<3xi1>>
  // CHECK-NEXT: llhd.store %[[TUP]], %[[TUPC]] : !llhd.ptr<!hw.struct<foo: i1, bar: i2, baz: i3>>
  llhd.store %tup, %tupC : !llhd.ptr<!hw.struct<foo: i1, bar: i2, baz: i3>>

  return
}
