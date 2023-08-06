// RUN: circt-opt %s -mlir-print-op-generic -split-input-file -verify-diagnostics | circt-opt | circt-opt | FileCheck %s

// CHECK-LABEL: @sigExtract
func.func @sigExtract (%arg0 : !llhd.sig<i32>, %arg1: i5) -> () {
  // CHECK-NEXT: %{{.*}} = llhd.sig.extract %arg0 from %arg1 : (!llhd.sig<i32>) -> !llhd.sig<i5>
  %1 = llhd.sig.extract %arg0 from %arg1 : (!llhd.sig<i32>) -> !llhd.sig<i5>

  return
}

// CHECK-LABEL: @sigArray
func.func @sigArray (%arg0 : !llhd.sig<!hw.array<5xi1>>, %arg1: i3) -> () {
  // CHECK-NEXT: %{{.*}} = llhd.sig.array_slice %arg0 at %arg1 : (!llhd.sig<!hw.array<5xi1>>) -> !llhd.sig<!hw.array<3xi1>>
  %0 = llhd.sig.array_slice %arg0 at %arg1 : (!llhd.sig<!hw.array<5xi1>>) -> !llhd.sig<!hw.array<3xi1>>
  // CHECK-NEXT: %{{.*}} = llhd.sig.array_get %arg0[%arg1] : !llhd.sig<!hw.array<5xi1>>
  %1 = llhd.sig.array_get %arg0[%arg1] : !llhd.sig<!hw.array<5xi1>>

  return
}

// CHECK-LABEL: @sigStructExtract
func.func @sigStructExtract(%arg0 : !llhd.sig<!hw.struct<foo: i1, bar: i2, baz: i3>>) {
  // CHECK-NEXT: %{{.*}} = llhd.sig.struct_extract %arg0["foo"] : !llhd.sig<!hw.struct<foo: i1, bar: i2, baz: i3>>
  %0 = llhd.sig.struct_extract %arg0["foo"] : !llhd.sig<!hw.struct<foo: i1, bar: i2, baz: i3>>
  // CHECK-NEXT: %{{.*}} = llhd.sig.struct_extract %arg0["baz"] : !llhd.sig<!hw.struct<foo: i1, bar: i2, baz: i3>>
  %1 = llhd.sig.struct_extract %arg0["baz"] : !llhd.sig<!hw.struct<foo: i1, bar: i2, baz: i3>>

  return
}
