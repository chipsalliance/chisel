// RUN: circt-opt %s -mlir-print-op-generic -split-input-file -verify-diagnostics | circt-opt | circt-opt | FileCheck %s

// CHECK-LABEL: checkSigInst
llhd.entity @checkSigInst () -> () {
  // CHECK: %[[CI1:.*]] = hw.constant
  %cI1 = hw.constant 0 : i1
  // CHECK-NEXT: %{{.*}} = llhd.sig "sigI1" %[[CI1]] : i1
  %sigI1 = llhd.sig "sigI1" %cI1 : i1
  // CHECK-NEXT: %[[CI64:.*]] = hw.constant
  %cI64 = hw.constant 0 : i64
  // CHECK-NEXT: %{{.*}} = llhd.sig "sigI64" %[[CI64]] : i64
  %sigI64 = llhd.sig "sigI64" %cI64 : i64

  // CHECK-NEXT: %[[TUP:.*]] = hw.struct_create
  %tup = hw.struct_create (%cI1, %cI64) : !hw.struct<foo: i1, bar: i64>
  // CHECK-NEXT: %{{.*}} = llhd.sig "sigTup" %[[TUP]] : !hw.struct<foo: i1, bar: i64>
  %sigTup = llhd.sig "sigTup" %tup : !hw.struct<foo: i1, bar: i64>

  // CHECK-NEXT: %[[ARRAY:.*]] = hw.array_create
  %array = hw.array_create %cI1, %cI1 : i1
  // CHECK-NEXT: %{{.*}} = llhd.sig "sigArray" %[[ARRAY]] : !hw.array<2xi1>
  %sigArray = llhd.sig "sigArray" %array : !hw.array<2xi1>
}

// CHECK-LABEL: checkPrb
func.func @checkPrb(%arg0 : !llhd.sig<i1>, %arg1 : !llhd.sig<i64>, %arg2 : !llhd.sig<!hw.array<3xi8>>, %arg3 : !llhd.sig<!hw.struct<foo: i1, bar: i2, baz: i4>>) {
  // CHECK: %{{.*}} = llhd.prb %arg0 : !llhd.sig<i1>
  %0 = llhd.prb %arg0 : !llhd.sig<i1>
  // CHECK-NEXT: %{{.*}} = llhd.prb %arg1 : !llhd.sig<i64>
  %1 = llhd.prb %arg1 : !llhd.sig<i64>
  // CHECK-NEXT: %{{.*}} = llhd.prb %arg2 : !llhd.sig<!hw.array<3xi8>>
  %2 = llhd.prb %arg2 : !llhd.sig<!hw.array<3xi8>>
  // CHECK-NEXT: %{{.*}} = llhd.prb %arg3 : !llhd.sig<!hw.struct<foo: i1, bar: i2, baz: i4>>
  %3 = llhd.prb %arg3 : !llhd.sig<!hw.struct<foo: i1, bar: i2, baz: i4>>

  return
}

// CHECK-LABEL: checkOutput
func.func @checkOutput(%arg0: i32, %arg1: !llhd.time) {
  // CHECK-NEXT: %{{.+}} = llhd.output %arg0 after %arg1 : i32
  %0 = llhd.output %arg0 after %arg1 : i32
  // CHECK-NEXT: %{{.+}} = llhd.output "sigName" %arg0 after %arg1 : i32
  %1 = llhd.output "sigName" %arg0 after %arg1 : i32

  return
}

// CHECK-LABEL: checkDrv
func.func @checkDrv(%arg0 : !llhd.sig<i1>, %arg1 : !llhd.sig<i64>, %arg2 : i1,
    %arg3 : i64, %arg4 : !llhd.time, %arg5 : !llhd.sig<!hw.array<3xi8>>,
    %arg6 : !llhd.sig<!hw.struct<foo: i1, bar: i2, baz: i4>>,
    %arg7 : !hw.array<3xi8>, %arg8 : !hw.struct<foo: i1, bar: i2, baz: i4>) {

  // CHECK-NEXT: llhd.drv %arg0, %arg2 after %arg4 : !llhd.sig<i1>
  llhd.drv %arg0, %arg2 after %arg4 : !llhd.sig<i1>
  // CHECK-NEXT: llhd.drv %arg1, %arg3 after %arg4 : !llhd.sig<i64>
  llhd.drv %arg1, %arg3 after %arg4 : !llhd.sig<i64>
  // CHECK-NEXT: llhd.drv %arg1, %arg3 after %arg4 if %arg2 : !llhd.sig<i64>
  llhd.drv %arg1, %arg3 after %arg4 if %arg2 : !llhd.sig<i64>
  // CHECK-NEXT: llhd.drv %arg5, %arg7 after %arg4 : !llhd.sig<!hw.array<3xi8>>
  llhd.drv %arg5, %arg7 after %arg4 : !llhd.sig<!hw.array<3xi8>>
  // CHECK-NEXT: llhd.drv %arg6, %arg8 after %arg4 : !llhd.sig<!hw.struct<foo: i1, bar: i2, baz: i4>>
  llhd.drv %arg6, %arg8 after %arg4 : !llhd.sig<!hw.struct<foo: i1, bar: i2, baz: i4>>

  return
}
