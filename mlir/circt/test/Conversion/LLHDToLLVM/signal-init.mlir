// RUN: circt-opt %s --convert-llhd-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @llhd_init
llhd.entity @root() -> () {
  llhd.inst "inst0" @initSig () -> () : () -> ()
  llhd.inst "inst1" @initPartiallyLowered () -> () : () -> ()
  llhd.inst "inst2" @initMultipleResults () -> () : () -> ()
}

llhd.entity @initSig () -> () {
  %1 = hw.aggregate_constant [ 0 : i1, 1 : i1] : !hw.array<2xi1>
  %2 = hw.aggregate_constant [ 0 : i1, 1 : i5] : !hw.struct<f1: i1, f2: i5>
  // CHECK: [[V0:%.+]] = llvm.mlir.addressof @{{.+}} : !llvm.ptr<array<2 x i1>>
  // CHECK: [[V1:%.+]] = llvm.load [[V0]] : !llvm.ptr<array<2 x i1>>
  // CHECK: llvm.store [[V1]], {{%.+}} : !llvm.ptr<array<2 x i1>>
  %3 = llhd.sig "sig1" %1 : !hw.array<2xi1>
  // CHECK: [[V2:%.+]] = llvm.mlir.addressof @{{.+}} : !llvm.ptr<struct<(i5, i1)>>
  // CHECK: [[V3:%.+]] = llvm.load [[V2]] : !llvm.ptr<struct<(i5, i1)>>
  // CHECK: llvm.store [[V3]], {{%.+}} : !llvm.ptr<struct<(i5, i1)>>
  %4 = llhd.sig "sig2" %2 : !hw.struct<f1: i1, f2: i5>
}

llhd.entity @initPartiallyLowered () -> () {
  %0 = llvm.mlir.constant(false) : i1
  %1 = llvm.mlir.undef : !llvm.array<2 x i1>
  %2 = llvm.insertvalue %0, %1[0] : !llvm.array<2 x i1>
  %3 = llvm.insertvalue %0, %2[1] : !llvm.array<2 x i1>
  %4 = builtin.unrealized_conversion_cast %3 : !llvm.array<2 x i1> to !hw.array<2xi1>
  %5 = llhd.sig "sig" %4 : !hw.array<2xi1>
}

func.func @getInitValue() -> (i32, i32, i32) {
  %0 = hw.constant 0 : i32
  return %0, %0, %0 : i32, i32, i32
}

// CHECK: [[RETURN:%.+]] = llvm.call @getInitValue() : () -> !llvm.struct<(i32, i32, i32)>
// CHECK: [[E1:%.+]] = llvm.extractvalue [[RETURN]][0] : !llvm.struct<(i32, i32, i32)>
// CHECK: [[E2:%.+]] = llvm.extractvalue [[RETURN]][1] : !llvm.struct<(i32, i32, i32)>
// CHECK: [[E3:%.+]] = llvm.extractvalue [[RETURN]][2] : !llvm.struct<(i32, i32, i32)>
// CHECK: llvm.store [[E2]], {{%.+}} : !llvm.ptr<i32>
llhd.entity @initMultipleResults () -> () {
  %0, %1, %2 = func.call @getInitValue() : () -> (i32, i32, i32)
  %3 = llhd.sig "sig" %1 : i32
}

// CHECK: }
