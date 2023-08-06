// RUN: circt-opt %s -canonicalize='top-down=true region-simplify=true' | FileCheck %s

// CHECK-LABEL: @sigExtractOp
func.func @sigExtractOp(%arg0 : !llhd.sig<i32>, %arg1: i5) -> (!llhd.sig<i32>, !llhd.sig<i32>) {
  %zero = hw.constant 0 : i5

  // CHECK: %[[EXT:.*]] = llhd.sig.extract %arg0 from %arg1 : (!llhd.sig<i32>) -> !llhd.sig<i32>
  %0 = llhd.sig.extract %arg0 from %arg1 : (!llhd.sig<i32>) -> !llhd.sig<i32>

  %1 = llhd.sig.extract %arg0 from %zero : (!llhd.sig<i32>) -> !llhd.sig<i32>

  // CHECK-NEXT: return %[[EXT]], %arg0 : !llhd.sig<i32>, !llhd.sig<i32>
  return %0, %1 : !llhd.sig<i32>, !llhd.sig<i32>
}

// CHECK-LABEL: @sigArraySlice
func.func @sigArraySliceOp(%arg0: !llhd.sig<!hw.array<30xi32>>, %arg1: i5)
    -> (!llhd.sig<!hw.array<30xi32>>, !llhd.sig<!hw.array<30xi32>>, !llhd.sig<!hw.array<20xi32>>, !llhd.sig<!hw.array<3xi32>>) {
  %zero = hw.constant 0 : i5

  // CHECK-NEXT: %c-13_i5 = hw.constant -13 : i5
  // CHECK-NEXT: hw.constant
  %a = hw.constant 3 : i5
  %b = hw.constant 16 : i5

  // CHECK: %[[EXT:.*]] = llhd.sig.array_slice %arg0 at %arg1 : (!llhd.sig<!hw.array<30xi32>>) -> !llhd.sig<!hw.array<30xi32>>
  %ext = llhd.sig.array_slice %arg0 at %arg1 : (!llhd.sig<!hw.array<30xi32>>) -> !llhd.sig<!hw.array<30xi32>>

  %identity = llhd.sig.array_slice %arg0 at %zero : (!llhd.sig<!hw.array<30xi32>>) -> !llhd.sig<!hw.array<30xi32>>

  // CHECK-NEXT: %[[RES1:.*]] = llhd.sig.array_slice
  // CHECK-NEXT: %[[RES2:.*]] = llhd.sig.array_slice %arg0 at %c-13_i5 : (!llhd.sig<!hw.array<30xi32>>) -> !llhd.sig<!hw.array<3xi32>>
  %1 = llhd.sig.array_slice %arg0 at %a : (!llhd.sig<!hw.array<30xi32>>) -> !llhd.sig<!hw.array<20xi32>>
  %2 = llhd.sig.array_slice %1 at %b : (!llhd.sig<!hw.array<20xi32>>) -> !llhd.sig<!hw.array<3xi32>>

  // CHECK-NEXT: return %[[EXT]], %arg0, %[[RES1]], %[[RES2]] : !llhd.sig<!hw.array<30xi32>>, !llhd.sig<!hw.array<30xi32>>, !llhd.sig<!hw.array<20xi32>>, !llhd.sig<!hw.array<3xi32>>
  return %ext, %identity, %1, %2 : !llhd.sig<!hw.array<30xi32>>, !llhd.sig<!hw.array<30xi32>>, !llhd.sig<!hw.array<20xi32>>, !llhd.sig<!hw.array<3xi32>>
}
