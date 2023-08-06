// RUN: circt-opt %s -cse | FileCheck %s

// CHECK-LABEL: @check_mem_dce
// CHECK-SAME: %[[INT:.*]]: i32,
// CHECK-SAME: %[[INT2:.*]]: i32
func.func @check_mem_dce(%int : i32, %int2 : i32) -> i32 {
  // CHECK-NEXT: %[[VAR:.*]] = llhd.var %[[INT]] : i32
  %0 = llhd.var %int : i32
  %1 = llhd.var %int2 : i32

  // CHECK-NEXT: llhd.store %[[VAR]], %[[INT2]] : !llhd.ptr<i32>
  llhd.store %0, %int2 : !llhd.ptr<i32>
  %l1 = llhd.load %0 : !llhd.ptr<i32>
  // CHECK-NEXT: llhd.store %[[VAR]], %[[INT]] : !llhd.ptr<i32>
  llhd.store %0, %int : !llhd.ptr<i32>
  // CHECK-NEXT: %[[LOAD:.*]] = llhd.load %[[VAR]] : !llhd.ptr<i32>
  %l2 = llhd.load %0 : !llhd.ptr<i32>

  // CHECK-NEXT: return %[[LOAD]] : i32
  return %l2 : i32
}

// CHECK-LABEL: @check_mem_cse
// CHECK-SAME: %[[INT:.*]]: i32,
// CHECK-SAME: %[[INT2:.*]]: i32
func.func @check_mem_cse(%int : i32, %int2 : i32) {
  // CHECK-NEXT: %[[VAR1:.*]] = llhd.var %[[INT]] : i32
  %0 = llhd.var %int : i32
  // CHECK-NEXT: %[[VAR2:.*]] = llhd.var %[[INT2]] : i32
  %1 = llhd.var %int2 : i32
  // CHECK-NEXT: llhd.store %[[VAR1]], %[[INT2]] : !llhd.ptr<i32>
  llhd.store %0, %int2 : !llhd.ptr<i32>
  // CHECK-NEXT: llhd.store %[[VAR2]], %[[INT]] : !llhd.ptr<i32>
  llhd.store %1, %int : !llhd.ptr<i32>

  return
}
