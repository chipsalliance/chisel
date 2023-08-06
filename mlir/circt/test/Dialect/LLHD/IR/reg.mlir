// RUN: circt-opt %s -split-input-file -verify-diagnostics | circt-opt | FileCheck %s

// CHECK-LABEL: @check_reg
// CHECK-SAME: %[[IN64:.*]] : !llhd.sig<i64>
llhd.entity @check_reg (%in64 : !llhd.sig<i64>) -> () {
  // CHECK: %[[C1:.*]] = hw.constant
  %c1 = hw.constant 0 : i1
  // CHECK-NEXT: %[[C64:.*]] = hw.constant
  %c64 = hw.constant 0 : i64
  // CHECK-NEXT: %[[TIME:.*]] = llhd.constant_time
  %time = llhd.constant_time #llhd.time<1ns, 0d, 0e>
  // one trigger with optional gate
  // CHECK-NEXT: llhd.reg %[[IN64]], (%[[C64]], "low" %[[C1]] after %[[TIME]] if %[[C1]] : i64) : !llhd.sig<i64>
  "llhd.reg"(%in64, %c64, %c1, %time, %c1) {modes=[0], gateMask=[1], operand_segment_sizes=array<i32: 1,1,1,1,1>} : (!llhd.sig<i64>, i64, i1, !llhd.time, i1) -> ()
  // two triggers with optional gates
  // CHECK-NEXT: llhd.reg %[[IN64]], (%[[C64]], "low" %[[C1]] after %[[TIME]] if %[[C1]] : i64), (%[[IN64]], "high" %[[C1]] after %[[TIME]] if %[[C1]] : !llhd.sig<i64>) : !llhd.sig<i64>
  "llhd.reg"(%in64, %c64, %in64, %c1, %c1, %time, %time, %c1, %c1) {modes=[0,1], gateMask=[1,2], operand_segment_sizes=array<i32: 1,2,2,2,2>} : (!llhd.sig<i64>, i64, !llhd.sig<i64>, i1, i1, !llhd.time, !llhd.time, i1, i1) -> ()
  // two triggers with only one optional gate
  // CHECK-NEXT: llhd.reg %[[IN64]], (%[[C64]], "low" %[[C1]] after %[[TIME]] : i64), (%[[IN64]], "high" %[[C1]] after %[[TIME]] if %[[C1]] : !llhd.sig<i64>) : !llhd.sig<i64>
  "llhd.reg"(%in64, %c64, %in64, %c1, %c1, %time, %time, %c1) {modes=[0,1], gateMask=[0,1], operand_segment_sizes=array<i32: 1,2,2,2,1>} : (!llhd.sig<i64>, i64, !llhd.sig<i64>, i1, i1, !llhd.time, !llhd.time, i1) -> ()
}

// TODO: add verification tests in reg-errors.mlir (expected-error tests)
