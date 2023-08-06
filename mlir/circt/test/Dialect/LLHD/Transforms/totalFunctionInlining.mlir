// RUN: circt-opt %s -inline="default-pipeline=''" -llhd-function-elimination | FileCheck %s

// This test checks the presence of inlining into entities and processes
// and their general structure after inlining. It also checks that the functions
// are deleted by the elimination pass.
// Note: Only functions which can be reduced to one basic block can be inlined
// into entities.

// CHECK-NOT: func
func.func @simple() -> i32 {
  %0 = hw.constant 5 : i32
  return %0 : i32
}

// CHECK-NOT: func
func.func @complex(%flag : i1) -> i32 {
  cf.cond_br %flag, ^bb1, ^bb2
^bb1:
  %0 = hw.constant 5 : i32
  return %0 : i32
^bb2:
  %1 = hw.constant 7 : i32
  return %1 : i32
}

// CHECK-LABEL: @check_entity_inline
llhd.entity @check_entity_inline() -> (%out : !llhd.sig<i32>) {
  // CHECK-NEXT: %{{.*}} = hw.constant
  // CHECK-NEXT: %{{.*}} = llhd.constant_time
  // CHECK-NEXT: llhd.drv
  // CHECK-NEXT: }
  %1 = func.call @simple() : () -> i32
  %time = llhd.constant_time #llhd.time<1ns, 0d, 0e>
  llhd.drv %out, %1 after %time : !llhd.sig<i32>
}

// CHECK-LABEL: @check_proc_inline
llhd.proc @check_proc_inline(%arg : !llhd.sig<i1>) -> (%out : !llhd.sig<i32>) {
  // CHECK-NEXT: %[[PRB:.*]] = llhd.prb
  // CHECK-NEXT: cf.cond_br %[[PRB]], ^[[BB1:.*]], ^[[BB2:.*]]
  // CHECK-NEXT: ^[[BB1]]:
  // CHECK-NEXT: %[[C0:.*]] = hw.constant
  // CHECK-NEXT: cf.br ^[[BB3:.*]](%[[C0]] : i32)
  // CHECK-NEXT: ^[[BB2]]:
  // CHECK-NEXT: %[[C1:.*]] = hw.constant
  // CHECK-NEXT: cf.br ^[[BB3]](%[[C1]] : i32)
  // CHECK-NEXT: ^[[BB3]](%[[A:.*]]: i32):
  // CHECK-NEXT: %[[C2:.*]] = llhd.constant_time
  // CHECK-NEXT: llhd.drv %{{.*}}, %[[A]] after %[[C2]] : !llhd.sig<i32>
  // CHECK-NEXT: llhd.halt
  // CHECK-NEXT: }
  %0 = llhd.prb %arg : !llhd.sig<i1>
  %1 = func.call @complex(%0) : (i1) -> i32
  %time = llhd.constant_time #llhd.time<1ns, 0d, 0e>
  llhd.drv %out, %1 after %time : !llhd.sig<i32>
  llhd.halt
}
