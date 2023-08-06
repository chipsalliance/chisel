// RUN: circt-opt %s | circt-opt | FileCheck %s

// Test Overview:
//   * 0 observed signals, no time, successor without arguments
//   * 0 observed signals, with time, sucessor with arguments
//   * 2 observed signals, no time, successor with arguments
//   * 2 observed signals, with time, successor with arguments

// CHECK-LABEL: @check_wait_0
llhd.proc @check_wait_0 () -> () {
  // CHECK: llhd.wait ^[[BB:.*]]
  "llhd.wait"() [^bb1] {operand_segment_sizes=array<i32: 0,0,0>} : () -> ()
  // CHECK-NEXT: ^[[BB]]
^bb1:
  llhd.halt
}

// CHECK-LABEL: @check_wait_1
llhd.proc @check_wait_1 () -> () {
  // CHECK-NEXT: %[[TIME:.*]] = llhd.constant_time
  %time = llhd.constant_time #llhd.time<0ns, 0d, 0e>
  // CHECK-NEXT: llhd.wait for %[[TIME]], ^[[BB:.*]](%[[TIME]] : !llhd.time)
  "llhd.wait"(%time, %time) [^bb1] {operand_segment_sizes=array<i32: 0,1,1>} : (!llhd.time, !llhd.time) -> ()
  // CHECK-NEXT: ^[[BB]](%[[T:.*]]: !llhd.time):
^bb1(%t: !llhd.time):
  llhd.halt
}

// CHECK: llhd.proc @check_wait_2(%[[ARG0:.*]] : !llhd.sig<i64>, %[[ARG1:.*]] : !llhd.sig<i1>) -> () {
llhd.proc @check_wait_2 (%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i1>) -> () {
  // CHECK-NEXT: llhd.wait (%[[ARG0]], %[[ARG1]] : !llhd.sig<i64>, !llhd.sig<i1>), ^[[BB:.*]](%[[ARG1]] : !llhd.sig<i1>)
  "llhd.wait"(%arg0, %arg1, %arg1) [^bb1] {operand_segment_sizes=array<i32: 2,0,1>} : (!llhd.sig<i64>, !llhd.sig<i1>, !llhd.sig<i1>) -> ()
  // CHECK: ^[[BB]](%[[A:.*]]: !llhd.sig<i1>):
^bb1(%a: !llhd.sig<i1>):
  llhd.halt
}

// CHECK: llhd.proc @check_wait_3(%[[ARG0:.*]] : !llhd.sig<i64>, %[[ARG1:.*]] : !llhd.sig<i1>) -> () {
llhd.proc @check_wait_3 (%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i1>) -> () {
  // CHECK-NEXT: %[[TIME:.*]] = llhd.constant_time
  %time = llhd.constant_time #llhd.time<0ns, 0d, 0e>
  // CHECK-NEXT: llhd.wait for %[[TIME]], (%[[ARG0]], %[[ARG1]] : !llhd.sig<i64>, !llhd.sig<i1>), ^[[BB:.*]](%[[ARG1]], %[[ARG0]] : !llhd.sig<i1>, !llhd.sig<i64>)
  "llhd.wait"(%arg0, %arg1, %time, %arg1, %arg0) [^bb1] {operand_segment_sizes=array<i32: 2,1,2>} : (!llhd.sig<i64>, !llhd.sig<i1>, !llhd.time, !llhd.sig<i1>, !llhd.sig<i64>) -> ()
  // CHECK: ^[[BB]](%[[A:.*]]: !llhd.sig<i1>, %[[B:.*]]: !llhd.sig<i64>):
^bb1(%a: !llhd.sig<i1>, %b: !llhd.sig<i64>):
  llhd.halt
}
