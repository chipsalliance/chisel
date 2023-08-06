// RUN: circt-opt %s | circt-opt | FileCheck %s

// no inputs and outputs
// CHECK: llhd.proc @empty() -> () {
"llhd.proc"() ({
  // CHECK: llhd.halt
  // CHECK-NEXT: }
  "llhd.halt"() {} : () -> ()
}) {sym_name="empty", ins=0, function_type=()->()} : () -> ()

// two inputs, one output
// CHECK-NEXT: llhd.proc @inputandoutput(%{{.*}} : !llhd.sig<i64>, %{{.*}} : !llhd.sig<i64>) -> (%{{.*}} : !llhd.sig<i64>) {
"llhd.proc"() ({
  // CHECK-NEXT: llhd.halt
  // CHECK-NEXT: }
^body(%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i64>, %out0 : !llhd.sig<i64>):
  "llhd.halt"() {} : () -> ()
}) {sym_name="inputandoutput", ins=2, function_type=(!llhd.sig<i64>, !llhd.sig<i64>, !llhd.sig<i64>)->()} : () -> ()

// zero inputs, one output
// CHECK-NEXT: llhd.proc @output() -> (%{{.*}} : !llhd.sig<i64>) {
"llhd.proc"() ({
  // CHECK-NEXT: llhd.halt
  // CHECK-NEXT: }
^body(%0 : !llhd.sig<i64>):
  "llhd.halt"() {} : () -> ()
}) {sym_name="output", ins=0, function_type=(!llhd.sig<i64>)->()} : () -> ()

// one input, zero outputs
// CHECK-NEXT: llhd.proc @input(%{{.*}} : !llhd.sig<i64>) -> () {
"llhd.proc"() ({
  // CHECK-NEXT: llhd.halt
  // CHECK-NEXT: }
^body(%arg0 : !llhd.sig<i64>):
  "llhd.halt"() {} : () -> ()
}) {sym_name="input", ins=1, function_type=(!llhd.sig<i64>)->()} : () -> ()
