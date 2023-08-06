// RUN: circt-opt %s | circt-opt | FileCheck %s

// check inputs and outputs, usage
// CHECK: llhd.entity @foo (%[[ARG0:.*]] : !llhd.sig<i64>, %[[ARG1:.*]] : !llhd.sig<i64>) -> (%[[OUT0:.*]] : !llhd.sig<i64>) {
"llhd.entity"() ({
^body(%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i64>, %out0 : !llhd.sig<i64>):
  // CHECK-NEXT: %[[C0:.*]] = hw.constant 1
  %0 = hw.constant 1 : i64
  // CHECK-NEXT: %[[P0:.*]] = llhd.prb %[[ARG0]]
  %1 = llhd.prb %arg0 : !llhd.sig<i64>
// CHECK-NEXT: }
}) {sym_name="foo", ins=2, function_type=(!llhd.sig<i64>, !llhd.sig<i64>, !llhd.sig<i64>)->()} : () -> ()

// check 0 inputs, empty body
// CHECK-NEXT: llhd.entity @bar () -> (%{{.*}} : !llhd.sig<i64>) {
"llhd.entity"() ({
^body(%0 : !llhd.sig<i64>):
// CHECK-NEXT: }
}) {sym_name="bar", ins=0, function_type=(!llhd.sig<i64>)->()} : () -> ()

// check 0 outputs, empty body
// CHECK-NEXT: llhd.entity @baz (%{{.*}} : !llhd.sig<i64>) -> () {
"llhd.entity"() ({
^body(%arg0 : !llhd.sig<i64>):
// CHECK-NEXT: }
}) {sym_name="baz", ins=1, function_type=(!llhd.sig<i64>)->()} : () -> ()

//check 0 arguments, empty body
// CHECK-NEXT: llhd.entity @out_of_names () -> () {
"llhd.entity"() ({
^body:
// CHECK-NEXT : }
}) {sym_name="out_of_names", ins=0, function_type=()->()} : () -> ()
