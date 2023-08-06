// RUN: circt-opt -pass-pipeline="builtin.module(func.func(dc-dematerialize-forks-sinks))" %s | FileCheck %s

// CHECK-LABEL:   func.func @testFork(
// CHECK-SAME:                        %[[VAL_0:.*]]: !dc.token) -> (!dc.token, !dc.token, !dc.token) {
// CHECK:           return %[[VAL_0]], %[[VAL_0]], %[[VAL_0]] : !dc.token, !dc.token, !dc.token
// CHECK:         }
func.func @testFork(%arg0: !dc.token) -> (!dc.token, !dc.token, !dc.token) {
  %0:3 = dc.fork [3] %arg0 
  return %0#0, %0#1, %0#2 : !dc.token, !dc.token, !dc.token
}

// CHECK-LABEL:   func.func @testSink(
// CHECK-SAME:                        %[[VAL_0:.*]]: !dc.value<i1>) -> !dc.token {
// CHECK:           %[[VAL_1:.*]], %[[VAL_2:.*]] = dc.branch %[[VAL_0]]
// CHECK:           return %[[VAL_1]] : !dc.token
// CHECK:         }
func.func @testSink(%arg0: !dc.value<i1>) -> !dc.token {
  %trueToken, %falseToken = dc.branch %arg0
  dc.sink %falseToken
  return %trueToken : !dc.token
}
