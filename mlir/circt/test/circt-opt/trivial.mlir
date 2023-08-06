// RUN: circt-opt %s -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func @simpleCFG(%{{.*}}: i32, %{{.*}}: f32) -> i1 {
func.func @simpleCFG(%arg0: i32, %f: f32) -> i1 {
  // CHECK: %{{.*}} = "foo"() : () -> i64
  %1 = "foo"() : ()->i64
  // CHECK: "bar"(%{{.*}}) : (i64) -> (i1, i1, i1)
  %2:3 = "bar"(%1) : (i64) -> (i1,i1,i1)
  // CHECK: return %{{.*}}#1
  return %2#1 : i1
// CHECK: }
}
