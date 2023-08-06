// RUN: circt-opt %s -allow-unregistered-dialect | FileCheck %s

func.func @test_time_type() {
  // CHECK: %[[CONST:.*]] = "time_result"() : () -> !llhd.time
  %0 = "time_result"() : () -> !llhd.time
  // CHECK-NEXT: "time_const_arg"(%[[CONST]]) : (!llhd.time) -> ()
  "time_const_arg"(%0) : (!llhd.time) -> ()
  return
}

func.func @test_time_attr() {
  "time_attr"() {
    // CHECK: time0 = #llhd.time<1ns, 0d, 0e>
    time0 = #llhd.time<1ns, 0d, 0e> : !llhd.time,
    // CHECK-SAME: time1 = #llhd.time<1ns, 2d, 0e>
    time1 = #llhd.time<1ns, 2d, 0e> : !llhd.time,
    // CHECK-SAME: time2 = #llhd.time<1ns, 2d, 3e>
    time2 = #llhd.time<1ns, 2d, 3e> : !llhd.time,
    // CHECK-SAME: time3 = #llhd.time<1ns, 0d, 3e>
    time3 = #llhd.time<1ns, 0d, 3e> : !llhd.time
  } : () -> ()
}
