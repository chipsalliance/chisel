// RUN: circt-opt %s --allow-unregistered-dialect -verify-diagnostics | circt-opt --allow-unregistered-dialect -verify-diagnostics | FileCheck %s

// CHECK-LABEL: msft.physical_region @region1
msft.physical_region @region1, [
  // CHECK-SAME: #msft.physical_bounds<x: [0, 10], y: [0, 10]>
  #msft.physical_bounds<x: [0, 10], y: [0, 10]>,
  // CHECK-SAME: #msft.physical_bounds<x: [20, 30], y: [20, 30]>
  #msft.physical_bounds<x: [20, 30], y: [20, 30]>]

// CHECK: #msft.location_vec<i3, [*, <1, 2, 3>, #msft.physloc<DSP, 4, 5, 6>]>
"dummy.op" () {"vec" = #msft.location_vec<i3, [*, <1, 2, 3>, #msft.physloc<DSP, 4, 5, 6>]>} : () -> ()

// CHECK: #msft.appid<"foo"[4]>
"dummy.op" () {"appid" = #msft.appid<"foo"[4]> } : () -> ()
