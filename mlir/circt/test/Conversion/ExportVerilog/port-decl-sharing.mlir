// RUN: circt-opt %s -export-verilog --split-input-file | FileCheck %s --match-full-lines

module attributes {circt.loweringOptions = "disallowPortDeclSharing"}{
// CHECK:      module Foo(     // dummy:1:1
// CHECK-NEXT:   input        a,       // dummy:1:2
// CHECK-NEXT:   input        b,       // dummy:1:3
// CHECK-NEXT:   output [1:0] a_0,     // dummy:1:4
// CHECK-NEXT:   output [1:0] b_0      // dummy:1:5
// CHECK-NEXT: );
hw.module @Foo(%a: i1 loc("dummy":1:2), %b: i1 loc("dummy":1:3)) -> (a: i2 loc("dummy":1:4), b: i2 loc("dummy":1:5)) {
  %ao = comb.concat %a, %b: i1, i1
  %bo = comb.concat %a, %a: i1, i1
  hw.output %ao, %bo : i2, i2
} loc("dummy":1:1)
}

// -----

module {
// CHECK:      module Foo(     // dummy:1:1
// CHECK-NEXT:   input        a,       // dummy:1:2
// CHECK-NEXT:                b,       // dummy:1:3
// CHECK-NEXT:   output [1:0] a_0,     // dummy:1:4
// CHECK-NEXT:                b_0      // dummy:1:5
// CHECK-NEXT: );
hw.module @Foo(%a: i1 loc("dummy":1:2), %b: i1 loc("dummy":1:3)) -> (a: i2 loc("dummy":1:4), b: i2 loc("dummy":1:5)) {
  %ao = comb.concat %a, %b: i1, i1
  %bo = comb.concat %a, %a: i1, i1
  hw.output %ao, %bo : i2, i2
} loc("dummy":1:1)
}

// -----
// CHECK:      module Foo(     // dummy:1:1
// CHECK-NEXT:   // input  /*Zero Width*/ a,   // dummy:1:2
// CHECK-NEXT:   // input  /*Zero Width*/ b,   // dummy:1:3
// CHECK-NEXT:   // output /*Zero Width*/ a_0, // dummy:1:4
// CHECK-NEXT:   // output /*Zero Width*/ b_0  // dummy:1:5
// CHECK-NEXT: );
module attributes {circt.loweringOptions = "disallowPortDeclSharing"}{
hw.module @Foo(%a: i0 loc("dummy":1:2), %b: i0 loc("dummy":1:3)) -> (a: i0 loc("dummy":1:4), b: i0 loc("dummy":1:5)) {
  hw.output %a, %b : i0, i0
} loc("dummy":1:1)
}

// -----

module {
// CHECK:      module Foo(     // dummy:1:1
// CHECK-NEXT:   // input  /*Zero Width*/ a,   // dummy:1:2
// CHECK-NEXT:   //                       b,   // dummy:1:3
// CHECK-NEXT:   // output /*Zero Width*/ a_0, // dummy:1:4
// CHECK-NEXT:   //                       b_0  // dummy:1:5
// CHECK-NEXT: );
hw.module @Foo(%a: i0 loc("dummy":1:2), %b: i0 loc("dummy":1:3)) -> (a: i0 loc("dummy":1:4), b: i0 loc("dummy":1:5)) {
  hw.output %a, %b : i0, i0
} loc("dummy":1:1)
}

// -----

module attributes {circt.loweringOptions = "disallowPortDeclSharing"}{
// CHECK:      module Foo(     // dummy:1:1
// CHECK-NEXT:   // input  /*Zero Width*/ a,   // dummy:1:2
// CHECK-NEXT:   // input  /*Zero Width*/ b,   // dummy:1:3
// CHECK-NEXT:      input  [99:0]         c    // new:1:1
// CHECK-NEXT:   // output /*Zero Width*/ a_0, // dummy:1:4
// CHECK-NEXT:   // output /*Zero Width*/ b_0  // dummy:1:5
// CHECK-NEXT: );
hw.module @Foo(%a: i0 loc("dummy":1:2), %b: i0 loc("dummy":1:3), %c : i100 loc("new":1:1)) -> (a: i0 loc("dummy":1:4), b: i0 loc("dummy":1:5)) {
  hw.output %a, %b : i0, i0
} loc("dummy":1:1)
}

// -----

module {
// CHECK:      module Foo(     // dummy:1:1
// CHECK-NEXT:   // input  /*Zero Width*/ a,   // dummy:1:2
// CHECK-NEXT:   //                       b,   // dummy:1:3
// CHECK-NEXT:      input  [99:0]         c    // new:1:1
// CHECK-NEXT:   // output /*Zero Width*/ a_0, // dummy:1:4
// CHECK-NEXT:   //                       b_0  // dummy:1:5
// CHECK-NEXT: );
hw.module @Foo(%a: i0 loc("dummy":1:2), %b: i0 loc("dummy":1:3), %c : i100 loc("new":1:1)) -> (a: i0 loc("dummy":1:4), b: i0 loc("dummy":1:5)) {
  hw.output %a, %b : i0, i0
} loc("dummy":1:1)
}
