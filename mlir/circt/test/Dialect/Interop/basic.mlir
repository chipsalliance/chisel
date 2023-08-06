// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: @basic
hw.module @basic (%arg0: i32, %arg1: i8) -> () {
  // CHECK-NEXT: interop.procedural.alloc cpp
  interop.procedural.alloc cpp

  // CHECK-NEXT: {{%.+}}:2 = interop.procedural.alloc cffi : i1, i32
  %s:2 = interop.procedural.alloc cffi : i1, i32

  // Test: no return values, cpp mechanism
  // CHECK-NEXT: interop.procedural.init cpp {
  interop.procedural.init cpp {
    // CHECK-NEXT: interop.return
    interop.return
  // CHECK-NEXT: }
  }

  // Test: one return value
  // CHECK-NEXT: interop.procedural.init cpp {{%.+}} : i1 {
  interop.procedural.init cpp %s#0 : i1 {
    // CHECK-NEXT: %true = hw.constant true
    %true = hw.constant true
    // CHECK-NEXT: interop.return %true : i1
    interop.return %true : i1
  // CHECK-NEXT: }
  }

  // Test: multiple return values, cffi mechanism
  // CHECK-NEXT: interop.procedural.init cffi {{%.+}}, {{%.+}}: i1, i1 {
  interop.procedural.init cffi %s#0, %s#0: i1, i1 {
    // CHECK-NEXT: %true = hw.constant true
    %true = hw.constant true
    // CHECK-NEXT: interop.return %true, %true : i1, i1
    interop.return %true, %true : i1, i1
  // CHECK-NEXT: }
  }

  // Test: no state, no input, no output
  // CHECK-NEXT: interop.procedural.update cpp : () -> () {
  interop.procedural.update cpp : () -> () {
    // CHECK-NEXT: interop.return
    interop.return
  // CHECK-NEXT: }
  }

  // Test: One state, one input, one output, cpp mechanism
  // CHECK-NEXT: {{%.+}} = interop.procedural.update cpp [{{%.+}}] (%arg0) : [i1] (i32) -> i32 {
  %0 = interop.procedural.update cpp [%s#0](%arg0) : [i1] (i32) -> i32 {
  // CHECK-NEXT: ^bb0({{%.+}}: i1, [[RV:%.+]]: i32):
  ^bb0(%barg0: i1, %barg1: i32):
    // CHECK-NEXT: interop.return [[RV]] : i32
    interop.return %barg1 : i32
  // CHECK-NEXT: }
  }

  // Test: multiple states, multiple inputs, multiple outputs, cffi mechanism
  // CHECK-NEXT: {{%.+}}:2 = interop.procedural.update cffi [{{%.+}}, {{%.+}}] (%arg0, %arg1) : [i1, i1] (i32, i8) -> (i32, i8) {
  %1:2 = interop.procedural.update cffi[%s#0, %s#0](%arg0, %arg1) : [i1, i1](i32, i8) -> (i32, i8) {
  // CHECK-NEXT: ^bb0({{%.+}}: i1, {{%.+}}: i1, [[RV1:%.+]]: i32, [[RV2:%.+]]: i8):
  ^bb0(%barg0: i1, %barg1: i1, %barg2: i32, %barg3: i8):
    // CHECK-NEXT: interop.return [[RV1]], [[RV2]] : i32, i8
    interop.return %barg2, %barg3 : i32, i8
  // CHECK-NEXT: }
  }

  // Test: zero states, multiple inputs, multiple outputs, cffi mechanism
  // CHECK-NEXT: {{%.+}}:2 = interop.procedural.update cffi (%arg0, %arg1) : (i32, i8) -> (i32, i8) {
  %2:2 = interop.procedural.update cffi (%arg0, %arg1) : (i32, i8) -> (i32, i8) {
  // CHECK-NEXT: ^bb0([[RV1:%.+]]: i32, [[RV2:%.+]]: i8):
  ^bb0(%barg2: i32, %barg3: i8):
    // CHECK-NEXT: interop.return [[RV1]], [[RV2]] : i32, i8
    interop.return %barg2, %barg3 : i32, i8
  // CHECK-NEXT: }
  }

  // Test: no state, cpp mechanism
  // CHECK-NEXT: interop.procedural.dealloc cpp {
  interop.procedural.dealloc cpp {
  // CHECK-NEXT: }
  }

  // Test: multiple states, cffi mechanism
  // CHECK-NEXT: interop.procedural.dealloc cffi {{%.+}}, {{%.+}} : i1, i32 {
  interop.procedural.dealloc cffi %s#0, %s#1 : i1, i32 {
  // CHECK-NEXT: ^bb0({{%.+}}: i1, {{%.+}}: i32):
  ^bb0(%barg0: i1, %barg1: i32):
  // CHECK-NEXT: }
  }
}
