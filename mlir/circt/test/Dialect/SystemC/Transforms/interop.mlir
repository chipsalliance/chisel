// RUN: circt-opt --systemc-lower-instance-interop %s | FileCheck %s

hw.module.extern @Bar (%a: i32, %b: i32) -> (c: i32)
hw.module.extern @Empty () -> ()

// CHECK-LABEL: @Foo
hw.module @Foo (%x: i32) -> (y: i32) {
  // CHECK-NEXT: [[STATE:%.+]] = interop.procedural.alloc cpp : !emitc.ptr<!emitc.opaque<"VBar">>
  // CHECK-NEXT: interop.procedural.init cpp [[STATE]] : !emitc.ptr<!emitc.opaque<"VBar">> {
  // CHECK-NEXT:   [[V1:%.+]] = systemc.cpp.new() : () -> !emitc.ptr<!emitc.opaque<"VBar">>
  // CHECK-NEXT:   interop.return [[V1]] : !emitc.ptr<!emitc.opaque<"VBar">>
  // CHECK-NEXT: }
  // CHECK-NEXT: {{%.+}} = interop.procedural.update cpp [[[STATE]]] (%x, %x) : [!emitc.ptr<!emitc.opaque<"VBar">>] (i32, i32) -> i32 {
  // CHECK-NEXT: ^bb0(%arg0: !emitc.ptr<!emitc.opaque<"VBar">>, %arg1: i32, %arg2: i32):
  // CHECK-NEXT:   [[V2:%.+]] = systemc.cpp.member_access %arg0 arrow "a" : (!emitc.ptr<!emitc.opaque<"VBar">>) -> i32
  // CHECK-NEXT:   systemc.cpp.assign [[V2]] = %arg1 : i32
  // CHECK-NEXT:   [[V3:%.+]] = systemc.cpp.member_access %arg0 arrow "b" : (!emitc.ptr<!emitc.opaque<"VBar">>) -> i32
  // CHECK-NEXT:   systemc.cpp.assign [[V3]] = %arg2 : i32
  // CHECK-NEXT:   [[V4:%.+]] = systemc.cpp.member_access %arg0 arrow "eval" : (!emitc.ptr<!emitc.opaque<"VBar">>) -> (() -> ())
  // CHECK-NEXT:   func.call_indirect [[V4]]() : () -> ()
  // CHECK-NEXT:   [[V5:%.+]] = systemc.cpp.member_access %arg0 arrow "c" : (!emitc.ptr<!emitc.opaque<"VBar">>) -> i32
  // CHECK-NEXT:   interop.return [[V5]] : i32
  // CHECK-NEXT: }
  // CHECK-NEXT: interop.procedural.dealloc cpp [[STATE]] : !emitc.ptr<!emitc.opaque<"VBar">> {
  // CHECK-NEXT: ^bb0(%arg0: !emitc.ptr<!emitc.opaque<"VBar">>):
  // CHECK-NEXT:   systemc.cpp.delete %arg0 : !emitc.ptr<!emitc.opaque<"VBar">>
  // CHECK-NEXT: }
  %c = systemc.interop.verilated "inst0" @Bar (a: %x: i32, b: %x: i32) -> (c: i32)

  // CHECK-NEXT: [[STATE2:%.+]] = interop.procedural.alloc cpp : !emitc.ptr<!emitc.opaque<"VEmpty">>
  // CHECK-NEXT: interop.procedural.init cpp [[STATE2]] : !emitc.ptr<!emitc.opaque<"VEmpty">> {
  // CHECK-NEXT:   [[V6:%.+]] = systemc.cpp.new() : () -> !emitc.ptr<!emitc.opaque<"VEmpty">>
  // CHECK-NEXT:   interop.return [[V6]] : !emitc.ptr<!emitc.opaque<"VEmpty">>
  // CHECK-NEXT: }
  // CHECK-NEXT: interop.procedural.update cpp [[[STATE2]]] : [!emitc.ptr<!emitc.opaque<"VEmpty">>] () -> () {
  // CHECK-NEXT: ^bb0(%arg0: !emitc.ptr<!emitc.opaque<"VEmpty">>):
  // CHECK-NEXT:   [[V7:%.+]] = systemc.cpp.member_access %arg0 arrow "eval" : (!emitc.ptr<!emitc.opaque<"VEmpty">>) -> (() -> ())
  // CHECK-NEXT:   func.call_indirect [[V7]]() : () -> ()
  // CHECK-NEXT:   interop.return
  // CHECK-NEXT: }
  // CHECK-NEXT: interop.procedural.dealloc cpp [[STATE2]] : !emitc.ptr<!emitc.opaque<"VEmpty">> {
  // CHECK-NEXT: ^bb0(%arg0: !emitc.ptr<!emitc.opaque<"VEmpty">>):
  // CHECK-NEXT:   systemc.cpp.delete %arg0 : !emitc.ptr<!emitc.opaque<"VEmpty">>
  // CHECK-NEXT: }
  systemc.interop.verilated "inst1" @Empty () -> ()

  // CHECK-NEXT: hw.output {{%.+}} : i32
  hw.output %c : i32
}
