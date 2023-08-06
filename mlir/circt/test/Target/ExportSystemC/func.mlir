// RUN: circt-translate %s --export-systemc | FileCheck %s

// CHECK-LABEL: // stdout.h
// CHECK-NEXT: #ifndef STDOUT_H
// CHECK-NEXT: #define STDOUT_H

// CHECK: uint32_t funcDeclaration(uint32_t funcArg0, uint32_t funcArg1);
systemc.cpp.func private @funcDeclaration (%funcArg0: i32, %funcArg1: i32) -> i32
// CHECK-EMPTY:
// CHECK-NEXT: void voidFunc() {
systemc.cpp.func @voidFunc () {
  // CHECK-NEXT: return;
  systemc.cpp.return
// CHECK-NEXT: }
}
// CHECK-EMPTY: 
// CHECK-NEXT: uint32_t testFunc(uint64_t a, uint32_t b) {
systemc.cpp.func @testFunc (%a: i64, %b: i32) -> i32 {
  %0 = systemc.cpp.call @funcDeclaration(%b, %b) : (i32, i32) -> i32
  // CHECK-NEXT: voidFunc();
  systemc.cpp.call @voidFunc() : () -> ()
  // CHECK-NEXT: SomeClass v0;
  // CHECK-NEXT: (v0.someFunc)();
  %v0 = systemc.cpp.variable : !emitc.opaque<"SomeClass">
  %1 = systemc.cpp.member_access %v0 dot "someFunc" : (!emitc.opaque<"SomeClass">) -> (() -> ())
  systemc.cpp.call_indirect %1 () : () -> ()
  // CHECK-NEXT: uint32_t v1 = funcDeclaration(b, b);
  %2 = systemc.cpp.call @funcDeclaration(%b, %b) : (i32, i32) -> i32
  %v1 = systemc.cpp.variable %2 : i32
  // CHECK-NEXT: return funcDeclaration(b, b);
  systemc.cpp.return %0 : i32
// CHECK-NEXT: }
}
// CHECK-EMPTY: 
// CHECK-NEXT: extern "C" void declarationWithoutArgNames(uint32_t, uint8_t);
systemc.cpp.func externC private @declarationWithoutArgNames(i32, i8)

// CHECK: #endif // STDOUT_H
