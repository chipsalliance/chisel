; RUN: circt-translate -import-firrtl -verify-diagnostics %s | circt-opt | FileCheck %s
FIRRTL version 1.1.0
circuit MyModule :     ; CHECK: firrtl.circuit "MyModule" {

  ; CHECK-LABEL: firrtl.module @MyModule(in %in: !firrtl.uint, out %out: !firrtl.uint<8>) {
  module MyModule :   @[FooBar.scala 369:27]
    input in: UInt
    output out: UInt<8>

    ; CHECK: firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint
    out <= in

  ; CHECK: }
