; RUN: firtool --split-input-file --verilog %s | FileCheck %s
; Tests extracted from:
; - test/scala/firrtlTests/AsyncResetSpec.scala

; Unclear if we want to require this split of processes in MFC.
; XFAIL: *

; Every async reset reg should generate its own always block.
; CHECK-LABEL: module Foo(
circuit Foo:
  module Foo:
    input clock0 : Clock
    input clock1 : Clock
    input syncReset : UInt<1>
    input asyncReset : AsyncReset
    input x : UInt<8>[5]
    output z : UInt<8>[5]
    reg r0 : UInt<8>, clock0 with : (reset => (syncReset, UInt(123)))
    reg r1 : UInt<8>, clock1 with : (reset => (syncReset, UInt(123)))
    reg r2 : UInt<8>, clock0 with : (reset => (asyncReset, UInt(123)))
    reg r3 : UInt<8>, clock0 with : (reset => (asyncReset, UInt(123)))
    reg r4 : UInt<8>, clock1 with : (reset => (asyncReset, UInt(123)))
    r0 <= x[0]
    r1 <= x[1]
    r2 <= x[2]
    r3 <= x[3]
    r4 <= x[4]
    z[0] <= r0
    z[1] <= r1
    z[2] <= r2
    z[3] <= r3
    z[4] <= r4

; CHECK:      always @(posedge clock0) begin
; CHECK:      if (syncReset)
; CHECK:      r0 <= 8'h7B;
; CHECK:      else
; CHECK:      r0 <= x_0;
; CHECK:      end

; CHECK:      always @(posedge clock1) begin
; CHECK:      if (syncReset)
; CHECK:      r1 <= 8'h7B;
; CHECK:      else
; CHECK:      r1 <= x_1;
; CHECK:      end

; CHECK:      always @(posedge clock0 or posedge asyncReset) begin
; CHECK:      if (asyncReset)
; CHECK:      r2 <= 8'h7B;
; CHECK:      else
; CHECK:      r2 <= x_2;
; CHECK:      end

; CHECK:      always @(posedge clock0 or posedge asyncReset) begin
; CHECK:      if (asyncReset)
; CHECK:      r3 <= 8'h7B;
; CHECK:      else
; CHECK:      r3 <= x_3;
; CHECK:      end

; CHECK:      always @(posedge clock1 or posedge asyncReset) begin
; CHECK:      if (asyncReset)
; CHECK:      r4 <= 8'h7B;
; CHECK:      else
; CHECK:      r4 <= x_4;
; CHECK:      end
