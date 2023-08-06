; RUN: firtool -split-input-file -verilog %s | FileCheck %s

; This test checks end-to-end compliance with the Scala FIRRTL Compiler (SFC)
; context-sensitive interpretation of invalid.

; CHECK-LABEL: module InvalidInterpretations
circuit InvalidInterpretations:
  module InvalidInterpretations:
    input clock: Clock
    input reset: UInt<1>
    input cond: UInt<1>
    input a: UInt<8>
    output out_when: UInt<8>
    output out_reg: UInt<8>
    output out_mux: UInt<8>
    output out_add: UInt<9>

    wire _inv: UInt<8>
    _inv is invalid

    reg r: UInt<8>, clock with : (reset => (reset, _inv))
    r <= a
    out_reg <= r
    ; Interpretation 1: Invalid is undefined if used as the initialization value
    ; of a register in a module-scoped analysis that looks through connects.
    ; CHECK:       always @(posedge clock)
    ; CHECK-NOT:     if (reset)

    out_when is invalid
    when cond:
      out_when <= a
    ; Interpretation 2: Invalid is undefined when used as a default value.
    ; CHECK:       assign out_when = a;

    out_mux <= mux(cond, a, _inv)
    out_add <= add(a, _inv)
    ; Interpretation 4: Invalid is zero otherwise.
    ; CHECK:       assign out_mux = cond ? a : 8'h0;
    ; CHECK-NEXT:  assign out_add = {1'h0, a};

; // -----

; This is checking that an invalid value in another module is not propagated,
; but is interpreted as "zero".  The end result of this is that the main module
; should have a register that will be reset to a non-zero value, but is
; constantly driven to zero by default.
;
; See: https://github.com/llvm/circt/issues/2782

; CHECK-LABEL: module InvalidInOtherModule
circuit InvalidInOtherModule :
  module InvalidInOtherModule :
    input clock: Clock
    input reset: UInt<1>
    output b: SInt<8>

    inst other of OtherModule

    ; CHECK:      always @(posedge clock)
    ; CHECK-NEXT:   if (reset)
    ; CHECK-NEXT:     r <= 8'h4;
    ; CHECK-NEXT:   else
    ; CHECK-NEXT:     r <= 8'h0;
    reg r : SInt<8>, clock with: (reset => (reset, SInt<8>(4)))
    r <= other.b
    b <= r

  module OtherModule :
    output b: SInt<8>
    b is invalid
