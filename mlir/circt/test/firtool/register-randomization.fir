; RUN: firtool -preserve-values=all -verilog %s | FileCheck %s --check-prefix=ALL
; RUN: firtool -preserve-values=named -verilog %s | FileCheck %s --check-prefix=NAMED
; RUN: firtool -preserve-values=none -verilog %s | FileCheck %s --check-prefix=NONE
; RUN: firtool -preserve-values=all -verilog %s --lowering-options=disallowLocalVariables| FileCheck %s --check-prefix=LOCAL

circuit Foo:
  module Foo:
    input clock: Clock
    input d: UInt<33>
    input d0: UInt<60000>
    input d1: UInt<120000>
    output q: UInt<33>

    ; ALL:       _r <= d;
    ; NAMED-NOT: _r = {{.*}};
    ; NONE-NOT:  _r = {{.*}};
    ; LOCAL:     _r <= d;
    reg _r: UInt<33>, clock
    _r <= d

    reg r: UInt<33>, clock
    r <= d

    reg s: UInt<33>, clock
    s <= d

    q <= r

    ; ALL:        automatic logic [31:0] _RANDOM[0:3];
    ; ALL:        for (logic [2:0] i = 3'h0; i < 3'h4; i += 3'h1) begin
    ; ALL-NEXT:     _RANDOM[i[1:0]] = `RANDOM;

    ; NAMED:        automatic logic [31:0] _RANDOM[0:3];
    ; NAMED:        for (logic [2:0] i = 3'h0; i < 3'h4; i += 3'h1) begin
    ; NAMED-NEXT:     _RANDOM[i[1:0]] = `RANDOM;

    ; NONE:        automatic logic [31:0] _RANDOM[0:2];
    ; NONE:        for (logic [1:0] i = 2'h0; i < 2'h3; i += 2'h1) begin
    ; NONE-NEXT:     _RANDOM[i] = `RANDOM;

    ; ALL:       _r = {_RANDOM[2'h0], _RANDOM[2'h1][0]};
    ; NAMED-NOT: _r =
    ; NONE-NOT:  _r =

    ; ALL:       r = {_RANDOM[2'h1][31:1], _RANDOM[2'h2][1:0]};
    ; NAMED:     r = {_RANDOM[2'h1][31:1], _RANDOM[2'h2][1:0]};
    ; NONE:      r = {_RANDOM[2'h1][31:1], _RANDOM[2'h2][1:0]};

    ; ALL:       s = {_RANDOM[2'h2][31:2], _RANDOM[2'h3][2:0]};
    ; NAMED:     s = {_RANDOM[2'h2][31:2], _RANDOM[2'h3][2:0]};
    ; NONE-NOT:  s = {{.*}};
