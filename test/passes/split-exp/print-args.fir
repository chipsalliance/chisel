; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

; CHECK: Split Expressions
; CHECK: node GEN_0 = and(a, b)
; CHECK: printf(clk, UInt<1>("h1"), "%d\n", GEN_0)

circuit Bug :
  module Bug :
    input clk : Clock
    input a : UInt<1>
    input b : UInt<1>

    printf(clk, UInt<1>(1), "%d\n", and(a, b))

