; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

circuit top :
  module top : 
    input T_4910 : UInt<1>
    input T_4581 : UInt<1>
    input reset : UInt<1>
    input clock : Clock
    output out : UInt<1>
    reg T_4590 : UInt<1>[2], clock, reset, T_4590
    T_4590[0] <= UInt(0)
    T_4590[1] <= UInt(0)
    out <= UInt(0)
    when T_4910 :
      out <= T_4590[T_4581]
;CHECK: Done!
