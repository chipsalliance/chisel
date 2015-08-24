; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Done!
circuit Counter : 
  module Counter : 
    input inc : UInt<1>
    input clk : Clock
    input reset : UInt<1>
    output tot : UInt<8>
    input amt : UInt<4>
    
    reg T_13 : UInt<8>,clk,reset
    onreset T_13 := UInt<8>(0)
    when inc : 
      node T_14 = addw(T_13, amt)
      node T_15 = gt(T_14, UInt<8>(255))
      node T_16 = mux(T_15, UInt<1>(0), T_14)
      T_13 := T_16
    tot := T_13
