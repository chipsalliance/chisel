; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Done!

circuit Tbl :
  module Tbl :
    input clk : Clock
    input reset : UInt<1>
    output io : {flip wi : UInt<3>, flip ri : UInt<3>, flip we : UInt<1>, flip d : UInt<3>, o : UInt<3>}

    io.o <= UInt<1>("h00")
    cmem m : UInt<3>[8]
    infer mport T_12 = m[io.ri], clk
    io.o <= T_12
    when io.we :
      infer mport T_13 = m[io.wi], clk
      T_13 <= io.d
      node T_14 = eq(io.ri, io.wi)
      when T_14 :
        io.o <= io.d
        skip
      skip
