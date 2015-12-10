; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Done!
circuit Tst : 
  module Tst : 
    input in : {valid : UInt<1>, flip ready : UInt<1>, bits : UInt<8>}
    output outs : {valid : UInt<1>, flip ready : UInt<1>, bits : UInt<8>}[4]
    
    in.ready <= UInt<1>(1)
    outs[0].valid <= UInt<1>(0)
    outs[0].bits <= UInt<1>(0)
    outs[1].valid <= UInt<1>(0)
    outs[1].bits <= UInt<1>(0)
    outs[2].valid <= UInt<1>(0)
    outs[2].bits <= UInt<1>(0)
    outs[3].valid <= UInt<1>(0)
    outs[3].bits <= UInt<1>(0)
    in.ready <= UInt<1>(1)
    when outs[in.bits].ready : 
      outs[in.bits].bits <= UInt<7>(99)
      outs[in.bits].valid <= UInt<1>(1)
