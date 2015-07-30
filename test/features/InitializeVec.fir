; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
;CHECK: Done!
circuit Tst : 
  module Tst : 
    output in : {valid : UInt<1>, flip ready : UInt<1>, bits : UInt<8>}
    output outs : {valid : UInt<1>, flip ready : UInt<1>, bits : UInt<8>}[4]
    
    in.ready := UInt<1>(1)
    outs[0].valid := UInt<1>(0)
    outs[0].bits := UInt<1>(0)
    outs[1].valid := UInt<1>(0)
    outs[1].bits := UInt<1>(0)
    outs[2].valid := UInt<1>(0)
    outs[2].bits := UInt<1>(0)
    outs[3].valid := UInt<1>(0)
    outs[3].bits := UInt<1>(0)
    in.ready := UInt<1>(1)
    infer accessor out = outs[in.bits]
    when out.ready : 
      out.bits := UInt<7>(99)
      out.valid := UInt<1>(1)
