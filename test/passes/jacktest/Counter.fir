; RUN: firrtl -i %s -o %s.flo -X flo -p cTwd | tee %s.out | FileCheck %s
; CHECK: Done!

circuit Counter : 
  module Counter : 
    input inc : UInt<1>
    output tot : UInt<8>
    input amt : UInt<4>
    
    reg T_13 : UInt<8>
    on-reset T_13 := Pad(UInt<8>(0),?)
    when inc : 
      node T_14 = add-wrap(Pad(T_13,?), Pad(amt,?))
      node T_15 = gt(Pad(T_14,?), Pad(UInt<8>(255),?))
      node T_16 = mux(Pad(T_15,?), Pad(UInt<1>(0),?), Pad(T_14,?))
      T_13 := Pad(T_16,?)
    tot := Pad(T_13,?)
