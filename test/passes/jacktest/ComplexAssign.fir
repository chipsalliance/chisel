; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Done!
circuit ComplexAssign : 
  module ComplexAssign : 
    input in : {re : UInt<10>, im : UInt<10>}
    output out : {re : UInt<10>, im : UInt<10>}
    input e : UInt<1>
    when e : 
      wire T_18 : {re : UInt<10>, im : UInt<10>}
      T_18 := in
      out.re := T_18.re
      out.im := T_18.im
    else : 
      out.re := UInt<1>(0)
      out.im := UInt<1>(0)
