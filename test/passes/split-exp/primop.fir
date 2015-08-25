; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Split Expressions
circuit Top :
   module Top :
      output out : UInt<1>

      wire m : UInt<1>[3]
      m[0] := UInt(0)
      m[1] := UInt(0)
      m[2] := UInt(0)

      wire x : UInt<1>
      x := not(UInt(1))
      infer accessor a = m[x]
      
      out := a



