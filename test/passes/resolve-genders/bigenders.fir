; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Resolve Genders
circuit top :
   module top :
      input i : UInt<10>
      output o : UInt<10>
      wire w : {x : UInt<10>, flip y : UInt<10>}
      w.x <= i
      w.y <= i
      o <= w.x
      o <= w.y
; CHECK: Finished Resolve Genders
