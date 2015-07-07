; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s

; CHECK: Expand Whens
circuit top :
   module top :
      reg r : UInt<1>[10]
      r[0] := UInt(1)
      r[1] := UInt(1)
      r[2] := UInt(1)
      r[3] := UInt(1)
      r[4] := UInt(1)
      r[5] := UInt(1)
      r[6] := UInt(1)
      r[7] := UInt(1)
      r[8] := UInt(1)
      r[9] := UInt(1)
      on-reset r[3] := UInt(0)

; CHECK: Finished Expand Whens
