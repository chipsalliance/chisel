; RUN: firrtl -i %s -o %s.flo -x abcdefghijk -p c | tee %s.out | FileCheck %s

; CHECK: Expand Whens
circuit top :
   module top :
      reg r : UInt<1>[10]
      on-reset r[3] := UInt(0)

; CHECK: Finished Expand Whens
