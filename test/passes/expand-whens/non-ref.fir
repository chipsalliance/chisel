; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s

; CHECK: Expand Whens
circuit top :
   module top :
      input clk : Clock
      input reset : UInt<1>
      wire x : UInt<1>

; CHECK-NOT: wire x : UInt<1>
; CHECK: Finished Expand Whens

