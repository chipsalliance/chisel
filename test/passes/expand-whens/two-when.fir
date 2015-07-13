; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s

; CHECK: Expand Whens
circuit top :
   module top :
      input clk : Clock
      cmem m :{ x : UInt<1>, y : UInt<1> }[2], clk
      wire i : UInt<1>
      i := UInt(1)
      wire p : UInt<1>
      p := UInt(1)
      wire q : { x : UInt<1>, y : UInt<1> }
      when p :
        wire p2 : UInt<1>
        p2 := UInt(1)
        when p2 :
          infer accessor a = m[i] 
          q := a
          infer accessor b = m[i]
          b := q
        else :
          infer accessor c = m[i] 
          q := c
          infer accessor d = m[i]
          d := q
      else :
        wire p3 : UInt<1>
        p3 := UInt(1)
        when p3 :
          infer accessor w = m[i] 
          q := w
          infer accessor x = m[i]
          x := q
        else :
          infer accessor y = m[i] 
          q := y
          infer accessor z = m[i]
          z := q
; CHECK: Finished Expand Whens
