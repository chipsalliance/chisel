; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s

; CHECK: Expand Whens
circuit top :
   module top :
      cmem m :{ x : UInt<1>, y : UInt<1> }[2] 
      wire i : UInt<1>
      wire p : UInt<1>
      wire q : { x : UInt<1>, y : UInt<1> }
      when p :
        wire p2 : UInt<1>
        reg r5 : UInt<1>
        when p2 :
          accessor a = m[i] 
          q := a
          accessor b = m[i]
          b := q
        else :
          accessor c = m[i] 
          q := c
          accessor d = m[i]
          d := q
      else :
        wire p3 : UInt<1>
        when p3 :
          accessor w = m[i] 
          q := w
          accessor x = m[i]
          x := q
        else :
          accessor y = m[i] 
          q := y
          accessor z = m[i]
          z := q
; CHECK: Finished Expand Whens
