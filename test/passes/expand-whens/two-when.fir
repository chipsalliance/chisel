; RUN: firrtl %s abcdefghijk c | tee %s.out | FileCheck %s 

; CHECK: Expand Whens
circuit top :
   module top :
      mem m : UInt(1)[2]
      wire i : UInt(1)
      wire p : UInt(1)
      when p :
        wire p2 : UInt(1)
        when p2 :
          accessor a = m[i] 
          i := a
          accessor b = m[i]
          b := i
        else :
          accessor c = m[i] 
          i := c
          accessor d = m[i]
          d := i
      else :
        wire p3 : UInt(1)
        when p3 :
          accessor w = m[i] 
          i := w
          accessor x = m[i]
          x := i
        else :
          accessor y = m[i] 
          i := y
          accessor z = m[i]
          z := i
; CHECK: Finished Expand Whens
