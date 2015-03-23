; RUN: firrtl %s abcdefghijk c | tee %s.out | FileCheck %s 

; CHECK: Expand Whens
circuit top :
   module top :
      mem m : UInt(1)[2]
      wire i : UInt(1)
      wire p : UInt(1)
      when p :
        accessor a = m[i] 
        i := a
        accessor b = m[i]
        b := i
      else :
        accessor c = m[i]
        i := c
        accessor d = m[i] 
        d := i

; CHECK: Finished Expand Whens
