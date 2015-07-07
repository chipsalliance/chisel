; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s

; CHECK: Expand Whens
circuit top :
   module top :
      cmem m : UInt<1>[2]
      wire i : UInt<1>
      wire p : UInt<1>
      wire j : UInt<1>
      j := UInt(1)
      reg r : UInt<1>

      p := j
      when p :
        on-reset r := i
        infer accessor a = m[i] 
        i := a
        infer accessor b = m[i]
        b := i
      else :
        infer accessor c = m[i]
        i := c
        infer accessor d = m[i] 
        d := i
      infer accessor e = m[i]
      when p :
        p := i
        when e :
           p := p
           on-reset r := p
           r := p
        

; CHECK: Finished Expand Whens
