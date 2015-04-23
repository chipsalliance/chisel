; RUN: firrtl -i %s -o %s.flo -x abcdefghijk -p c | tee %s.out | FileCheck %s

; CHECK: Expand Whens
circuit top :
   module top :
      mem m : UInt<1>[2]
      wire i : UInt<1>
      wire p : UInt<1>
      wire j : UInt<1>
      reg r : UInt<1>

      p := j
      when p :
        on-reset r := i
        accessor a = m[i] 
        i := a
        accessor b = m[i]
        b := i
      else :
        accessor c = m[i]
        i := c
        accessor d = m[i] 
        d := i
      accessor e = m[i]
      when p :
        p := i
        when e :
           p := p
           on-reset r := p
           r := p
        

; CHECK: Finished Expand Whens
