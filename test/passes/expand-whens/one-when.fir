; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; XFAIL: *

; CHECK: Expand Whens
circuit top :
   module top :
      input clk : Clock
      input reset : UInt<1>
      mem m : 
         data-type => UInt<1>
         depth => 2
         read-latency => 0
         write-latency => 1
      wire i : UInt<1>
      wire p : UInt<1>
      wire j : UInt<1>
      j <= UInt(1)
      reg r : UInt<1>, clk, reset,i

      p <= j
      when p :
        
        infer accessor a = m[i] 
        i <= a
        infer accessor b = m[i]
        b <= i
      else :
        infer accessor c = m[i]
        i <= c
        infer accessor d = m[i] 
        d <= i
      infer accessor e = m[i]
      when p :
        p <= i
        when e :
           p <= p
           r <= p
        

; CHECK: Finished Expand Whens
