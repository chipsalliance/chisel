; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
circuit top :
   module top :
      input clk : Clock
      wire p : UInt
      wire q : UInt
      cmem m : {a:UInt<4>,b:UInt<4>}[10]
      p <= UInt(1)
      q <= UInt(1)
      wire x : {a:UInt<4>,b:UInt<4>}
      x.a <= UInt(1)
      x.b <= UInt(1)
      when p :
         write mport a = m[UInt(3)],clk
         when q :
            a <= x


; CHECK: Remove CHIRRTL

; CHECK: circuit top :
; CHECK:    module top :
; CHECK:       input clk : Clock
; CHECK:       wire p : UInt
; CHECK:       wire q : UInt
; CHECK:       mem m :          
; CHECK:          data-type => { a : UInt<4>,  b : UInt<4>}
; CHECK:          depth => 10
; CHECK:          write-latency => 1
; CHECK:          read-latency => 0
; CHECK:          writer => a
; CHECK:       m.a.addr is invalid
; CHECK:       m.a.clk <= clk
; CHECK:       m.a.en <= UInt("h0")
; CHECK:       m.a.data is invalid
; CHECK:       m.a.mask.a <= UInt("h0")
; CHECK:       m.a.mask.b <= UInt("h0")
; CHECK:       p <= UInt("h1")
; CHECK:       q <= UInt("h1")
; CHECK:       wire x : { a : UInt<4>,  b : UInt<4>}
; CHECK:       x.a <= UInt("h1")
; CHECK:       x.b <= UInt("h1")
; CHECK:       when p :
; CHECK:          m.a.addr <= UInt("h3")
; CHECK:          m.a.en <= UInt("h1")
; CHECK:          when q :
; CHECK:             m.a.data <= x
; CHECK:             m.a.mask.a <= UInt("h1")
; CHECK:             m.a.mask.b <= UInt("h1")   

; CHECK: Finished Remove CHIRRTL
; CHECK: Done!


