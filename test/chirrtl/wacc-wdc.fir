; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
circuit top :
   module top :
      input clk : Clock
      wire p : UInt
      cmem m : UInt<4>[10]
      p <= UInt(1)
      when p :
         write mport a = m[UInt(3)],clk,UInt(1)
         a <= UInt(2)

; CHECK: To FIRRTL

; CHECK: mem m :          
; CHECK:    data-type: UInt<4>
; CHECK:    depth: 10
; CHECK:    write-latency: 1
; CHECK:    read-latency: 0
; CHECK:    writer: a
; CHECK: poison GEN : UInt<4>
; CHECK: poison GEN_1 : UInt<4>
; CHECK: m.a.addr <= GEN
; CHECK: m.a.clk <= clk
; CHECK: m.a.en <= UInt("h0")
; CHECK: m.a.data <= GEN_1
; CHECK: m.a.mask <= UInt("h0")
; CHECK: p <= UInt("h1")
; CHECK: when p :
; CHECK:    m.a.addr <= UInt("h3")
; CHECK:    m.a.en <= UInt("h1")
; CHECK:    m.a.mask <= UInt("h1")
; CHECK:    m.a.data <= UInt("h2")   

; CHECK: Finished To FIRRTL
; CHECK: Done!


