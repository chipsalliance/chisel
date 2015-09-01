; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; XFAIL: *
circuit CondRead :
  module CondRead :
    input pred : UInt<1>
    input index : UInt<6>
    input clk : Clock
    output out : UInt<20>

    smem mem : UInt<20>[128],clk
    
    poison xxx : UInt<20>
    wire data : UInt<20>
    data := xxx
    when pred :
      read accessor readport = mem[index]
      data := readport
    
    out := data


; CHECK: poison index_0 : UInt<6>
; CHECK: read accessor readport = mem[mux(pred,index,index_0)]
; CHECK: Done!

