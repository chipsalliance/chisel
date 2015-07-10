; RUN: firrtl -i %s -o %s.v -X verilog -p c | tee %s.out | FileCheck %s
;CHECK: Done!
circuit Top : 
  module Top : 
    input clk : Clock
    wire i : UInt<5>
    i := UInt(1)
    wire i0 : UInt<5>
    wire j : UInt<128>

    i0 := UInt(10)

    cmem m-com : UInt<128>[32], clk
    infer accessor r-com = m-com[i]
    infer accessor w-com = m-com[i]
    j := r-com
    w-com := j


    smem m-seq : UInt<128>[32], clk
    infer accessor r-seq = m-seq[i]
    infer accessor w-seq = m-seq[i]
    j := r-seq
    w-seq := j
