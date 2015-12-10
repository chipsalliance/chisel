; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Done!
circuit Top : 
  module Top : 
    input clk : Clock
    wire i : UInt<5>
    i <= UInt(1)
    wire i0 : UInt<5>
    wire j : UInt<128>

    i0 <= UInt(10)

    mem m-com : 
       data-type => UInt<128>
       depth => 32
       reader => r
       writer => w
       read-latency => 0
       write-latency => 1
    m-com.r.addr <= i
    m-com.r.en <= UInt(1)
    m-com.r.clk <= clk
    m-com.w.addr <= i
    m-com.w.mask <= UInt(1)
    m-com.w.en <= UInt(1)
    m-com.w.clk <= clk
    j <= m-com.r.data
    m-com.w.data <= j


    mem m-seq : 
       data-type => UInt<128>
       depth => 32
       reader => r
       writer => w
       read-latency => 1
       write-latency => 1
    m-seq.r.addr <= i
    m-seq.r.en <= UInt(1)
    m-seq.r.clk <= clk
    m-seq.w.addr <= i
    m-seq.w.mask <= UInt(1)
    m-seq.w.en <= UInt(1)
    m-seq.w.clk <= clk
    j <= m-seq.r.data
    m-seq.w.data <= j
