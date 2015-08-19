; RUN: firrtl -i %s -o %s.v -X verilog -p cTd 2>&1 |  tee %s.out | FileCheck %s

;CHECK: Infer Widths
;CHECK: node s1_addr = cat(dtlb<t:{ resp$ppn : UInt<20><t:UInt<20>>}>.resp$ppn<t:UInt<20>>, bits(s1_req$addr<t:UInt<40>>, 11, 0)<t:UInt<12>>)<t:UInt<32>>
;CHECK: Finished Infer Widths

circuit HellaCache : 
  module TLB_60 : 
    output resp : {ppn : UInt<20>}
    resp.ppn := UInt<1>("h00")

  module HellaCache : 
    input clock : Clock
    input reset : UInt<1>

    inst dtlb of TLB_60
    reg s1_req : {addr : UInt<40>}, clock, reset
    reg s2_req : {addr : UInt<40>}, clock, reset
    reg s1_clk_en : UInt<1>, clock, reset

    node T_928 = bits(s1_req.addr, 11, 0)
    node s1_addr = cat(dtlb.resp.ppn, T_928)
    when s1_clk_en :
      s2_req.addr := s1_addr

    s1_req.addr := UInt<?>(0)
    s1_clk_en := UInt<?>(0)

    wire foo : UInt<28>
    foo := shr(s1_addr, 3)
