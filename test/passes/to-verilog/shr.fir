; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 |  tee %s.out | FileCheck %s

;CHECK: Lower Types
;CHECK: node T_928 = bits(s1_req_addr, 11, 0)
;CHECK: node s1_addr = cat(dtlb.resp_ppn, T_928)
;CHECK: Finished Lower Types

circuit HellaCache : 
  module TLB_60 : 
    output resp : {ppn : UInt<20>}
    resp.ppn <= UInt<1>("h00")

  module HellaCache : 
    input clock : Clock
    input reset : UInt<1>

    inst dtlb of TLB_60
    poison init : {addr : UInt<40>}
    reg s1_req : {addr : UInt<40>}, clock, reset, init
    reg s2_req : {addr : UInt<40>}, clock, reset, init
    reg s1_clk_en : UInt<1>, clock, reset, UInt(0)

    node T_928 = bits(s1_req.addr, 11, 0)
    node s1_addr = cat(dtlb.resp.ppn, T_928)
    when s1_clk_en :
      s2_req.addr <= s1_addr

    s1_req.addr <= UInt<?>(0)
    s1_clk_en <= UInt<?>(0)

    wire foo : UInt<28>
    foo <= shr(s1_addr, 3)
