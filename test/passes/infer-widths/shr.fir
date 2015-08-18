; RUN: firrtl -i %s -o %s.v -X verilog -p cd 2>&1 | tee %s.out | FileCheck %s

;CHECK: Infer Widths
; CHECK: Finished Infer Widths

circuit MemSerdes : 
  module MemSerdes : 
    input clock : Clock
    input reset : UInt<1>
    input wide : {req_cmd : {flip ready : UInt<1>, valid : UInt<1>, bits : {addr : UInt<26>, tag : UInt<7>, rw : UInt<1>}}, req_data : {flip ready : UInt<1>, valid : UInt<1>, bits : {data : UInt<128>}}, flip resp : {flip ready : UInt<1>, valid : UInt<1>, bits : {data : UInt<128>, tag : UInt<7>}}}
    output narrow : {req : {flip ready : UInt<1>, valid : UInt<1>, bits : UInt<16>}, flip resp : {valid : UInt<1>, bits : UInt<16>}}
    
    wide.resp.bits.tag := UInt<1>("h00")
    wide.resp.bits.data := UInt<1>("h00")
    wide.resp.valid := UInt<1>("h00")
    wide.req_data.ready := UInt<1>("h00")
    wide.req_cmd.ready := UInt<1>("h00")
    narrow.req.bits := UInt<1>("h00")
    narrow.req.valid := UInt<1>("h00")
    
    node T_218961 = cat(wide.req_cmd.bits.tag, wide.req_cmd.bits.rw)
    node T_218962 = cat(wide.req_cmd.bits.addr, T_218961)
    reg out_buf : UInt, clock, reset
    reg in_buf : UInt, clock, reset
    reg state : UInt<3>, clock, reset
    onreset state := UInt<3>("h00")
    reg send_cnt : UInt<3>, clock, reset
    onreset send_cnt := UInt<3>("h00")
    reg data_send_cnt : UInt<2>, clock, reset
    onreset data_send_cnt := UInt<2>("h00")
    node T_218984 = eq(send_cnt, UInt<2>("h02"))
    node adone = and(narrow.req.ready, T_218984)
    node T_218987 = eq(send_cnt, UInt<3>("h07"))
    node ddone = and(narrow.req.ready, T_218987)
    
    node T_218989 = and(narrow.req.valid, narrow.req.ready)
    when T_218989 : 
      node T_218991 = addw(send_cnt, UInt<1>("h01"))
      send_cnt := T_218991
      node T_218992 = shr(out_buf, 16)
      out_buf := T_218992
    
    node T_218993 = and(wide.req_cmd.valid, wide.req_cmd.ready)
    when T_218993 : 
      node T_218994 = cat(wide.req_cmd.bits.tag, wide.req_cmd.bits.rw)
      node T_218995 = cat(wide.req_cmd.bits.addr, T_218994)
      out_buf := T_218995
    
    node T_218996 = and(wide.req_data.valid, wide.req_data.ready)
    when T_218996 : out_buf := wide.req_data.bits.data
    node T_218997 = eq(state, UInt<3>("h00"))
    wide.req_cmd.ready := T_218997
    node T_218998 = eq(state, UInt<3>("h03"))
    wide.req_data.ready := T_218998
    node T_218999 = eq(state, UInt<3>("h01"))
    node T_219000 = eq(state, UInt<3>("h02"))
    node T_219001 = or(T_218999, T_219000)
    node T_219002 = eq(state, UInt<3>("h04"))
    node T_219003 = or(T_219001, T_219002)
    narrow.req.valid := T_219003
    narrow.req.bits := out_buf
    
    
    node T_219004 = eq(state, UInt<3>("h00"))
    node T_219005 = and(T_219004, wide.req_cmd.valid)
    when T_219005 : 
      node T_219006 = mux(wide.req_cmd.bits.rw, UInt<3>("h02"), UInt<3>("h01"))
      state := T_219006
    
    
    node T_219007 = eq(state, UInt<3>("h01"))
    node T_219008 = and(T_219007, adone)
    when T_219008 : 
      state := UInt<3>("h00")
      send_cnt := UInt<1>("h00")
    
    
    node T_219010 = eq(state, UInt<3>("h02"))
    node T_219011 = and(T_219010, adone)
    when T_219011 : 
      state := UInt<3>("h03")
      send_cnt := UInt<1>("h00")
    
    
    node T_219013 = eq(state, UInt<3>("h03"))
    node T_219014 = and(T_219013, wide.req_data.valid)
    when T_219014 : state := UInt<3>("h04")
    
    
    node T_219015 = eq(state, UInt<3>("h04"))
    node T_219016 = and(T_219015, ddone)
    when T_219016 : 
      node T_219018 = addw(data_send_cnt, UInt<1>("h01"))
      data_send_cnt := T_219018
      node T_219020 = eq(data_send_cnt, UInt<2>("h03"))
      node T_219021 = mux(T_219020, UInt<3>("h00"), UInt<3>("h03"))
      state := T_219021
      send_cnt := UInt<1>("h00")
    reg recv_cnt : UInt<4>, clock, reset
    onreset recv_cnt := UInt<4>("h00")
    reg data_recv_cnt : UInt<2>, clock, reset
    onreset data_recv_cnt := UInt<2>("h00")
    reg resp_val : UInt<1>, clock, reset
    onreset resp_val := UInt<1>("h00")
    resp_val := UInt<1>("h00")
    when narrow.resp.valid : 
      node T_219031 = addw(recv_cnt, UInt<1>("h01"))
      recv_cnt := T_219031
      
      node T_219033 = eq(recv_cnt, UInt<4>("h08"))
      when T_219033 : 
        recv_cnt := UInt<1>("h00")
        node T_219036 = addw(data_recv_cnt, UInt<1>("h01"))
        data_recv_cnt := T_219036
        resp_val := UInt<1>("h01")
      node T_219038 = bits(in_buf, 143, 16)
      node T_219039 = cat(narrow.resp.bits, T_219038)
      in_buf := T_219039
    wide.resp.valid := resp_val
    wire T_219043 : {data : UInt<128>, tag : UInt<7>}
    T_219043.tag := UInt<1>("h00")
    T_219043.data := UInt<1>("h00")
    node T_219048 = bits(in_buf, 6, 0)
    T_219043.tag := T_219048
    node T_219049 = bits(in_buf, 134, 7)
    T_219043.data := T_219049
    wide.resp.bits <> T_219043
