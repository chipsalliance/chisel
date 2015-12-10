; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Expression out.grant is used as a sink but can only be used as a source.

circuit ClientTileLinkIOWrapper_49 : 
  module ClientTileLinkIOWrapper_49 : 
    input clock : Clock
    input reset : UInt<1>
    input in : {acquire : {flip ready : UInt<1>, valid : UInt<1>, bits : {addr_block : UInt<26>, client_xact_id : UInt<4>, addr_beat : UInt<2>, data : UInt<128>, is_builtin_type : UInt<1>, a_type : UInt<3>, union : UInt<17>}}, flip grant : {flip ready : UInt<1>, valid : UInt<1>, bits : {addr_beat : UInt<2>, data : UInt<128>, client_xact_id : UInt<4>, manager_xact_id : UInt<1>, is_builtin_type : UInt<1>, g_type : UInt<4>}}}
    output out : {acquire : {flip ready : UInt<1>, valid : UInt<1>, bits : {addr_block : UInt<26>, client_xact_id : UInt<4>, addr_beat : UInt<2>, data : UInt<128>, is_builtin_type : UInt<1>, a_type : UInt<3>, union : UInt<17>}}, flip grant : {flip ready : UInt<1>, valid : UInt<1>, bits : {addr_beat : UInt<2>, data : UInt<128>, client_xact_id : UInt<4>, manager_xact_id : UInt<1>, is_builtin_type : UInt<1>, g_type : UInt<4>}}, flip probe : {flip ready : UInt<1>, valid : UInt<1>, bits : {addr_block : UInt<26>, p_type : UInt<2>}}, release : {flip ready : UInt<1>, valid : UInt<1>, bits : {addr_block : UInt<26>, client_xact_id : UInt<4>, addr_beat : UInt<2>, data : UInt<128>, r_type : UInt<3>, voluntary : UInt<1>}}}
    
    in.grant.bits.g_type <= UInt<1>("h00")
    in.grant.bits.is_builtin_type <= UInt<1>("h00")
    in.grant.bits.manager_xact_id <= UInt<1>("h00")
    in.grant.bits.client_xact_id <= UInt<1>("h00")
    in.grant.bits.data <= UInt<1>("h00")
    in.grant.bits.addr_beat <= UInt<1>("h00")
    in.grant.valid <= UInt<1>("h00")
    in.acquire.ready <= UInt<1>("h00")
    out.release.bits.voluntary <= UInt<1>("h00")
    out.release.bits.r_type <= UInt<1>("h00")
    out.release.bits.data <= UInt<1>("h00")
    out.release.bits.addr_beat <= UInt<1>("h00")
    out.release.bits.client_xact_id <= UInt<1>("h00")
    out.release.bits.addr_block <= UInt<1>("h00")
    out.release.valid <= UInt<1>("h00")
    out.probe.ready <= UInt<1>("h00")
    out.grant.ready <= UInt<1>("h00")
    out.acquire.bits.union <= UInt<1>("h00")
    out.acquire.bits.a_type <= UInt<1>("h00")
    out.acquire.bits.is_builtin_type <= UInt<1>("h00")
    out.acquire.bits.data <= UInt<1>("h00")
    out.acquire.bits.addr_beat <= UInt<1>("h00")
    out.acquire.bits.client_xact_id <= UInt<1>("h00")
    out.acquire.bits.addr_block <= UInt<1>("h00")
    out.acquire.valid <= UInt<1>("h00")
    
    wire T_173394 : UInt<2>[2]
    T_173394[0] <= UInt<2>("h01")
    T_173394[1] <= UInt<2>("h02")
    wire T_173397 : UInt<2>[2]
    T_173397[0] <= UInt<2>("h01")
    T_173397[1] <= UInt<2>("h02")
    wire T_173399 : UInt<2>[1]
    T_173399[0] <= UInt<2>("h02")
    wire T_173439 : UInt<2>[2]
    T_173439[0] <= UInt<2>("h01")
    T_173439[1] <= UInt<2>("h02")
    wire T_173442 : UInt<2>[2]
    T_173442[0] <= UInt<2>("h01")
    T_173442[1] <= UInt<2>("h02")
    wire T_173444 : UInt<2>[1]
    T_173444[0] <= UInt<2>("h02")
    wire T_173483 : UInt<2>[2]
    T_173483[0] <= UInt<2>("h01")
    T_173483[1] <= UInt<2>("h02")
    wire T_173486 : UInt<2>[2]
    T_173486[0] <= UInt<2>("h01")
    T_173486[1] <= UInt<2>("h02")
    wire T_173488 : UInt<2>[1]
    T_173488[0] <= UInt<2>("h02")
    wire T_173537 : UInt<2>[2]
    T_173537[0] <= UInt<2>("h01")
    T_173537[1] <= UInt<2>("h02")
    wire T_173540 : UInt<2>[2]
    T_173540[0] <= UInt<2>("h01")
    T_173540[1] <= UInt<2>("h02")
    wire T_173542 : UInt<2>[1]
    T_173542[0] <= UInt<2>("h02")
    wire T_173588 : UInt<2>[2]
    T_173588[0] <= UInt<2>("h01")
    T_173588[1] <= UInt<2>("h02")
    wire T_173591 : UInt<2>[2]
    T_173591[0] <= UInt<2>("h01")
    T_173591[1] <= UInt<2>("h02")
    wire T_173593 : UInt<2>[1]
    T_173593[0] <= UInt<2>("h02")
    wire T_173639 : UInt<2>[2]
    T_173639[0] <= UInt<2>("h01")
    T_173639[1] <= UInt<2>("h02")
    wire T_173642 : UInt<2>[2]
    T_173642[0] <= UInt<2>("h01")
    T_173642[1] <= UInt<2>("h02")
    wire T_173644 : UInt<2>[1]
    T_173644[0] <= UInt<2>("h02")
    wire T_173692 : UInt<2>[2]
    T_173692[0] <= UInt<2>("h01")
    T_173692[1] <= UInt<2>("h02")
    wire T_173695 : UInt<2>[2]
    T_173695[0] <= UInt<2>("h01")
    T_173695[1] <= UInt<2>("h02")
    wire T_173697 : UInt<2>[1]
    T_173697[0] <= UInt<2>("h02")
    wire T_173742 : UInt<2>[2]
    T_173742[0] <= UInt<2>("h01")
    T_173742[1] <= UInt<2>("h02")
    wire T_173745 : UInt<2>[2]
    T_173745[0] <= UInt<2>("h01")
    T_173745[1] <= UInt<2>("h02")
    wire T_173747 : UInt<2>[1]
    T_173747[0] <= UInt<2>("h02")
    wire T_173795 : UInt<2>[2]
    T_173795[0] <= UInt<2>("h01")
    T_173795[1] <= UInt<2>("h02")
    wire T_173798 : UInt<2>[2]
    T_173798[0] <= UInt<2>("h01")
    T_173798[1] <= UInt<2>("h02")
    wire T_173800 : UInt<2>[1]
    T_173800[0] <= UInt<2>("h02")
    wire T_173845 : UInt<2>[2]
    T_173845[0] <= UInt<2>("h01")
    T_173845[1] <= UInt<2>("h02")
    wire T_173848 : UInt<2>[2]
    T_173848[0] <= UInt<2>("h01")
    T_173848[1] <= UInt<2>("h02")
    wire T_173850 : UInt<2>[1]
    T_173850[0] <= UInt<2>("h02")
    wire T_173895 : UInt<2>[2]
    T_173895[0] <= UInt<2>("h01")
    T_173895[1] <= UInt<2>("h02")
    wire T_173898 : UInt<2>[2]
    T_173898[0] <= UInt<2>("h01")
    T_173898[1] <= UInt<2>("h02")
    wire T_173900 : UInt<2>[1]
    T_173900[0] <= UInt<2>("h02")
    wire T_173939 : UInt<2>[2]
    T_173939[0] <= UInt<2>("h01")
    T_173939[1] <= UInt<2>("h02")
    wire T_173942 : UInt<2>[2]
    T_173942[0] <= UInt<2>("h01")
    T_173942[1] <= UInt<2>("h02")
    wire T_173944 : UInt<2>[1]
    T_173944[0] <= UInt<2>("h02")
    wire T_173993 : UInt<2>[2]
    T_173993[0] <= UInt<2>("h01")
    T_173993[1] <= UInt<2>("h02")
    wire T_173996 : UInt<2>[2]
    T_173996[0] <= UInt<2>("h01")
    T_173996[1] <= UInt<2>("h02")
    wire T_173998 : UInt<2>[1]
    T_173998[0] <= UInt<2>("h02")
    wire T_174044 : UInt<2>[2]
    T_174044[0] <= UInt<2>("h01")
    T_174044[1] <= UInt<2>("h02")
    wire T_174047 : UInt<2>[2]
    T_174047[0] <= UInt<2>("h01")
    T_174047[1] <= UInt<2>("h02")
    wire T_174049 : UInt<2>[1]
    T_174049[0] <= UInt<2>("h02")
    wire T_174095 : UInt<2>[2]
    T_174095[0] <= UInt<2>("h01")
    T_174095[1] <= UInt<2>("h02")
    wire T_174098 : UInt<2>[2]
    T_174098[0] <= UInt<2>("h01")
    T_174098[1] <= UInt<2>("h02")
    wire T_174100 : UInt<2>[1]
    T_174100[0] <= UInt<2>("h02")
    wire T_174148 : UInt<2>[2]
    T_174148[0] <= UInt<2>("h01")
    T_174148[1] <= UInt<2>("h02")
    wire T_174151 : UInt<2>[2]
    T_174151[0] <= UInt<2>("h01")
    T_174151[1] <= UInt<2>("h02")
    wire T_174153 : UInt<2>[1]
    T_174153[0] <= UInt<2>("h02")
    wire T_174198 : UInt<2>[2]
    T_174198[0] <= UInt<2>("h01")
    T_174198[1] <= UInt<2>("h02")
    wire T_174201 : UInt<2>[2]
    T_174201[0] <= UInt<2>("h01")
    T_174201[1] <= UInt<2>("h02")
    wire T_174203 : UInt<2>[1]
    T_174203[0] <= UInt<2>("h02")
    wire T_174251 : UInt<2>[2]
    T_174251[0] <= UInt<2>("h01")
    T_174251[1] <= UInt<2>("h02")
    wire T_174254 : UInt<2>[2]
    T_174254[0] <= UInt<2>("h01")
    T_174254[1] <= UInt<2>("h02")
    wire T_174256 : UInt<2>[1]
    T_174256[0] <= UInt<2>("h02")
    wire T_174301 : UInt<2>[2]
    T_174301[0] <= UInt<2>("h01")
    T_174301[1] <= UInt<2>("h02")
    wire T_174304 : UInt<2>[2]
    T_174304[0] <= UInt<2>("h01")
    T_174304[1] <= UInt<2>("h02")
    wire T_174306 : UInt<2>[1]
    T_174306[0] <= UInt<2>("h02")
    wire T_174351 : UInt<2>[2]
    T_174351[0] <= UInt<2>("h01")
    T_174351[1] <= UInt<2>("h02")
    wire T_174354 : UInt<2>[2]
    T_174354[0] <= UInt<2>("h01")
    T_174354[1] <= UInt<2>("h02")
    wire T_174356 : UInt<2>[1]
    T_174356[0] <= UInt<2>("h02")
    wire T_174395 : UInt<2>[2]
    T_174395[0] <= UInt<2>("h01")
    T_174395[1] <= UInt<2>("h02")
    wire T_174398 : UInt<2>[2]
    T_174398[0] <= UInt<2>("h01")
    T_174398[1] <= UInt<2>("h02")
    wire T_174400 : UInt<2>[1]
    T_174400[0] <= UInt<2>("h02")
    wire T_174449 : UInt<2>[2]
    T_174449[0] <= UInt<2>("h01")
    T_174449[1] <= UInt<2>("h02")
    wire T_174452 : UInt<2>[2]
    T_174452[0] <= UInt<2>("h01")
    T_174452[1] <= UInt<2>("h02")
    wire T_174454 : UInt<2>[1]
    T_174454[0] <= UInt<2>("h02")
    wire T_174500 : UInt<2>[2]
    T_174500[0] <= UInt<2>("h01")
    T_174500[1] <= UInt<2>("h02")
    wire T_174503 : UInt<2>[2]
    T_174503[0] <= UInt<2>("h01")
    T_174503[1] <= UInt<2>("h02")
    wire T_174505 : UInt<2>[1]
    T_174505[0] <= UInt<2>("h02")
    wire T_174551 : UInt<2>[2]
    T_174551[0] <= UInt<2>("h01")
    T_174551[1] <= UInt<2>("h02")
    wire T_174554 : UInt<2>[2]
    T_174554[0] <= UInt<2>("h01")
    T_174554[1] <= UInt<2>("h02")
    wire T_174556 : UInt<2>[1]
    T_174556[0] <= UInt<2>("h02")
    wire T_174604 : UInt<2>[2]
    T_174604[0] <= UInt<2>("h01")
    T_174604[1] <= UInt<2>("h02")
    wire T_174607 : UInt<2>[2]
    T_174607[0] <= UInt<2>("h01")
    T_174607[1] <= UInt<2>("h02")
    wire T_174609 : UInt<2>[1]
    T_174609[0] <= UInt<2>("h02")
    wire T_174654 : UInt<2>[2]
    T_174654[0] <= UInt<2>("h01")
    T_174654[1] <= UInt<2>("h02")
    wire T_174657 : UInt<2>[2]
    T_174657[0] <= UInt<2>("h01")
    T_174657[1] <= UInt<2>("h02")
    wire T_174659 : UInt<2>[1]
    T_174659[0] <= UInt<2>("h02")
    wire T_174707 : UInt<2>[2]
    T_174707[0] <= UInt<2>("h01")
    T_174707[1] <= UInt<2>("h02")
    wire T_174710 : UInt<2>[2]
    T_174710[0] <= UInt<2>("h01")
    T_174710[1] <= UInt<2>("h02")
    wire T_174712 : UInt<2>[1]
    T_174712[0] <= UInt<2>("h02")
    wire T_174757 : UInt<2>[2]
    T_174757[0] <= UInt<2>("h01")
    T_174757[1] <= UInt<2>("h02")
    wire T_174760 : UInt<2>[2]
    T_174760[0] <= UInt<2>("h01")
    T_174760[1] <= UInt<2>("h02")
    wire T_174762 : UInt<2>[1]
    T_174762[0] <= UInt<2>("h02")
    wire T_174807 : UInt<2>[2]
    T_174807[0] <= UInt<2>("h01")
    T_174807[1] <= UInt<2>("h02")
    wire T_174810 : UInt<2>[2]
    T_174810[0] <= UInt<2>("h01")
    T_174810[1] <= UInt<2>("h02")
    wire T_174812 : UInt<2>[1]
    T_174812[0] <= UInt<2>("h02")
    wire T_174856 : UInt<2>[2]
    T_174856[0] <= UInt<2>("h01")
    T_174856[1] <= UInt<2>("h02")
    wire T_174859 : UInt<2>[2]
    T_174859[0] <= UInt<2>("h01")
    T_174859[1] <= UInt<2>("h02")
    wire T_174861 : UInt<2>[1]
    T_174861[0] <= UInt<2>("h02")
    wire T_174902 : UInt<2>[2]
    T_174902[0] <= UInt<2>("h01")
    T_174902[1] <= UInt<2>("h02")
    wire T_174905 : UInt<2>[2]
    T_174905[0] <= UInt<2>("h01")
    T_174905[1] <= UInt<2>("h02")
    wire T_174907 : UInt<2>[1]
    T_174907[0] <= UInt<2>("h02")
    wire T_174951 : UInt<2>[2]
    T_174951[0] <= UInt<2>("h01")
    T_174951[1] <= UInt<2>("h02")
    wire T_174954 : UInt<2>[2]
    T_174954[0] <= UInt<2>("h01")
    T_174954[1] <= UInt<2>("h02")
    wire T_174956 : UInt<2>[1]
    T_174956[0] <= UInt<2>("h02")
    wire T_174997 : UInt<2>[2]
    T_174997[0] <= UInt<2>("h01")
    T_174997[1] <= UInt<2>("h02")
    wire T_175000 : UInt<2>[2]
    T_175000[0] <= UInt<2>("h01")
    T_175000[1] <= UInt<2>("h02")
    wire T_175002 : UInt<2>[1]
    T_175002[0] <= UInt<2>("h02")
    wire T_175043 : UInt<2>[2]
    T_175043[0] <= UInt<2>("h01")
    T_175043[1] <= UInt<2>("h02")
    wire T_175046 : UInt<2>[2]
    T_175046[0] <= UInt<2>("h01")
    T_175046[1] <= UInt<2>("h02")
    wire T_175048 : UInt<2>[1]
    T_175048[0] <= UInt<2>("h02")
    wire T_175096 : UInt<2>[2]
    T_175096[0] <= UInt<2>("h01")
    T_175096[1] <= UInt<2>("h02")
    wire T_175099 : UInt<2>[2]
    T_175099[0] <= UInt<2>("h01")
    T_175099[1] <= UInt<2>("h02")
    wire T_175101 : UInt<2>[1]
    T_175101[0] <= UInt<2>("h02")
    wire T_175146 : UInt<2>[2]
    T_175146[0] <= UInt<2>("h01")
    T_175146[1] <= UInt<2>("h02")
    wire T_175149 : UInt<2>[2]
    T_175149[0] <= UInt<2>("h01")
    T_175149[1] <= UInt<2>("h02")
    wire T_175151 : UInt<2>[1]
    T_175151[0] <= UInt<2>("h02")
    out.acquire <- in.acquire
    out.grant <- in.grant
    out.probe.ready <= UInt<1>("h01")
    out.release.valid <= UInt<1>("h00")
