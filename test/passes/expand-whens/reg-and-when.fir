; RUN: firrtl -i %s -o %s.v -X verilog -p cd 2>&1 | tee %s.out ; cat %s.v | FileCheck %s
; CHECK: out__slow__bits <= fromhost__q$deq$valid ? fromhost__q$deq$bits : tohost__q$deq$bits;
circuit Top : 
  module Top : 
    input clock : Clock
    input reset : UInt<1>
    input out_fast : {flip ready : UInt<1>, valid : UInt<1>, bits : UInt<17>}
    output out_slow : {flip ready : UInt<1>, valid : UInt<1>, bits : UInt<17>}
    output in_fast : {flip ready : UInt<1>, valid : UInt<1>, bits : UInt<17>}
    input in_slow : {flip ready : UInt<1>, valid : UInt<1>, bits : UInt<17>}
    output clk_slow : UInt<1>
    input set_divisor : {valid : UInt<1>, bits : UInt<32>}
    output divisor : UInt<32>
    
    out_fast.ready <= UInt<1>("h00")
    out_slow.bits <= UInt<1>("h00")
    out_slow.valid <= UInt<1>("h00")
    in_fast.bits <= UInt<1>("h00")
    in_fast.valid <= UInt<1>("h00")
    in_slow.ready <= UInt<1>("h00")
    clk_slow <= UInt<1>("h00")
    divisor <= UInt<1>("h00")
    
    inst fromhost_q of Queue_50
    fromhost_q.reset <= UInt<1>("h00")
    fromhost_q.enq.bits <= UInt<1>("h00")
    fromhost_q.enq.valid <= UInt<1>("h00")
    fromhost_q.deq.ready <= UInt<1>("h00")
    fromhost_q.clock <= clock
    fromhost_q.reset <= reset

    inst tohost_q of Queue_51
    tohost_q.reset <= UInt<1>("h00")
    tohost_q.enq.bits <= UInt<1>("h00")
    tohost_q.enq.valid <= UInt<1>("h00")
    tohost_q.deq.ready <= UInt<1>("h00")
    tohost_q.clock <= clock
    tohost_q.reset <= reset

    reg out_slow_bits : UInt<17>, clock, reset
    out_slow_bits <= tohost_q.deq.bits
    when fromhost_q.deq.valid : 
      out_slow_bits <= fromhost_q.deq.bits
    
  module Queue_50 : 
    input clock : Clock
    input reset : UInt<1>
    input enq : {flip ready : UInt<1>, valid : UInt<1>, bits : UInt<17>}
    output deq : {flip ready : UInt<1>, valid : UInt<1>, bits : UInt<17>}
    output count : UInt<1>
    
    enq.ready <= UInt<1>("h00")
    deq.bits <= UInt<1>("h00")
    deq.valid <= UInt<1>("h00")
    count <= UInt<1>("h00")
    
    cmem ram : UInt<17>[1], clock
    reg maybe_full : UInt<1>, clock, reset
    onreset maybe_full <= UInt<1>("h00")
    node ptr_match = eq(UInt<1>("h00"), UInt<1>("h00"))
    node T_115167 = eq(maybe_full, UInt<1>("h00"))
    node empty = and(ptr_match, T_115167)
    node full = and(ptr_match, maybe_full)
    node maybe_flow = and(UInt<1>("h00"), empty)
    node do_flow = and(maybe_flow, deq.ready)
    node T_115173 = and(enq.ready, enq.valid)
    node T_115175 = eq(do_flow, UInt<1>("h00"))
    node do_enq = and(T_115173, T_115175)
    node T_115177 = and(deq.ready, deq.valid)
    node T_115179 = eq(do_flow, UInt<1>("h00"))
    node do_deq = and(T_115177, T_115179)
    when do_enq :
      infer accessor T_115181 = ram[UInt<1>("h00")]
      T_115181 <= enq.bits
      skip
    when do_deq :
      skip
    node T_115184 = neq(do_enq, do_deq)
    when T_115184 :
      maybe_full <= do_enq
      skip
    node T_115186 = eq(empty, UInt<1>("h00"))
    node T_115188 = and(UInt<1>("h00"), enq.valid)
    node T_115189 = or(T_115186, T_115188)
    deq.valid <= T_115189
    node T_115191 = eq(full, UInt<1>("h00"))
    node T_115193 = and(UInt<1>("h00"), deq.ready)
    node T_115194 = or(T_115191, T_115193)
    enq.ready <= T_115194
    infer accessor T_115195 = ram[UInt<1>("h00")]
    wire T_115197 : UInt<17>
    T_115197 <= T_115195
    when maybe_flow :
      T_115197 <= enq.bits
      skip
    deq.bits <= T_115197
    node ptr_diff = subw(UInt<1>("h00"), UInt<1>("h00"))
    node T_115199 = and(maybe_full, ptr_match)
    node T_115200 = T_115199
    node T_115201 = cat(T_115200, ptr_diff)
    count <= T_115201
    
  module Queue_51 : 
    input clock : Clock
    input reset : UInt<1>
    input enq : {flip ready : UInt<1>, valid : UInt<1>, bits : UInt<17>}
    output deq : {flip ready : UInt<1>, valid : UInt<1>, bits : UInt<17>}
    output count : UInt<1>
    
    enq.ready <= UInt<1>("h00")
    deq.bits <= UInt<1>("h00")
    deq.valid <= UInt<1>("h00")
    count <= UInt<1>("h00")
    
    cmem ram : UInt<17>[1], clock
    reg maybe_full : UInt<1>, clock, reset
    onreset maybe_full <= UInt<1>("h00")
    node ptr_match = eq(UInt<1>("h00"), UInt<1>("h00"))
    node T_115235 = eq(maybe_full, UInt<1>("h00"))
    node empty = and(ptr_match, T_115235)
    node full = and(ptr_match, maybe_full)
    node maybe_flow = and(UInt<1>("h00"), empty)
    node do_flow = and(maybe_flow, deq.ready)
    node T_115241 = and(enq.ready, enq.valid)
    node T_115243 = eq(do_flow, UInt<1>("h00"))
    node do_enq = and(T_115241, T_115243)
    node T_115245 = and(deq.ready, deq.valid)
    node T_115247 = eq(do_flow, UInt<1>("h00"))
    node do_deq = and(T_115245, T_115247)
    when do_enq :
      infer accessor T_115249 = ram[UInt<1>("h00")]
      T_115249 <= enq.bits
      skip
    when do_deq :
      skip
    node T_115252 = neq(do_enq, do_deq)
    when T_115252 :
      maybe_full <= do_enq
      skip
    node T_115254 = eq(empty, UInt<1>("h00"))
    node T_115256 = and(UInt<1>("h00"), enq.valid)
    node T_115257 = or(T_115254, T_115256)
    deq.valid <= T_115257
    node T_115259 = eq(full, UInt<1>("h00"))
    node T_115261 = and(UInt<1>("h00"), deq.ready)
    node T_115262 = or(T_115259, T_115261)
    enq.ready <= T_115262
    infer accessor T_115263 = ram[UInt<1>("h00")]
    wire T_115265 : UInt<17>
    T_115265 <= T_115263
    when maybe_flow :
      T_115265 <= enq.bits
      skip
    deq.bits <= T_115265
    node ptr_diff = subw(UInt<1>("h00"), UInt<1>("h00"))
    node T_115267 = and(maybe_full, ptr_match)
    node T_115268 = T_115267
    node T_115269 = cat(T_115268, ptr_diff)
    count <= T_115269

