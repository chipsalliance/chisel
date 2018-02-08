;buildInfoPackage: chisel3, version: 3.1-SNAPSHOT, scalaVersion: 2.11.12, sbtVersion: 1.0.4
circuit QueueTestanonfun1anonfunapplymcVsp1anon1 : 
  module Queue : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip enq : {flip ready : UInt<1>, valid : UInt<1>, bits : UInt<8>}, deq : {flip ready : UInt<1>, valid : UInt<1>, bits : UInt<8>}, count : UInt<2>}
    
    cmem ram : UInt<8>[2] @[Decoupled.scala 214:24]
    reg value : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Counter.scala 26:33]
    reg value_1 : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Counter.scala 26:33]
    reg maybe_full : UInt<1>, clock with : (reset => (reset, UInt<1>("h00"))) @[Decoupled.scala 217:35]
    node _T_23 = eq(value, value_1) @[Decoupled.scala 219:41]
    node _T_25 = eq(maybe_full, UInt<1>("h00")) @[Decoupled.scala 220:36]
    node empty = and(_T_23, _T_25) @[Decoupled.scala 220:33]
    node _T_26 = and(_T_23, maybe_full) @[Decoupled.scala 221:32]
    node _T_27 = and(io.enq.ready, io.enq.valid) @[Decoupled.scala 37:37]
    wire do_enq : UInt<1>
    do_enq <= _T_27
    node _T_29 = and(io.deq.ready, io.deq.valid) @[Decoupled.scala 37:37]
    wire do_deq : UInt<1>
    do_deq <= _T_29
    when do_enq : @[Decoupled.scala 225:17]
      infer mport _T_31 = ram[value], clock @[Decoupled.scala 226:8]
      _T_31 <= io.enq.bits @[Decoupled.scala 226:24]
      node wrap = eq(value, UInt<1>("h01")) @[Counter.scala 34:24]
      node _T_34 = add(value, UInt<1>("h01")) @[Counter.scala 35:22]
      node _T_35 = tail(_T_34, 1) @[Counter.scala 35:22]
      value <= _T_35 @[Counter.scala 35:13]
      skip @[Decoupled.scala 225:17]
    when do_deq : @[Decoupled.scala 229:17]
      node wrap_1 = eq(value_1, UInt<1>("h01")) @[Counter.scala 34:24]
      node _T_38 = add(value_1, UInt<1>("h01")) @[Counter.scala 35:22]
      node _T_39 = tail(_T_38, 1) @[Counter.scala 35:22]
      value_1 <= _T_39 @[Counter.scala 35:13]
      skip @[Decoupled.scala 229:17]
    node _T_40 = neq(do_enq, do_deq) @[Decoupled.scala 232:16]
    when _T_40 : @[Decoupled.scala 232:28]
      maybe_full <= do_enq @[Decoupled.scala 233:16]
      skip @[Decoupled.scala 232:28]
    node _T_42 = eq(empty, UInt<1>("h00")) @[Decoupled.scala 236:19]
    io.deq.valid <= _T_42 @[Decoupled.scala 236:16]
    node _T_44 = eq(_T_26, UInt<1>("h00")) @[Decoupled.scala 237:19]
    io.enq.ready <= _T_44 @[Decoupled.scala 237:16]
    infer mport _T_45 = ram[value_1], clock @[Decoupled.scala 238:21]
    io.deq.bits <= _T_45 @[Decoupled.scala 238:15]
    node _T_46 = sub(value, value_1) @[Decoupled.scala 253:40]
    node _T_47 = asUInt(_T_46) @[Decoupled.scala 253:40]
    node _T_48 = tail(_T_47, 1) @[Decoupled.scala 253:40]
    node _T_49 = and(maybe_full, _T_23) @[Decoupled.scala 255:32]
    node _T_50 = cat(_T_49, _T_48) @[Cat.scala 30:58]
    io.count <= _T_50 @[Decoupled.scala 255:14]
    
  module QueueTestanonfun1anonfunapplymcVsp1anon1 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip in : {flip ready : UInt<1>, valid : UInt<1>, bits : UInt<8>}, out : {flip ready : UInt<1>, valid : UInt<1>, bits : UInt<8>}}
    
    inst Queue of Queue @[Decoupled.scala 285:19]
    Queue.clock <= clock
    Queue.reset <= reset
    Queue.io.enq.valid <= io.in.valid @[Decoupled.scala 286:20]
    Queue.io.enq.bits <= io.in.bits @[Decoupled.scala 287:19]
    io.in.ready <= Queue.io.enq.ready @[Decoupled.scala 288:15]
    io.out.bits <= Queue.io.deq.bits @[QueueTester.scala 19:14]
    io.out.valid <= Queue.io.deq.valid @[QueueTester.scala 19:14]
    Queue.io.deq.ready <= io.out.ready @[QueueTester.scala 19:14]
    
