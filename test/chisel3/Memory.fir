; RUN: firrtl -i %s -o %s.flo -x X -p c | tee %s.out | FileCheck %s
; CHECK: Done!

circuit Memory : 
  module Queue : 
    output count : UInt<3>
    input enq : {valid : UInt<1>, flip ready : UInt<1>, bits : {mask : UInt<4>, rw : UInt<1>, tag : UInt<5>, addr : UInt<32>}}
    output deq : {valid : UInt<1>, flip ready : UInt<1>, bits : {mask : UInt<4>, rw : UInt<1>, tag : UInt<5>, addr : UInt<32>}}
    
    mem ram : {mask : UInt<4>, rw : UInt<1>, tag : UInt<5>, addr : UInt<32>}[4]
    reg T_292 : UInt<2>
    on-reset T_292 := UInt<2>(0)
    reg T_293 : UInt<2>
    on-reset T_293 := UInt<2>(0)
    reg maybe_full : UInt<1>
    on-reset maybe_full := UInt<1>(0)
    node ptr_match = eq(T_292, T_293)
    node T_294 = bit-not(maybe_full)
    node empty = bit-and(ptr_match, T_294)
    node full = bit-and(ptr_match, maybe_full)
    node maybe_flow = bit-and(UInt<1>(0), empty)
    node do_flow = bit-and(maybe_flow, deq.ready)
    node T_295 = bit-and(enq.ready, enq.valid)
    node T_296 = bit-not(do_flow)
    node do_enq = bit-and(T_295, T_296)
    node T_297 = bit-and(deq.ready, deq.valid)
    node T_298 = bit-not(do_flow)
    node do_deq = bit-and(T_297, T_298)
    when do_enq : 
      accessor T_299 = ram[T_292]
      T_299 := enq.bits
      node T_300 = eq(T_292, UInt<2>(3))
      node T_301 = bit-and(UInt<1>(0), T_300)
      node T_302 = add-wrap(T_292, UInt<2>(1))
      node T_303 = mux(T_301, UInt<2>(0), T_302)
      T_292 := T_303
    when do_deq : 
      node T_304 = eq(T_293, UInt<2>(3))
      node T_305 = bit-and(UInt<1>(0), T_304)
      node T_306 = add-wrap(T_293, UInt<2>(1))
      node T_307 = mux(T_305, UInt<2>(0), T_306)
      T_293 := T_307
    node T_308 = neq(do_enq, do_deq)
    when T_308 : maybe_full := do_enq
    node T_309 = bit-not(empty)
    node T_310 = bit-and(UInt<1>(0), enq.valid)
    node T_311 = bit-or(T_309, T_310)
    deq.valid := T_311
    node T_312 = bit-not(full)
    node T_313 = bit-and(UInt<1>(0), deq.ready)
    node T_314 = bit-or(T_312, T_313)
    enq.ready := T_314
    accessor T_315 = ram[T_293]
    wire T_316 : {mask : UInt<4>, rw : UInt<1>, tag : UInt<5>, addr : UInt<32>}
    node T_317 = mux(maybe_flow, enq.bits.mask, T_315.mask)
    T_316.mask := T_317
    node T_318 = mux(maybe_flow, enq.bits.rw, T_315.rw)
    T_316.rw := T_318
    node T_319 = mux(maybe_flow, enq.bits.tag, T_315.tag)
    T_316.tag := T_319
    node T_320 = mux(maybe_flow, enq.bits.addr, T_315.addr)
    T_316.addr := T_320
    deq.bits := T_316
    node ptr_diff = sub-wrap(T_292, T_293)
    node T_321 = bit-and(maybe_full, ptr_match)
    node T_322 = cat(T_321, ptr_diff)
    count := Pad(T_322,3)
  module Queue_228 : 
    output count : UInt<3>
    input enq : {valid : UInt<1>, flip ready : UInt<1>, bits : {data : UInt<32>}}
    output deq : {valid : UInt<1>, flip ready : UInt<1>, bits : {data : UInt<32>}}
    
    mem ram : {data : UInt<32>}[4]
    reg T_323 : UInt<2>
    on-reset T_323 := UInt<2>(0)
    reg T_324 : UInt<2>
    on-reset T_324 := UInt<2>(0)
    reg maybe_full : UInt<1>
    on-reset maybe_full := UInt<1>(0)
    node ptr_match = eq(T_323, T_324)
    node T_325 = bit-not(maybe_full)
    node empty = bit-and(ptr_match, T_325)
    node full = bit-and(ptr_match, maybe_full)
    node maybe_flow = bit-and(UInt<1>(0), empty)
    node do_flow = bit-and(maybe_flow, deq.ready)
    node T_326 = bit-and(enq.ready, enq.valid)
    node T_327 = bit-not(do_flow)
    node do_enq = bit-and(T_326, T_327)
    node T_328 = bit-and(deq.ready, deq.valid)
    node T_329 = bit-not(do_flow)
    node do_deq = bit-and(T_328, T_329)
    when do_enq : 
      accessor T_330 = ram[T_323]
      T_330 := enq.bits
      node T_331 = eq(T_323, UInt<2>(3))
      node T_332 = bit-and(UInt<1>(0), T_331)
      node T_333 = add-wrap(T_323, UInt<2>(1))
      node T_334 = mux(T_332, UInt<2>(0), T_333)
      T_323 := T_334
    when do_deq : 
      node T_335 = eq(T_324, UInt<2>(3))
      node T_336 = bit-and(UInt<1>(0), T_335)
      node T_337 = add-wrap(T_324, UInt<2>(1))
      node T_338 = mux(T_336, UInt<2>(0), T_337)
      T_324 := T_338
    node T_339 = neq(do_enq, do_deq)
    when T_339 : maybe_full := do_enq
    node T_340 = bit-not(empty)
    node T_341 = bit-and(UInt<1>(0), enq.valid)
    node T_342 = bit-or(T_340, T_341)
    deq.valid := T_342
    node T_343 = bit-not(full)
    node T_344 = bit-and(UInt<1>(0), deq.ready)
    node T_345 = bit-or(T_343, T_344)
    enq.ready := T_345
    accessor T_346 = ram[T_324]
    wire T_347 : {data : UInt<32>}
    node T_348 = mux(maybe_flow, enq.bits.data, T_346.data)
    T_347.data := T_348
    deq.bits := T_347
    node ptr_diff = sub-wrap(T_323, T_324)
    node T_349 = bit-and(maybe_full, ptr_match)
    node T_350 = cat(T_349, ptr_diff)
    count := Pad(T_350,3)
  module Memory : 
    output stall : UInt<1>
    output memory : {flip resp : {valid : UInt<1>, flip ready : UInt<1>, bits : {data : UInt<32>, tag : UInt<5>}}, req_data : {valid : UInt<1>, flip ready : UInt<1>, bits : {data : UInt<32>}}, req_cmd : {valid : UInt<1>, flip ready : UInt<1>, bits : {mask : UInt<4>, rw : UInt<1>, tag : UInt<5>, addr : UInt<32>}}}
    input icache : {re : UInt<1>, addr : UInt<32>, din : UInt<32>, flip dout : UInt<32>, we : UInt<4>}
    input dcache : {re : UInt<1>, addr : UInt<32>, din : UInt<32>, flip dout : UInt<32>, we : UInt<4>}
    
    inst memReqCmdQueue of Queue
    inst memReqDataQueue of Queue_228
    reg state : UInt<1>
    on-reset state := UInt<1>(0)
    reg tag : UInt<5>
    on-reset tag := UInt<5>(0)
    node T_351 = eq(state, UInt<1>(0))
    node T_352 = bit-or(icache.re, dcache.re)
    node T_353 = eq(dcache.we, UInt<4>(0))
    node T_354 = bit-not(T_353)
    node T_355 = bit-or(T_352, T_354)
    node cpuReq = bit-and(T_351, T_355)
    node T_356 = bits(icache.addr, 31, 2)
    node iaddr = cat(T_356, UInt<2>(0))
    node T_357 = bits(dcache.addr, 31, 2)
    node daddr = cat(T_357, UInt<2>(0))
    reg idata : UInt
    reg ddata : UInt
    reg ire : UInt<1>
    reg dre : UInt<1>
    icache.dout := Pad(idata,?)
    dcache.dout := Pad(ddata,?)
    memReqCmdQueue.deq := memory.req_cmd
    memReqDataQueue.deq := memory.req_data
    memory.resp.ready := UInt<1>(0)
    node T_358 = eq(state, UInt<1>(1))
    node T_359 = bit-not(memReqCmdQueue.enq.ready)
    node T_360 = bit-or(T_358, T_359)
    node T_361 = bit-not(memReqDataQueue.enq.ready)
    node T_362 = bit-or(T_360, T_361)
    stall := T_362
    node T_363 = eq(dcache.we, UInt<4>(0))
    node T_364 = bit-not(T_363)
    memReqCmdQueue.enq.bits.rw := T_364
    memReqCmdQueue.enq.bits.tag := tag
    node T_365 = eq(dcache.we, UInt<4>(0))
    node T_366 = bit-not(T_365)
    node T_367 = bit-not(icache.re)
    node T_368 = bit-or(T_366, T_367)
    node T_369 = mux(T_368, daddr, iaddr)
    memReqCmdQueue.enq.bits.addr := T_369
    memReqCmdQueue.enq.bits.mask := dcache.we
    node T_370 = bit-and(memReqDataQueue.enq.ready, cpuReq)
    memReqCmdQueue.enq.valid := T_370
    memReqDataQueue.enq.bits.data := dcache.din
    node T_371 = bit-and(memReqCmdQueue.enq.ready, cpuReq)
    node T_372 = eq(dcache.we, UInt<4>(0))
    node T_373 = bit-not(T_372)
    node T_374 = bit-and(T_371, T_373)
    memReqDataQueue.enq.valid := T_374
    node T_375 = eq(UInt<1>(0), state)
    when T_375 : 
      node T_376 = bit-or(icache.re, dcache.re)
      node T_377 = eq(dcache.we, UInt<4>(0))
      node T_378 = bit-not(T_377)
      node T_379 = bit-not(T_378)
      node T_380 = bit-and(T_376, T_379)
      node T_381 = bit-and(T_380, memReqCmdQueue.enq.ready)
      when T_381 : 
        ire := icache.re
        dre := dcache.re
        state := UInt<1>(1)
    node T_382 = eq(UInt<1>(1), state)
    when T_382 : 
      memory.resp.ready := UInt<1>(1)
      node T_383 = eq(memory.resp.bits.tag, tag)
      node T_384 = bit-and(memory.resp.valid, T_383)
      when T_384 : 
        state := UInt<1>(0)
        node T_385 = add-wrap(tag, UInt<5>(1))
        tag := T_385
        memory.resp.ready := UInt<1>(0)
        when ire : idata := Pad(memory.resp.bits.data,?)
        when dre : ddata := Pad(memory.resp.bits.data,?)
