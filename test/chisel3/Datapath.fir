; RUN: firrtl -i %s -o %s.flo -x X -p cd | tee %s.out | FileCheck %s
; CHECK: Done!

circuit Datapath : 
  module ALU : 
    input B : UInt<32>
    output out : UInt<32>
    output sum : UInt<32>
    input A : UInt<32>
    input alu_op : UInt<4>
    
    node T_431 = bits(B, 4, 0)
    wire shamt : UInt<5>
    shamt := T_431
    node T_432 = add-wrap(A, B)
    node T_433 = sub-wrap(A, B)
    wire T_434 : SInt<32>
    T_434 := A
    node T_435 = dshr(T_434, shamt)
    wire T_436 : UInt<32>
    T_436 := T_435
    node T_437 = dshr(A, shamt)
    node T_438 = dshl(A, shamt)
    node T_439 = bits(T_438, 31, 0)
    wire T_440 : SInt<32>
    T_440 := A
    wire T_441 : SInt<32>
    T_441 := B
    node T_442 = lt(T_440, T_441)
    wire T_443 : UInt<1>
    T_443 := T_442
    node T_444 = lt(A, B)
    wire T_445 : UInt<1>
    T_445 := T_444
    node T_446 = bit-and(A, B)
    node T_447 = bit-or(A, B)
    node T_448 = bit-xor(A, B)
    node T_449 = eq(UInt<4>(10), alu_op)
    node T_450 = mux(T_449, A, B)
    node T_451 = eq(UInt<4>(4), alu_op)
    node T_452 = mux(T_451, T_448, T_450)
    node T_453 = eq(UInt<4>(3), alu_op)
    node T_454 = mux(T_453, T_447, T_452)
    node T_455 = eq(UInt<4>(2), alu_op)
    node T_456 = mux(T_455, T_446, T_454)
    node T_457 = eq(UInt<4>(7), alu_op)
    node T_458 = mux(T_457, Pad(T_445,32), T_456)
    node T_459 = eq(UInt<4>(5), alu_op)
    node T_460 = mux(T_459, Pad(T_443,32), T_458)
    node T_461 = eq(UInt<4>(6), alu_op)
    node T_462 = mux(T_461, T_439, T_460)
    node T_463 = eq(UInt<4>(8), alu_op)
    node T_464 = mux(T_463, T_437, T_462)
    node T_465 = eq(UInt<4>(9), alu_op)
    node T_466 = mux(T_465, T_436, T_464)
    node T_467 = eq(UInt<4>(1), alu_op)
    node T_468 = mux(T_467, T_433, T_466)
    node T_469 = eq(UInt<4>(0), alu_op)
    node T_470 = mux(T_469, T_432, T_468)
    out := T_470
    node T_471 = bit(alu_op, 0)
    node T_472 = sub-wrap(UInt<32>(0), B)
    node T_473 = mux(T_471, T_472, B)
    node T_474 = add-wrap(A, T_473)
    sum := T_474
  module BrCond : 
    input rs1 : UInt<32>
    input rs2 : UInt<32>
    output taken : UInt<1>
    input br_type : UInt<3>
    
    node eq = eq(rs1, rs2)
    node neq = bit-not(eq)
    wire T_475 : SInt<32>
    T_475 := rs1
    wire T_476 : SInt<32>
    T_476 := rs2
    node lt = lt(T_475, T_476)
    node ge = bit-not(lt)
    node ltu = lt(rs1, rs2)
    node geu = bit-not(ltu)
    node T_477 = UInt<1>(1)
    node T_478 = UInt<1>(0)
    node T_479 = eq(br_type, UInt<3>(2))
    node T_480 = bit-and(T_479, eq)
    node T_481 = eq(br_type, UInt<3>(6))
    node T_482 = bit-and(T_481, neq)
    node T_483 = bit-or(T_480, T_482)
    node T_484 = eq(br_type, UInt<3>(1))
    node T_485 = bit-and(T_484, lt)
    node T_486 = bit-or(T_483, T_485)
    node T_487 = eq(br_type, UInt<3>(5))
    node T_488 = bit-and(T_487, ge)
    node T_489 = bit-or(T_486, T_488)
    node T_490 = eq(br_type, UInt<3>(0))
    node T_491 = bit-and(T_490, ltu)
    node T_492 = bit-or(T_489, T_491)
    node T_493 = eq(br_type, UInt<3>(4))
    node T_494 = bit-and(T_493, geu)
    node T_495 = bit-or(T_492, T_494)
    taken := T_495
  module RegFile : 
    input raddr1 : UInt<5>
    input raddr2 : UInt<5>
    output rdata1 : UInt<32>
    output rdata2 : UInt<32>
    input wen : UInt<1>
    input waddr : UInt<5>
    input wdata : UInt<32>
    
    mem regs : UInt<32>[32]
    node T_496 = eq(raddr1, UInt<5>(0))
    node T_497 = bit-not(T_496)
    accessor T_498 = regs[raddr1]
    node T_499 = mux(T_497, T_498, UInt<32>(0))
    rdata1 := T_499
    node T_500 = eq(raddr2, UInt<5>(0))
    node T_501 = bit-not(T_500)
    accessor T_502 = regs[raddr2]
    node T_503 = mux(T_501, T_502, UInt<32>(0))
    rdata2 := T_503
    node T_504 = eq(waddr, UInt<5>(0))
    node T_505 = bit-not(T_504)
    node T_506 = bit-and(wen, T_505)
    when T_506 : 
      accessor T_507 = regs[waddr]
      T_507 := wdata
  module ImmGenWire : 
    output out : UInt<32>
    input inst : UInt<32>
    input sel : UInt<3>
    
    node T_508 = bits(inst, 31, 20)
    wire Iimm : SInt<12>
    Iimm := T_508
    node T_509 = bits(inst, 31, 25)
    node T_510 = bits(inst, 11, 7)
    node T_511 = cat(T_509, T_510)
    wire Simm : SInt<12>
    Simm := T_511
    node T_512 = bit(inst, 31)
    node T_513 = bit(inst, 7)
    node T_514 = bits(inst, 30, 25)
    node T_515 = bits(inst, 11, 8)
    node T_516 = cat(T_512, T_513)
    node T_517 = cat(T_515, UInt<1>(0))
    node T_518 = cat(T_514, T_517)
    node T_519 = cat(T_516, T_518)
    wire Bimm : SInt<1>
    Bimm := T_519
    node T_520 = bits(inst, 31, 12)
    node T_521 = cat(T_520, UInt<12>(0))
    wire Uimm : SInt<32>
    Uimm := T_521
    node T_522 = bit(inst, 31)
    node T_523 = bits(inst, 19, 12)
    node T_524 = bit(inst, 20)
    node T_525 = bits(inst, 30, 25)
    node T_526 = bits(inst, 24, 21)
    node T_527 = cat(T_523, T_524)
    node T_528 = cat(T_522, T_527)
    node T_529 = cat(T_526, UInt<1>(0))
    node T_530 = cat(T_525, T_529)
    node T_531 = cat(T_528, T_530)
    wire Jimm : SInt<1>
    Jimm := T_531
    node T_532 = bits(inst, 19, 15)
    wire T_533 : UInt<5>
    T_533 := T_532
    node T_534 = Pad(T_533, 32)
    wire Zimm : SInt<32>
    Zimm := T_534
    node T_535 = eq(UInt<3>(3), sel)
    node T_536 = mux(T_535, Pad(Jimm,32), Zimm)
    node T_537 = eq(UInt<3>(2), sel)
    node T_538 = mux(T_537, Uimm, T_536)
    node T_539 = eq(UInt<3>(4), sel)
    node T_540 = mux(T_539, Pad(Bimm,32), T_538)
    node T_541 = eq(UInt<3>(1), sel)
    node T_542 = mux(T_541, Pad(Simm,32), T_540)
    node T_543 = eq(UInt<3>(0), sel)
    node T_544 = mux(T_543, Pad(Iimm,32), T_542)
    out := T_544
  module CSR : 
    output host : {status : UInt<32>, flip hid : UInt<1>, tohost : UInt<32>}
    input src : UInt<32>
    output data : UInt<32>
    input cmd : UInt<2>
    input addr : UInt<12>
    
    reg reg_tohost : UInt<32>
    on-reset reg_tohost := UInt<32>(0)
    reg reg_status : UInt<32>
    on-reset reg_status := UInt<32>(0)
    host.tohost := reg_tohost
    host.status := reg_status
    node T_545 = eq(UInt<12>(1291), addr)
    node T_546 = mux(T_545, host.hid, UInt<1>(0))
    node T_547 = eq(UInt<12>(1290), addr)
    node T_548 = mux(T_547, reg_status, Pad(T_546,32))
    node T_549 = eq(UInt<12>(1310), addr)
    node T_550 = mux(T_549, reg_tohost, T_548)
    wire T_551 : UInt<32>
    T_551 := T_550
    data := T_551
    node T_552 = eq(cmd, UInt<2>(1))
    when T_552 : 
      node T_553 = eq(addr, UInt<12>(1310))
      when T_553 : reg_tohost := src
      node T_554 = eq(addr, UInt<12>(1290))
      when T_554 : reg_status := src
    node T_555 = eq(cmd, UInt<2>(2))
    node T_556 = neq(src, UInt<32>(0))
    node T_557 = bit-and(T_555, T_556)
    when T_557 : 
      node T_558 = eq(addr, UInt<12>(1310))
      when T_558 : 
        node T_559 = dshl(UInt<1>(1), src)
        node T_560 = bit-or(data, Pad(T_559,32))
        reg_tohost := T_560
      node T_561 = eq(addr, UInt<12>(1290))
      when T_561 : 
        node T_562 = dshl(UInt<1>(1), src)
        node T_563 = bit-or(data, Pad(T_562,32))
        reg_status := T_563
    node T_564 = eq(cmd, UInt<2>(3))
    node T_565 = neq(src, UInt<32>(0))
    node T_566 = bit-and(T_564, T_565)
    when T_566 : 
      node T_567 = eq(addr, UInt<12>(1310))
      when T_567 : 
        node T_568 = dshl(UInt<1>(0), src)
        node T_569 = bit-and(data, Pad(T_568,32))
        reg_tohost := T_569
      node T_570 = eq(addr, UInt<12>(1290))
      when T_570 : 
        node T_571 = dshl(UInt<1>(0), src)
        node T_572 = bit-and(data, Pad(T_571,32))
        reg_status := T_572
  module Datapath : 
    output host : {status : UInt<32>, flip hid : UInt<1>, tohost : UInt<32>}
    input ctrl : {flip inst : UInt<32>, pc_sel : UInt<1>, inst_type : UInt<1>, inst_re : UInt<1>, flip stall : UInt<1>, imm_sel : UInt<3>, wb_en : UInt<1>, wb_sel : UInt<2>, A_sel : UInt<1>, B_sel : UInt<1>, alu_op : UInt<4>, br_type : UInt<3>, data_re : UInt<1>, st_type : UInt<2>, ld_type : UInt<3>, csr_cmd : UInt<2>}
    output icache : {re : UInt<1>, addr : UInt<32>, we : UInt<4>, din : UInt<32>, flip dout : UInt<32>}
    output dcache : {re : UInt<1>, addr : UInt<32>, we : UInt<4>, din : UInt<32>, flip dout : UInt<32>}
    input stall : UInt<1>
    
    inst alu of ALU
    inst brCond of BrCond
    inst regFile of RegFile
    inst immGen of ImmGenWire
    reg fe_inst : UInt<32>
    on-reset fe_inst := UInt<32>(0)
    reg fe_pc : UInt
    reg ew_inst : UInt<32>
    on-reset ew_inst := UInt<32>(0)
    reg ew_pc : UInt
    reg ew_alu : UInt
    node T_573 = sub-wrap(UInt<14>(8192), UInt<14>(4))
    reg pc : UInt<14>
    on-reset pc := T_573
    node T_574 = eq(ctrl.pc_sel, UInt<1>(1))
    node T_575 = bit-or(T_574, brCond.taken)
    node T_576 = add-wrap(pc, UInt<14>(4))
    node iaddr = mux(T_575, alu.sum, Pad(T_576,32))
    node T_577 = eq(ctrl.inst_type, UInt<1>(1))
    node T_578 = bit-or(T_577, brCond.taken)
    node inst = mux(T_578, UInt<32>(19), icache.dout)
    icache.addr := iaddr
    icache.re := ctrl.inst_re
    node T_579 = eq(dcache.we, UInt<4>(0))
    node T_580 = bit-not(T_579)
    node T_581 = bit-not(T_580)
    node T_582 = bit-and(icache.re, T_581)
    node T_583 = mux(T_582, iaddr, Pad(pc,32))
    pc := T_583
    node T_584 = bit-not(stall)
    when T_584 : 
      fe_pc := Pad(pc,?)
      fe_inst := inst
    ctrl.inst := fe_inst
    ctrl.stall := stall
    node rd_addr = bits(fe_inst, 11, 7)
    node rs1_addr = bits(fe_inst, 19, 15)
    node rs2_addr = bits(fe_inst, 24, 20)
    regFile.raddr1 := rs1_addr
    regFile.raddr2 := rs2_addr
    immGen.inst := fe_inst
    immGen.sel := ctrl.imm_sel
    node T_585 = eq(rs1_addr, UInt<5>(0))
    node rs1NotZero = bit-not(T_585)
    node T_586 = eq(rs2_addr, UInt<5>(0))
    node rs2NotZero = bit-not(T_586)
    node T_587 = eq(ctrl.wb_sel, UInt<2>(0))
    node alutype = bit-and(ctrl.wb_en, T_587)
    node ex_rd_addr = bits(ew_inst, 11, 7)
    node T_588 = bit-and(alutype, rs1NotZero)
    node T_589 = eq(rs1_addr, ex_rd_addr)
    node T_590 = bit-and(T_588, T_589)
    node rs1 = mux(T_590, Pad(ew_alu,?), Pad(regFile.rdata1,?))
    node T_591 = bit-and(alutype, rs2NotZero)
    node T_592 = eq(rs2_addr, ex_rd_addr)
    node T_593 = bit-and(T_591, T_592)
    node rs2 = mux(T_593, Pad(ew_alu,?), Pad(regFile.rdata2,?))
    node T_594 = eq(ctrl.A_sel, UInt<1>(0))
    node T_595 = mux(T_594, Pad(rs1,?), Pad(fe_pc,?))
    alu.A := Pad(T_595,?)
    node T_596 = eq(ctrl.B_sel, UInt<1>(0))
    node T_597 = mux(T_596, Pad(rs2,?), Pad(immGen.out,?))
    alu.B := Pad(T_597,?)
    alu.alu_op := ctrl.alu_op
    brCond.rs1 := Pad(rs1,?)
    brCond.rs2 := Pad(rs2,?)
    brCond.br_type := ctrl.br_type
    node T_598 = bit(alu.sum, 1)
    node T_599 = dshl(T_598, UInt<3>(4))
    node T_600 = bit(alu.sum, 0)
    node T_601 = dshl(T_600, UInt<2>(3))
    node woffset = bit-or(T_599, T_601)
    dcache.re := ctrl.data_re
    node T_602 = mux(stall, Pad(ew_alu,?), Pad(alu.sum,?))
    dcache.addr := Pad(T_602,?)
    node T_603 = bits(alu.sum, 1, 0)
    node T_604 = dshl(UInt<2>(3), T_603)
    node T_605 = bits(alu.sum, 1, 0)
    node T_606 = dshl(UInt<1>(1), T_605)
    node T_607 = eq(UInt<2>(2), ctrl.st_type)
    node T_608 = mux(T_607, T_606, UInt<5>(0))
    node T_609 = eq(UInt<2>(1), ctrl.st_type)
    node T_610 = mux(T_609, T_604, Pad(T_608,6))
    node T_611 = eq(UInt<2>(0), ctrl.st_type)
    node T_612 = mux(T_611, UInt<6>(15), T_610)
    node T_613 = mux(stall, UInt<6>(0), T_612)
    dcache.we := T_613
    node T_614 = dshl(rs2, woffset)
    dcache.din := Pad(T_614,?)
    node T_615 = bit-not(stall)
    when T_615 : 
      ew_pc := Pad(fe_pc,?)
      ew_inst := fe_inst
      ew_alu := Pad(alu.out,?)
    node T_616 = bit(ew_alu, 1)
    node T_617 = dshl(T_616, UInt<3>(4))
    node T_618 = bit(ew_alu, 0)
    node T_619 = dshl(T_618, UInt<2>(3))
    node loffset = bit-or(T_617, T_619)
    node lshift = dshr(dcache.dout, loffset)
    node T_620 = bits(lshift, 15, 0)
    wire T_621 : SInt<16>
    T_621 := T_620
    node T_622 = Pad(T_621, 32)
    wire T_623 : UInt<32>
    T_623 := T_622
    node T_624 = bits(lshift, 7, 0)
    wire T_625 : SInt<8>
    T_625 := T_624
    node T_626 = Pad(T_625, 32)
    wire T_627 : UInt<32>
    T_627 := T_626
    node T_628 = bits(lshift, 15, 0)
    wire T_629 : UInt<16>
    T_629 := T_628
    node T_630 = bits(lshift, 7, 0)
    wire T_631 : UInt<8>
    T_631 := T_630
    node T_632 = eq(UInt<3>(4), ctrl.ld_type)
    node T_633 = mux(T_632, Pad(T_631,32), dcache.dout)
    node T_634 = eq(UInt<3>(3), ctrl.ld_type)
    node T_635 = mux(T_634, Pad(T_629,32), T_633)
    node T_636 = eq(UInt<3>(2), ctrl.ld_type)
    node T_637 = mux(T_636, T_627, T_635)
    node T_638 = eq(UInt<3>(1), ctrl.ld_type)
    node load = mux(T_638, T_623, T_637)
    inst csr of CSR
    host := csr.host
    csr.src := Pad(ew_alu,?)
    node T_639 = bits(ew_inst, 31, 20)
    csr.addr := T_639
    csr.cmd := ctrl.csr_cmd
    node T_640 = add-wrap(Pad(ew_pc,?), Pad(UInt<3>(4),?))
    node T_641 = eq(UInt<2>(3), ctrl.wb_sel)
    node T_642 = mux(T_641, Pad(csr.data,?), Pad(ew_alu,?))
    node T_643 = eq(UInt<2>(2), ctrl.wb_sel)
    node T_644 = mux(T_643, Pad(T_640,?), Pad(T_642,?))
    node T_645 = eq(UInt<2>(1), ctrl.wb_sel)
    node regWrite = mux(T_645, Pad(load,?), Pad(T_644,?))
    regFile.wen := ctrl.wb_en
    regFile.waddr := ex_rd_addr
    regFile.wdata := Pad(regWrite,?)
