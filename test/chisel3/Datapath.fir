; RUN: firrtl -i %s -o %s.flo -x X -p cdt | tee %s.out | FileCheck %s
; CHECK: Done!

circuit Datapath : 
  module ALU : 
    input B : UInt<32>
    output out : UInt<32>
    output sum : UInt<32>
    input A : UInt<32>
    input alu_op : UInt<4>
    
    node T_418 = bits(B, 4, 0)
    wire shamt : UInt<5>
    shamt := T_418
    node T_419 = add-wrap(A, B)
    node T_420 = sub-wrap(A, B)
    wire T_421 : SInt<32>
    T_421 := A
    node T_422 = dshr(T_421, shamt)
    wire T_423 : UInt
    T_423 := Pad(T_422,?)
    node T_424 = dshr(A, shamt)
    node T_425 = dshl(A, shamt)
    node T_426 = bits(T_425, 31, 0)
    wire T_427 : SInt<32>
    T_427 := A
    wire T_428 : SInt<32>
    T_428 := B
    node T_429 = lt(T_427, T_428)
    wire T_430 : UInt<1>
    T_430 := T_429
    node T_431 = lt(A, B)
    node T_432 = bit-and(A, B)
    node T_433 = bit-or(A, B)
    node T_434 = bit-xor(A, B)
    node T_435 = eq(UInt<4>(10), alu_op)
    node T_436 = mux(T_435, A, B)
    node T_437 = eq(UInt<4>(4), alu_op)
    node T_438 = mux(T_437, T_434, T_436)
    node T_439 = eq(UInt<4>(3), alu_op)
    node T_440 = mux(T_439, T_433, T_438)
    node T_441 = eq(UInt<4>(2), alu_op)
    node T_442 = mux(T_441, T_432, T_440)
    node T_443 = eq(UInt<4>(7), alu_op)
    node T_444 = mux(T_443, Pad(T_431,32), T_442)
    node T_445 = eq(UInt<4>(5), alu_op)
    node T_446 = mux(T_445, T_430, T_444)
    node T_447 = eq(UInt<4>(6), alu_op)
    node T_448 = mux(T_447, T_426, Pad(T_446,32))
    node T_449 = eq(UInt<4>(8), alu_op)
    node T_450 = mux(T_449, Pad(T_424,?), Pad(T_448,?))
    node T_451 = eq(UInt<4>(9), alu_op)
    node T_452 = mux(T_451, Pad(T_423,?), Pad(T_450,?))
    node T_453 = eq(UInt<4>(1), alu_op)
    node T_454 = mux(T_453, Pad(T_420,?), Pad(T_452,?))
    node T_455 = eq(UInt<4>(0), alu_op)
    node T_456 = mux(T_455, Pad(T_419,?), Pad(T_454,?))
    out := Pad(T_456,?)
    node T_457 = bit(alu_op, 0)
    node T_458 = sub-wrap(UInt<32>(0), B)
    node T_459 = mux(T_457, T_458, B)
    node T_460 = add-wrap(A, T_459)
    sum := T_460
  module BrCond : 
    input br_type : UInt<3>
    input rs1 : UInt<32>
    input rs2 : UInt<32>
    output taken : UInt<1>
    
    node eq = eq(rs1, rs2)
    node neq = bit-not(eq)
    wire T_461 : SInt<32>
    T_461 := rs1
    wire T_462 : SInt<32>
    T_462 := rs2
    node lt = lt(T_461, T_462)
    node ge = bit-not(lt)
    node ltu = lt(rs1, rs2)
    node geu = bit-not(ltu)
    node T_463 = UInt<1>(1)
    node T_464 = UInt<1>(0)
    node T_465 = eq(br_type, UInt<3>(2))
    node T_466 = bit-and(T_465, eq)
    node T_467 = eq(br_type, UInt<3>(6))
    node T_468 = bit-and(T_467, neq)
    node T_469 = bit-or(T_466, T_468)
    node T_470 = eq(br_type, UInt<3>(1))
    node T_471 = bit-and(T_470, lt)
    node T_472 = bit-or(T_469, T_471)
    node T_473 = eq(br_type, UInt<3>(5))
    node T_474 = bit-and(T_473, ge)
    node T_475 = bit-or(T_472, T_474)
    node T_476 = eq(br_type, UInt<3>(0))
    node T_477 = bit-and(T_476, ltu)
    node T_478 = bit-or(T_475, T_477)
    node T_479 = eq(br_type, UInt<3>(4))
    node T_480 = bit-and(T_479, geu)
    node T_481 = bit-or(T_478, T_480)
    taken := T_481
  module RegFile : 
    output rdata1 : UInt<32>
    input raddr2 : UInt<5>
    input raddr1 : UInt<5>
    input wdata : UInt<32>
    output rdata2 : UInt<32>
    input wen : UInt<1>
    input waddr : UInt<5>
    
    mem regs : UInt<32>[32]
    node T_482 = bit-or-reduce(raddr1)
    accessor T_483 = regs[raddr1]
    node T_484 = mux(T_482, T_483, UInt<32>(0))
    rdata1 := T_484
    node T_485 = bit-or-reduce(raddr2)
    accessor T_486 = regs[raddr2]
    node T_487 = mux(T_485, T_486, UInt<32>(0))
    rdata2 := T_487
    node T_488 = bit-or-reduce(waddr)
    node T_489 = bit-and(wen, T_488)
    when T_489 : 
      accessor T_490 = regs[waddr]
      T_490 := wdata
  module ImmGenWire : 
    output out : UInt<32>
    input inst : UInt<32>
    input sel : UInt<3>
    
    node T_491 = bits(inst, 31, 20)
    wire Iimm : SInt<12>
    Iimm := T_491
    node T_492 = bits(inst, 31, 25)
    node T_493 = bits(inst, 11, 7)
    node T_494 = cat(T_492, T_493)
    wire Simm : SInt<12>
    Simm := T_494
    node T_495 = bit(inst, 31)
    node T_496 = bit(inst, 7)
    node T_497 = bits(inst, 30, 25)
    node T_498 = bits(inst, 11, 8)
    node T_499 = cat(T_495, T_496)
    node T_500 = cat(T_498, UInt<1>(0))
    node T_501 = cat(T_497, T_500)
    node T_502 = cat(T_499, T_501)
    wire Bimm : SInt<1>
    Bimm := T_502
    node T_503 = bits(inst, 31, 12)
    node T_504 = cat(T_503, UInt<12>(0))
    wire Uimm : SInt<32>
    Uimm := T_504
    node T_505 = bit(inst, 31)
    node T_506 = bits(inst, 19, 12)
    node T_507 = bit(inst, 20)
    node T_508 = bits(inst, 30, 25)
    node T_509 = bits(inst, 24, 21)
    node T_510 = cat(T_506, T_507)
    node T_511 = cat(T_505, T_510)
    node T_512 = cat(T_509, UInt<1>(0))
    node T_513 = cat(T_508, T_512)
    node T_514 = cat(T_511, T_513)
    wire Jimm : SInt<1>
    Jimm := T_514
    node T_515 = bits(inst, 19, 15)
    wire T_516 : UInt<5>
    T_516 := T_515
    node T_517 = Pad(T_516, 32)
    wire Zimm : SInt<32>
    Zimm := T_517
    node T_518 = eq(UInt<3>(3), sel)
    node T_519 = mux(T_518, Pad(Jimm,32), Zimm)
    node T_520 = eq(UInt<3>(2), sel)
    node T_521 = mux(T_520, Uimm, T_519)
    node T_522 = eq(UInt<3>(4), sel)
    node T_523 = mux(T_522, Pad(Bimm,32), T_521)
    node T_524 = eq(UInt<3>(1), sel)
    node T_525 = mux(T_524, Pad(Simm,32), T_523)
    node T_526 = eq(UInt<3>(0), sel)
    node T_527 = mux(T_526, Pad(Iimm,32), T_525)
    out := T_527
  module CSR : 
    output host : {status : UInt<32>, tohost : UInt<32>, flip hid : UInt<1>}
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
    node T_528 = eq(UInt<12>(1291), addr)
    node T_529 = mux(T_528, host.hid, UInt<1>(0))
    node T_530 = eq(UInt<12>(1290), addr)
    node T_531 = mux(T_530, reg_status, Pad(T_529,32))
    node T_532 = eq(UInt<12>(1310), addr)
    node T_533 = mux(T_532, reg_tohost, T_531)
    wire T_534 : UInt<32>
    T_534 := T_533
    data := T_534
    node T_535 = eq(cmd, UInt<2>(1))
    when T_535 : 
      node T_536 = eq(addr, UInt<12>(1310))
      when T_536 : reg_tohost := src
      node T_537 = eq(addr, UInt<12>(1290))
      when T_537 : reg_status := src
    node T_538 = eq(cmd, UInt<2>(2))
    node T_539 = neq(src, UInt<32>(0))
    node T_540 = bit-and(T_538, T_539)
    when T_540 : 
      node T_541 = eq(addr, UInt<12>(1310))
      when T_541 : 
        node T_542 = dshl(UInt<1>(1), src)
        node T_543 = bit-or(Pad(data,?), Pad(T_542,?))
        reg_tohost := Pad(T_543,?)
      node T_544 = eq(addr, UInt<12>(1290))
      when T_544 : 
        node T_545 = dshl(UInt<1>(1), src)
        node T_546 = bit-or(Pad(data,?), Pad(T_545,?))
        reg_status := Pad(T_546,?)
    node T_547 = eq(cmd, UInt<2>(3))
    node T_548 = neq(src, UInt<32>(0))
    node T_549 = bit-and(T_547, T_548)
    when T_549 : 
      node T_550 = eq(addr, UInt<12>(1310))
      when T_550 : 
        node T_551 = dshl(UInt<1>(0), src)
        node T_552 = bit-and(Pad(data,?), Pad(T_551,?))
        reg_tohost := Pad(T_552,?)
      node T_553 = eq(addr, UInt<12>(1290))
      when T_553 : 
        node T_554 = dshl(UInt<1>(0), src)
        node T_555 = bit-and(Pad(data,?), Pad(T_554,?))
        reg_status := Pad(T_555,?)
  module Datapath : 
    output host : {status : UInt<32>, tohost : UInt<32>, flip hid : UInt<1>}
    output dcache : {din : UInt<32>, flip dout : UInt<32>, re : UInt<1>, addr : UInt<32>, we : UInt<4>}
    input stall : UInt<1>
    output icache : {din : UInt<32>, flip dout : UInt<32>, re : UInt<1>, addr : UInt<32>, we : UInt<4>}
    input ctrl : {inst_re : UInt<1>, data_re : UInt<1>, flip inst : UInt<32>, pc_sel : UInt<1>, br_type : UInt<3>, alu_op : UInt<4>, B_sel : UInt<1>, wb_sel : UInt<2>, st_type : UInt<2>, imm_sel : UInt<3>, wb_en : UInt<1>, ld_type : UInt<3>, A_sel : UInt<1>, csr_cmd : UInt<2>, flip stall : UInt<1>, inst_type : UInt<1>}
    
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
    node T_556 = sub-wrap(UInt<14>(8192), UInt<14>(4))
    reg pc : UInt<14>
    on-reset pc := T_556
    node T_557 = eq(ctrl.pc_sel, UInt<1>(1))
    node T_558 = bit-or(T_557, brCond.taken)
    node T_559 = add-wrap(pc, UInt<14>(4))
    node iaddr = mux(T_558, alu.sum, Pad(T_559,32))
    node T_560 = eq(ctrl.inst_type, UInt<1>(1))
    node T_561 = bit-or(T_560, brCond.taken)
    node inst = mux(T_561, UInt<32>(19), icache.dout)
    icache.addr := iaddr
    icache.re := ctrl.inst_re
    node T_562 = bit-or-reduce(dcache.we)
    node T_563 = bit-not(T_562)
    node T_564 = bit-and(icache.re, T_563)
    node T_565 = mux(T_564, iaddr, Pad(pc,32))
    pc := T_565
    node T_566 = bit-not(stall)
    when T_566 : 
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
    node rs1NotZero = bit-or-reduce(rs1_addr)
    node rs2NotZero = bit-or-reduce(rs2_addr)
    node T_567 = eq(ctrl.wb_sel, UInt<2>(0))
    node alutype = bit-and(ctrl.wb_en, T_567)
    node ex_rd_addr = bits(ew_inst, 11, 7)
    node T_568 = bit-and(alutype, rs1NotZero)
    node T_569 = eq(rs1_addr, ex_rd_addr)
    node T_570 = bit-and(T_568, T_569)
    node rs1 = mux(T_570, Pad(ew_alu,?), Pad(regFile.rdata1,?))
    node T_571 = bit-and(alutype, rs2NotZero)
    node T_572 = eq(rs2_addr, ex_rd_addr)
    node T_573 = bit-and(T_571, T_572)
    node rs2 = mux(T_573, Pad(ew_alu,?), Pad(regFile.rdata2,?))
    node T_574 = eq(ctrl.A_sel, UInt<1>(0))
    node T_575 = mux(T_574, Pad(rs1,?), Pad(fe_pc,?))
    alu.A := Pad(T_575,?)
    node T_576 = eq(ctrl.B_sel, UInt<1>(0))
    node T_577 = mux(T_576, Pad(rs2,?), Pad(immGen.out,?))
    alu.B := Pad(T_577,?)
    alu.alu_op := ctrl.alu_op
    brCond.rs1 := Pad(rs1,?)
    brCond.rs2 := Pad(rs2,?)
    brCond.br_type := ctrl.br_type
    node T_578 = bit(alu.sum, 1)
    node T_579 = dshl(T_578, UInt<3>(4))
    node T_580 = bit(alu.sum, 0)
    node T_581 = dshl(T_580, UInt<2>(3))
    node woffset = bit-or(T_579, T_581)
    dcache.re := ctrl.data_re
    node T_582 = mux(stall, Pad(ew_alu,?), Pad(alu.sum,?))
    dcache.addr := Pad(T_582,?)
    node T_583 = bits(alu.sum, 1, 0)
    node T_584 = dshl(UInt<2>(3), T_583)
    node T_585 = bits(alu.sum, 1, 0)
    node T_586 = dshl(UInt<1>(1), T_585)
    node T_587 = eq(UInt<2>(2), ctrl.st_type)
    node T_588 = mux(T_587, Pad(T_586,?), Pad(UInt<1>(0),?))
    node T_589 = eq(UInt<2>(1), ctrl.st_type)
    node T_590 = mux(T_589, Pad(T_584,?), Pad(T_588,?))
    node T_591 = eq(UInt<2>(0), ctrl.st_type)
    node T_592 = mux(T_591, Pad(UInt<4>(15),?), Pad(T_590,?))
    node T_593 = mux(stall, Pad(UInt<1>(0),?), Pad(T_592,?))
    dcache.we := Pad(T_593,?)
    node T_594 = dshl(rs2, woffset)
    dcache.din := Pad(T_594,?)
    node T_595 = bit-not(stall)
    when T_595 : 
      ew_pc := Pad(fe_pc,?)
      ew_inst := fe_inst
      ew_alu := Pad(alu.out,?)
    node T_596 = bit(ew_alu, 1)
    node T_597 = dshl(T_596, UInt<3>(4))
    node T_598 = bit(ew_alu, 0)
    node T_599 = dshl(T_598, UInt<2>(3))
    node loffset = bit-or(T_597, T_599)
    node lshift = dshr(dcache.dout, loffset)
    node T_600 = bits(lshift, 15, 0)
    wire T_601 : SInt<16>
    T_601 := T_600
    node T_602 = Pad(T_601, 32)
    wire T_603 : UInt<32>
    T_603 := T_602
    node T_604 = bits(lshift, 7, 0)
    wire T_605 : SInt<8>
    T_605 := T_604
    node T_606 = Pad(T_605, 32)
    wire T_607 : UInt<32>
    T_607 := T_606
    node T_608 = bits(lshift, 15, 0)
    wire T_609 : UInt<16>
    T_609 := T_608
    node T_610 = bits(lshift, 7, 0)
    wire T_611 : UInt<8>
    T_611 := T_610
    node T_612 = eq(UInt<3>(4), ctrl.ld_type)
    node T_613 = mux(T_612, Pad(T_611,32), dcache.dout)
    node T_614 = eq(UInt<3>(3), ctrl.ld_type)
    node T_615 = mux(T_614, Pad(T_609,32), T_613)
    node T_616 = eq(UInt<3>(2), ctrl.ld_type)
    node T_617 = mux(T_616, T_607, T_615)
    node T_618 = eq(UInt<3>(1), ctrl.ld_type)
    node load = mux(T_618, T_603, T_617)
    inst csr of CSR
    csr.host := host
    csr.src := Pad(ew_alu,?)
    node T_619 = bits(ew_inst, 31, 20)
    csr.addr := T_619
    csr.cmd := ctrl.csr_cmd
    node T_620 = add-wrap(Pad(ew_pc,?), Pad(UInt<3>(4),?))
    node T_621 = eq(UInt<2>(3), ctrl.wb_sel)
    node T_622 = mux(T_621, Pad(csr.data,?), Pad(ew_alu,?))
    node T_623 = eq(UInt<2>(2), ctrl.wb_sel)
    node T_624 = mux(T_623, Pad(T_620,?), Pad(T_622,?))
    node T_625 = eq(UInt<2>(1), ctrl.wb_sel)
    node regWrite = mux(T_625, Pad(load,?), Pad(T_624,?))
    regFile.wen := ctrl.wb_en
    regFile.waddr := ex_rd_addr
    regFile.wdata := Pad(regWrite,?)
