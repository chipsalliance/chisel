; RUN: firrtl -i %s -o %s.flo -x X -p c | tee %s.out | FileCheck %s
; CHECK: Done!

circuit Datapath : 
  module ALU : 
    input B : UInt<32>
    output out : UInt<32>
    output sum : UInt<32>
    input A : UInt<32>
    input alu_op : UInt<4>
    
    node T_416 = bits(B, 4, 0)
    wire shamt : UInt
    shamt := T_416
    node T_417 = add-wrap(Pad(A,?), Pad(B,?))
    node T_418 = sub-wrap(Pad(A,?), Pad(B,?))
    wire T_419 : SInt
    T_419 := A
    node T_420 = dshr(T_419, shamt)
    wire T_421 : UInt
    T_421 := T_420
    node T_422 = dshr(A, shamt)
    node T_423 = dshl(A, shamt)
    node T_424 = bits(T_423, 31, 0)
    wire T_425 : SInt
    T_425 := A
    wire T_426 : SInt
    T_426 := B
    node T_427 = lt(Pad(T_425,?), Pad(T_426,?))
    wire T_428 : UInt
    T_428 := T_427
    node T_429 = lt(Pad(A,?), Pad(B,?))
    node T_430 = bit-and(Pad(A,?), Pad(B,?))
    node T_431 = bit-or(Pad(A,?), Pad(B,?))
    node T_432 = bit-xor(Pad(A,?), Pad(B,?))
    node T_433 = eq(Pad(UInt<4>(10),?), Pad(alu_op,?))
    node T_434 = mux(Pad(T_433,?), Pad(A,?), Pad(B,?))
    node T_435 = eq(Pad(UInt<4>(4),?), Pad(alu_op,?))
    node T_436 = mux(Pad(T_435,?), Pad(T_432,?), Pad(T_434,?))
    node T_437 = eq(Pad(UInt<4>(3),?), Pad(alu_op,?))
    node T_438 = mux(Pad(T_437,?), Pad(T_431,?), Pad(T_436,?))
    node T_439 = eq(Pad(UInt<4>(2),?), Pad(alu_op,?))
    node T_440 = mux(Pad(T_439,?), Pad(T_430,?), Pad(T_438,?))
    node T_441 = eq(Pad(UInt<4>(7),?), Pad(alu_op,?))
    node T_442 = mux(Pad(T_441,?), Pad(T_429,?), Pad(T_440,?))
    node T_443 = eq(Pad(UInt<4>(5),?), Pad(alu_op,?))
    node T_444 = mux(Pad(T_443,?), Pad(T_428,?), Pad(T_442,?))
    node T_445 = eq(Pad(UInt<4>(6),?), Pad(alu_op,?))
    node T_446 = mux(Pad(T_445,?), Pad(T_424,?), Pad(T_444,?))
    node T_447 = eq(Pad(UInt<4>(8),?), Pad(alu_op,?))
    node T_448 = mux(Pad(T_447,?), Pad(T_422,?), Pad(T_446,?))
    node T_449 = eq(Pad(UInt<4>(9),?), Pad(alu_op,?))
    node T_450 = mux(Pad(T_449,?), Pad(T_421,?), Pad(T_448,?))
    node T_451 = eq(Pad(UInt<4>(1),?), Pad(alu_op,?))
    node T_452 = mux(Pad(T_451,?), Pad(T_418,?), Pad(T_450,?))
    node T_453 = eq(Pad(UInt<4>(0),?), Pad(alu_op,?))
    node T_454 = mux(Pad(T_453,?), Pad(T_417,?), Pad(T_452,?))
    out := Pad(T_454,?)
    node T_455 = bit(alu_op, 0)
    node T_456 = sub-wrap(Pad(UInt<1>(0),?), Pad(B,?))
    node T_457 = mux(Pad(T_455,?), Pad(T_456,?), Pad(B,?))
    node T_458 = add-wrap(Pad(A,?), Pad(T_457,?))
    sum := Pad(T_458,?)
  module BrCond : 
    input br_type : UInt<3>
    input rs2 : UInt<32>
    input rs1 : UInt<32>
    output taken : UInt<1>
    
    node eq = eq(Pad(rs1,?), Pad(rs2,?))
    node neq = bit-not(eq)
    wire T_459 : SInt
    T_459 := rs1
    wire T_460 : SInt
    T_460 := rs2
    node lt = lt(Pad(T_459,?), Pad(T_460,?))
    node ge = bit-not(lt)
    node ltu = lt(Pad(rs1,?), Pad(rs2,?))
    node geu = bit-not(ltu)
    node T_461 = UInt<1>(1)
    node T_462 = UInt<1>(0)
    node T_463 = eq(Pad(br_type,?), Pad(UInt<3>(2),?))
    node T_464 = bit-and(T_463, eq)
    node T_465 = eq(Pad(br_type,?), Pad(UInt<3>(6),?))
    node T_466 = bit-and(T_465, neq)
    node T_467 = bit-or(T_464, T_466)
    node T_468 = eq(Pad(br_type,?), Pad(UInt<3>(1),?))
    node T_469 = bit-and(T_468, lt)
    node T_470 = bit-or(T_467, T_469)
    node T_471 = eq(Pad(br_type,?), Pad(UInt<3>(5),?))
    node T_472 = bit-and(T_471, ge)
    node T_473 = bit-or(T_470, T_472)
    node T_474 = eq(Pad(br_type,?), Pad(UInt<3>(0),?))
    node T_475 = bit-and(T_474, ltu)
    node T_476 = bit-or(T_473, T_475)
    node T_477 = eq(Pad(br_type,?), Pad(UInt<3>(4),?))
    node T_478 = bit-and(T_477, geu)
    node T_479 = bit-or(T_476, T_478)
    taken := Pad(T_479,?)
  module RegFile : 
    input raddr1 : UInt<5>
    input raddr2 : UInt<5>
    output rdata1 : UInt<32>
    output rdata2 : UInt<32>
    input wen : UInt<1>
    input waddr : UInt<5>
    input wdata : UInt<32>
    
    mem regs : UInt<32>[32]
    node T_480 = bit-or-reduce(raddr1)
    accessor T_481 = regs[raddr1]
    node T_482 = mux(Pad(T_480,?), Pad(T_481,?), Pad(UInt<1>(0),?))
    rdata1 := Pad(T_482,?)
    node T_483 = bit-or-reduce(raddr2)
    accessor T_484 = regs[raddr2]
    node T_485 = mux(Pad(T_483,?), Pad(T_484,?), Pad(UInt<1>(0),?))
    rdata2 := Pad(T_485,?)
    node T_486 = bit-or-reduce(waddr)
    node T_487 = bit-and(wen, T_486)
    when T_487 : 
      accessor T_488 = regs[waddr]
      T_488 := Pad(wdata,?)
  module ImmGenWire : 
    output out : UInt<32>
    input sel : UInt<3>
    input inst : UInt<32>
    
    node T_489 = bits(inst, 31, 20)
    wire Iimm : SInt
    Iimm := T_489
    node T_490 = bits(inst, 31, 25)
    node T_491 = bits(inst, 11, 7)
    node T_492 = cat(T_490, T_491)
    wire Simm : SInt
    Simm := T_492
    node T_493 = bit(inst, 31)
    node T_494 = bit(inst, 7)
    node T_495 = bits(inst, 30, 25)
    node T_496 = bits(inst, 11, 8)
    node T_497 = cat(T_493, T_494)
    node T_498 = cat(T_496, UInt<1>(0))
    node T_499 = cat(T_495, T_498)
    node T_500 = cat(T_497, T_499)
    wire Bimm : SInt
    Bimm := T_500
    node T_501 = bits(inst, 31, 12)
    node T_502 = cat(T_501, UInt<12>(0))
    wire Uimm : SInt
    Uimm := T_502
    node T_503 = bit(inst, 31)
    node T_504 = bits(inst, 19, 12)
    node T_505 = bit(inst, 20)
    node T_506 = bits(inst, 30, 25)
    node T_507 = bits(inst, 24, 21)
    node T_508 = cat(T_504, T_505)
    node T_509 = cat(T_503, T_508)
    node T_510 = cat(T_507, UInt<1>(0))
    node T_511 = cat(T_506, T_510)
    node T_512 = cat(T_509, T_511)
    wire Jimm : SInt
    Jimm := T_512
    node T_513 = bits(inst, 19, 15)
    wire Zimm : UInt
    Zimm := T_513
    node T_514 = eq(Pad(UInt<3>(3),?), Pad(sel,?))
    node T_515 = mux(Pad(T_514,?), Pad(Jimm,?), Pad(Zimm,?))
    node T_516 = eq(Pad(UInt<3>(2),?), Pad(sel,?))
    node T_517 = mux(Pad(T_516,?), Pad(Uimm,?), Pad(T_515,?))
    node T_518 = eq(Pad(UInt<3>(4),?), Pad(sel,?))
    node T_519 = mux(Pad(T_518,?), Pad(Bimm,?), Pad(T_517,?))
    node T_520 = eq(Pad(UInt<3>(1),?), Pad(sel,?))
    node T_521 = mux(Pad(T_520,?), Pad(Simm,?), Pad(T_519,?))
    node T_522 = eq(Pad(UInt<3>(0),?), Pad(sel,?))
    node T_523 = mux(Pad(T_522,?), Pad(Iimm,?), Pad(T_521,?))
    out := Pad(T_523,?)
  module CSR : 
    output host : {status : UInt<32>, tohost : UInt<32>, flip hid : UInt<1>}
    input src : UInt<32>
    input cmd : UInt<2>
    output data : UInt<32>
    input addr : UInt<12>
    
    reg reg_tohost : UInt<32>
    on-reset reg_tohost := Pad(UInt<32>(0),?)
    reg reg_status : UInt<32>
    on-reset reg_status := Pad(UInt<32>(0),?)
    host.tohost := Pad(reg_tohost,?)
    host.status := Pad(reg_status,?)
    node T_524 = eq(Pad(UInt<12>(1291),?), Pad(addr,?))
    node T_525 = mux(Pad(T_524,?), Pad(host.hid,?), Pad(UInt<1>(0),?))
    node T_526 = eq(Pad(UInt<12>(1290),?), Pad(addr,?))
    node T_527 = mux(Pad(T_526,?), Pad(reg_status,?), Pad(T_525,?))
    node T_528 = eq(Pad(UInt<12>(1310),?), Pad(addr,?))
    node T_529 = mux(Pad(T_528,?), Pad(reg_tohost,?), Pad(T_527,?))
    wire T_530 : UInt
    T_530 := T_529
    data := Pad(T_530,?)
    node T_531 = eq(Pad(cmd,?), Pad(UInt<2>(1),?))
    when T_531 : 
      node T_532 = eq(Pad(addr,?), Pad(UInt<12>(1310),?))
      when T_532 : reg_tohost := Pad(src,?)
      node T_533 = eq(Pad(addr,?), Pad(UInt<12>(1290),?))
      when T_533 : reg_status := Pad(src,?)
    node T_534 = eq(Pad(cmd,?), Pad(UInt<2>(2),?))
    node T_535 = neq(Pad(src,?), Pad(UInt<1>(0),?))
    node T_536 = bit-and(T_534, T_535)
    when T_536 : 
      node T_537 = eq(Pad(addr,?), Pad(UInt<12>(1310),?))
      when T_537 : 
        node T_538 = dshl(UInt<1>(1), src)
        node T_539 = bit-or(Pad(data,?), Pad(T_538,?))
        reg_tohost := Pad(T_539,?)
      node T_540 = eq(Pad(addr,?), Pad(UInt<12>(1290),?))
      when T_540 : 
        node T_541 = dshl(UInt<1>(1), src)
        node T_542 = bit-or(Pad(data,?), Pad(T_541,?))
        reg_status := Pad(T_542,?)
    node T_543 = eq(Pad(cmd,?), Pad(UInt<2>(3),?))
    node T_544 = neq(Pad(src,?), Pad(UInt<1>(0),?))
    node T_545 = bit-and(T_543, T_544)
    when T_545 : 
      node T_546 = eq(Pad(addr,?), Pad(UInt<12>(1310),?))
      when T_546 : 
        node T_547 = dshl(UInt<1>(0), src)
        node T_548 = bit-and(Pad(data,?), Pad(T_547,?))
        reg_tohost := Pad(T_548,?)
      node T_549 = eq(Pad(addr,?), Pad(UInt<12>(1290),?))
      when T_549 : 
        node T_550 = dshl(UInt<1>(0), src)
        node T_551 = bit-and(Pad(data,?), Pad(T_550,?))
        reg_status := Pad(T_551,?)
  module Datapath : 
    output host : {status : UInt<32>, tohost : UInt<32>, flip hid : UInt<1>}
    output dcache : {re : UInt<1>, we : UInt<4>, addr : UInt<32>, flip dout : UInt<32>, din : UInt<32>}
    input stall : UInt<1>
    input ctrl : {flip inst : UInt<32>, flip stall : UInt<1>, imm_sel : UInt<3>, wb_en : UInt<1>, wb_sel : UInt<2>, A_sel : UInt<1>, B_sel : UInt<1>, alu_op : UInt<4>, br_type : UInt<3>, data_re : UInt<1>, st_type : UInt<2>, ld_type : UInt<3>, csr_cmd : UInt<2>, pc_sel : UInt<1>, inst_type : UInt<1>, inst_re : UInt<1>}
    output icache : {re : UInt<1>, we : UInt<4>, addr : UInt<32>, flip dout : UInt<32>, din : UInt<32>}
    
    inst alu of ALU
    inst brCond of BrCond
    inst regFile of RegFile
    inst immGen of ImmGenWire
    reg fe_inst : UInt<32>
    on-reset fe_inst := Pad(UInt<32>(0),?)
    reg fe_pc : UInt
    reg ew_inst : UInt<32>
    on-reset ew_inst := Pad(UInt<32>(0),?)
    reg ew_pc : UInt
    reg ew_alu : UInt
    node T_552 = sub-wrap(Pad(UInt<14>(8192),?), Pad(UInt<3>(4),?))
    reg pc : UInt
    on-reset pc := Pad(T_552,?)
    node T_553 = eq(Pad(ctrl.pc_sel,?), Pad(UInt<1>(1),?))
    node T_554 = bit-or(T_553, brCond.taken)
    node T_555 = add-wrap(Pad(pc,?), Pad(UInt<3>(4),?))
    node iaddr = mux(Pad(T_554,?), Pad(alu.sum,?), Pad(T_555,?))
    node T_556 = eq(Pad(ctrl.inst_type,?), Pad(UInt<1>(1),?))
    node T_557 = bit-or(T_556, brCond.taken)
    node inst = mux(Pad(T_557,?), Pad(UInt<5>(19),?), Pad(icache.dout,?))
    icache.addr := Pad(iaddr,?)
    icache.re := Pad(ctrl.inst_re,?)
    node T_558 = bit-or-reduce(dcache.we)
    node T_559 = bit-not(T_558)
    node T_560 = bit-and(icache.re, T_559)
    node T_561 = mux(Pad(T_560,?), Pad(iaddr,?), Pad(pc,?))
    pc := Pad(T_561,?)
    node T_562 = bit-not(stall)
    when T_562 : 
      fe_pc := Pad(pc,?)
      fe_inst := Pad(inst,?)
    ctrl.inst := Pad(fe_inst,?)
    ctrl.stall := Pad(stall,?)
    node rd_addr = bits(fe_inst, 11, 7)
    node rs1_addr = bits(fe_inst, 19, 15)
    node rs2_addr = bits(fe_inst, 24, 20)
    regFile.raddr1 := Pad(rs1_addr,?)
    regFile.raddr2 := Pad(rs2_addr,?)
    immGen.inst := Pad(fe_inst,?)
    immGen.sel := Pad(ctrl.imm_sel,?)
    node rs1NotZero = bit-or-reduce(rs1_addr)
    node rs2NotZero = bit-or-reduce(rs2_addr)
    node T_563 = eq(Pad(ctrl.wb_sel,?), Pad(UInt<2>(0),?))
    node alutype = bit-and(ctrl.wb_en, T_563)
    node ex_rd_addr = bits(ew_inst, 11, 7)
    node T_564 = bit-and(alutype, rs1NotZero)
    node T_565 = eq(Pad(rs1_addr,?), Pad(ex_rd_addr,?))
    node T_566 = bit-and(T_564, T_565)
    node rs1 = mux(Pad(T_566,?), Pad(ew_alu,?), Pad(regFile.rdata1,?))
    node T_567 = bit-and(alutype, rs2NotZero)
    node T_568 = eq(Pad(rs2_addr,?), Pad(ex_rd_addr,?))
    node T_569 = bit-and(T_567, T_568)
    node rs2 = mux(Pad(T_569,?), Pad(ew_alu,?), Pad(regFile.rdata2,?))
    node T_570 = eq(Pad(ctrl.A_sel,?), Pad(UInt<1>(0),?))
    node T_571 = mux(Pad(T_570,?), Pad(rs1,?), Pad(fe_pc,?))
    alu.A := Pad(T_571,?)
    node T_572 = eq(Pad(ctrl.B_sel,?), Pad(UInt<1>(0),?))
    node T_573 = mux(Pad(T_572,?), Pad(rs2,?), Pad(immGen.out,?))
    alu.B := Pad(T_573,?)
    alu.alu_op := Pad(ctrl.alu_op,?)
    brCond.rs1 := Pad(rs1,?)
    brCond.rs2 := Pad(rs2,?)
    brCond.br_type := Pad(ctrl.br_type,?)
    node T_574 = bit(alu.sum, 1)
    node T_575 = dshl(T_574, UInt<3>(4))
    node T_576 = bit(alu.sum, 0)
    node T_577 = dshl(T_576, UInt<2>(3))
    node woffset = bit-or(Pad(T_575,?), Pad(T_577,?))
    dcache.re := Pad(ctrl.data_re,?)
    node T_578 = mux(Pad(stall,?), Pad(ew_alu,?), Pad(alu.sum,?))
    dcache.addr := Pad(T_578,?)
    node T_579 = bits(alu.sum, 1, 0)
    node T_580 = dshl(UInt<2>(3), T_579)
    node T_581 = bits(alu.sum, 1, 0)
    node T_582 = dshl(UInt<1>(1), T_581)
    node T_583 = eq(Pad(UInt<2>(2),?), Pad(ctrl.st_type,?))
    node T_584 = mux(Pad(T_583,?), Pad(T_582,?), Pad(UInt<1>(0),?))
    node T_585 = eq(Pad(UInt<2>(1),?), Pad(ctrl.st_type,?))
    node T_586 = mux(Pad(T_585,?), Pad(T_580,?), Pad(T_584,?))
    node T_587 = eq(Pad(UInt<2>(0),?), Pad(ctrl.st_type,?))
    node T_588 = mux(Pad(T_587,?), Pad(UInt<4>(15),?), Pad(T_586,?))
    node T_589 = mux(Pad(stall,?), Pad(UInt<1>(0),?), Pad(T_588,?))
    dcache.we := Pad(T_589,?)
    node T_590 = dshl(rs2, woffset)
    dcache.din := Pad(T_590,?)
    node T_591 = bit-not(stall)
    when T_591 : 
      ew_pc := Pad(fe_pc,?)
      ew_inst := Pad(fe_inst,?)
      ew_alu := Pad(alu.out,?)
    node T_592 = bit(ew_alu, 1)
    node T_593 = dshl(T_592, UInt<3>(4))
    node T_594 = bit(ew_alu, 0)
    node T_595 = dshl(T_594, UInt<2>(3))
    node loffset = bit-or(Pad(T_593,?), Pad(T_595,?))
    node lshift = dshr(dcache.dout, loffset)
    node T_596 = bits(lshift, 15, 0)
    wire T_597 : SInt
    T_597 := T_596
    node T_598 = Pad(T_597, 32)
    wire T_599 : UInt
    T_599 := T_598
    node T_600 = bits(lshift, 7, 0)
    wire T_601 : SInt
    T_601 := T_600
    node T_602 = Pad(T_601, 32)
    wire T_603 : UInt
    T_603 := T_602
    node T_604 = bits(lshift, 15, 0)
    wire T_605 : UInt
    T_605 := T_604
    node T_606 = bits(lshift, 7, 0)
    wire T_607 : UInt
    T_607 := T_606
    node T_608 = eq(Pad(UInt<3>(4),?), Pad(ctrl.ld_type,?))
    node T_609 = mux(Pad(T_608,?), Pad(T_607,?), Pad(dcache.dout,?))
    node T_610 = eq(Pad(UInt<3>(3),?), Pad(ctrl.ld_type,?))
    node T_611 = mux(Pad(T_610,?), Pad(T_605,?), Pad(T_609,?))
    node T_612 = eq(Pad(UInt<3>(2),?), Pad(ctrl.ld_type,?))
    node T_613 = mux(Pad(T_612,?), Pad(T_603,?), Pad(T_611,?))
    node T_614 = eq(Pad(UInt<3>(1),?), Pad(ctrl.ld_type,?))
    node load = mux(Pad(T_614,?), Pad(T_599,?), Pad(T_613,?))
    inst csr of CSR
    csr.host := host
    csr.src := Pad(ew_alu,?)
    node T_615 = bits(ew_inst, 31, 20)
    csr.addr := Pad(T_615,?)
    csr.cmd := Pad(ctrl.csr_cmd,?)
    node T_616 = add-wrap(Pad(ew_pc,?), Pad(UInt<3>(4),?))
    node T_617 = eq(Pad(UInt<2>(3),?), Pad(ctrl.wb_sel,?))
    node T_618 = mux(Pad(T_617,?), Pad(csr.data,?), Pad(ew_alu,?))
    node T_619 = eq(Pad(UInt<2>(2),?), Pad(ctrl.wb_sel,?))
    node T_620 = mux(Pad(T_619,?), Pad(T_616,?), Pad(T_618,?))
    node T_621 = eq(Pad(UInt<2>(1),?), Pad(ctrl.wb_sel,?))
    node regWrite = mux(Pad(T_621,?), Pad(load,?), Pad(T_620,?))
    regFile.wen := Pad(ctrl.wb_en,?)
    regFile.waddr := Pad(ex_rd_addr,?)
    regFile.wdata := Pad(regWrite,?)
