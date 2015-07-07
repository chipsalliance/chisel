; RUN: firrtl -i %s -o %s.v -X verilog -p c | tee %s.out | FileCheck %s
;CHECK: Done!

circuit Datapath : 
  module ALU : 
    input B : UInt<32>
    output out : UInt<32>
    output sum : UInt<32>
    input A : UInt<32>
    input alu_op : UInt<4>
    
    node shamt = bits(B, 4, 0)
    node T_426 = add-wrap(A, B)
    node T_427 = sub-wrap(A, B)
    node T_428 = as-SInt(A)
    node T_429 = dshr(T_428, shamt)
    node T_430 = as-UInt(T_429)
    node T_431 = dshr(A, shamt)
    node T_432 = dshl(A, shamt)
    node T_433 = as-SInt(A)
    node T_434 = as-SInt(B)
    node T_435 = lt(T_433, T_434)
    node T_436 = lt(A, B)
    node T_437 = bit-and(A, B)
    node T_438 = bit-or(A, B)
    node T_439 = bit-xor(A, B)
    node T_440 = eq(UInt<4>(10), alu_op)
    node T_441 = mux(T_440, A, B)
    node T_442 = eq(UInt<4>(4), alu_op)
    node T_443 = mux(T_442, T_439, T_441)
    node T_444 = eq(UInt<4>(3), alu_op)
    node T_445 = mux(T_444, T_438, T_443)
    node T_446 = eq(UInt<4>(2), alu_op)
    node T_447 = mux(T_446, T_437, T_445)
    node T_448 = eq(UInt<4>(7), alu_op)
    node T_449 = mux(T_448, T_436, T_447)
    node T_450 = eq(UInt<4>(5), alu_op)
    node T_451 = mux(T_450, T_435, T_449)
    node T_452 = eq(UInt<4>(6), alu_op)
    node T_453 = mux(T_452, T_432, T_451)
    node T_454 = eq(UInt<4>(8), alu_op)
    node T_455 = mux(T_454, T_431, T_453)
    node T_456 = eq(UInt<4>(9), alu_op)
    node T_457 = mux(T_456, T_430, T_455)
    node T_458 = eq(UInt<4>(1), alu_op)
    node T_459 = mux(T_458, T_427, T_457)
    node T_460 = eq(UInt<4>(0), alu_op)
    node T_461 = mux(T_460, T_426, T_459)
    out := T_461
    node T_462 = bit(alu_op, 0)
    node T_463 = sub-wrap(UInt<1>(0), B)
    node T_464 = mux(T_462, T_463, B)
    node T_465 = add-wrap(A, T_464)
    sum := T_465
  module BrCond : 
    output taken : UInt<1>
    input rs2 : UInt<32>
    input rs1 : UInt<32>
    input br_type : UInt<3>
    
    node eq = eq(rs1, rs2)
    node neq = bit-not(eq)
    node T_466 = as-SInt(rs1)
    node T_467 = as-SInt(rs2)
    node lt = lt(T_466, T_467)
    node ge = bit-not(lt)
    node ltu = lt(rs1, rs2)
    node geu = bit-not(ltu)
    node T_468 = eq(br_type, UInt<3>(2))
    node T_469 = bit-and(T_468, eq)
    node T_470 = eq(br_type, UInt<3>(6))
    node T_471 = bit-and(T_470, neq)
    node T_472 = bit-or(T_469, T_471)
    node T_473 = eq(br_type, UInt<3>(1))
    node T_474 = bit-and(T_473, lt)
    node T_475 = bit-or(T_472, T_474)
    node T_476 = eq(br_type, UInt<3>(5))
    node T_477 = bit-and(T_476, ge)
    node T_478 = bit-or(T_475, T_477)
    node T_479 = eq(br_type, UInt<3>(0))
    node T_480 = bit-and(T_479, ltu)
    node T_481 = bit-or(T_478, T_480)
    node T_482 = eq(br_type, UInt<3>(4))
    node T_483 = bit-and(T_482, geu)
    node T_484 = bit-or(T_481, T_483)
    taken := T_484
  module RegFile : 
    input raddr1 : UInt<5>
    input raddr2 : UInt<5>
    output rdata1 : UInt<32>
    output rdata2 : UInt<32>
    input wen : UInt<1>
    input waddr : UInt<5>
    input wdata : UInt<32>
    
    cmem regs : UInt<32>[32]
    node T_485 = eq(raddr1, UInt<1>(0))
    node T_486 = bit-not(T_485)
    accessor T_487 = regs[raddr1]
    node T_488 = mux(T_486, T_487, UInt<1>(0))
    rdata1 := T_488
    node T_489 = eq(raddr2, UInt<1>(0))
    node T_490 = bit-not(T_489)
    accessor T_491 = regs[raddr2]
    node T_492 = mux(T_490, T_491, UInt<1>(0))
    rdata2 := T_492
    node T_493 = eq(waddr, UInt<1>(0))
    node T_494 = bit-not(T_493)
    node T_495 = bit-and(wen, T_494)
    when T_495 : 
      accessor T_496 = regs[waddr]
      T_496 := wdata
  module ImmGenWire : 
    output out : UInt<32>
    input inst_252 : UInt<32>
    input sel : UInt<3>
    
    node T_497 = bits(inst_252, 31, 20)
    node Iimm = as-SInt(T_497)
    node T_498 = bits(inst_252, 31, 25)
    node T_499 = bits(inst_252, 11, 7)
    node T_500 = cat(T_498, T_499)
    node Simm = as-SInt(T_500)
    node T_501 = bit(inst_252, 31)
    node T_502 = bit(inst_252, 7)
    node T_503 = bits(inst_252, 30, 25)
    node T_504 = bits(inst_252, 11, 8)
    node T_505 = cat(T_501, T_502)
    node T_506 = cat(T_504, UInt<1>(0))
    node T_507 = cat(T_503, T_506)
    node T_508 = cat(T_505, T_507)
    node Bimm = as-SInt(T_508)
    node T_509 = bits(inst_252, 31, 12)
    node T_510 = cat(T_509, UInt<12>(0))
    node Uimm = as-SInt(T_510)
    node T_511 = bit(inst_252, 31)
    node T_512 = bits(inst_252, 19, 12)
    node T_513 = bit(inst_252, 20)
    node T_514 = bits(inst_252, 30, 25)
    node T_515 = bits(inst_252, 24, 21)
    node T_516 = cat(T_512, T_513)
    node T_517 = cat(T_511, T_516)
    node T_518 = cat(T_515, UInt<1>(0))
    node T_519 = cat(T_514, T_518)
    node T_520 = cat(T_517, T_519)
    node Jimm = as-SInt(T_520)
    node T_521 = bits(inst_252, 19, 15)
    node Zimm = convert(T_521)
    node T_522 = eq(UInt<3>(3), sel)
    node T_523 = mux(T_522, Jimm, Zimm)
    node T_524 = eq(UInt<3>(2), sel)
    node T_525 = mux(T_524, Uimm, T_523)
    node T_526 = eq(UInt<3>(4), sel)
    node T_527 = mux(T_526, Bimm, T_525)
    node T_528 = eq(UInt<3>(1), sel)
    node T_529 = mux(T_528, Simm, T_527)
    node T_530 = eq(UInt<3>(0), sel)
    node T_531 = mux(T_530, Iimm, T_529)
    node T_532 = as-UInt(T_531)
    out := T_532
  module CSR : 
    output host : {status : UInt<32>, flip id : UInt<1>, tohost : UInt<32>}
    input cmd : UInt<2>
    input src : UInt<32>
    output data : UInt<32>
    input addr : UInt<12>
    
    reg reg_tohost : UInt<32>
    on-reset reg_tohost := UInt<32>(0)
    reg reg_status : UInt<32>
    on-reset reg_status := UInt<32>(0)
    host.tohost := reg_tohost
    host.status := reg_status
    node T_533 = eq(UInt<12>(1291), addr)
    node T_534 = mux(T_533, host.id, UInt<1>(0))
    node T_535 = eq(UInt<12>(1290), addr)
    node T_536 = mux(T_535, reg_status, T_534)
    node T_537 = eq(UInt<12>(1310), addr)
    node T_538 = mux(T_537, reg_tohost, T_536)
    data := T_538
    node T_539 = eq(cmd, UInt<2>(1))
    when T_539 : 
      node T_540 = eq(addr, UInt<12>(1310))
      when T_540 : reg_tohost := src
      node T_541 = eq(addr, UInt<12>(1290))
      when T_541 : reg_status := src
    node T_542 = eq(cmd, UInt<2>(2))
    node T_543 = neq(src, UInt<1>(0))
    node T_544 = bit-and(T_542, T_543)
    when T_544 : 
      node T_545 = eq(addr, UInt<12>(1310))
      when T_545 : 
        node T_546 = dshl(UInt<1>(1), src)
        node T_547 = bit-or(data, T_546)
        reg_tohost := T_547
      node T_548 = eq(addr, UInt<12>(1290))
      when T_548 : 
        node T_549 = dshl(UInt<1>(1), src)
        node T_550 = bit-or(data, T_549)
        reg_status := T_550
    node T_551 = eq(cmd, UInt<2>(3))
    node T_552 = neq(src, UInt<1>(0))
    node T_553 = bit-and(T_551, T_552)
    when T_553 : 
      node T_554 = eq(addr, UInt<12>(1310))
      when T_554 : 
        node T_555 = dshl(UInt<1>(0), src)
        node T_556 = bit-and(data, T_555)
        reg_tohost := T_556
      node T_557 = eq(addr, UInt<12>(1290))
      when T_557 : 
        node T_558 = dshl(UInt<1>(0), src)
        node T_559 = bit-and(data, T_558)
        reg_status := T_559
  module Datapath : 
    output host : {status : UInt<32>, flip id : UInt<1>, tohost : UInt<32>}
    input ctrl : {flip inst_424 : UInt<32>, pc_sel : UInt<1>, inst_type : UInt<1>, inst_re : UInt<1>, flip stall : UInt<1>, imm_sel : UInt<3>, wb_en : UInt<1>, wb_sel : UInt<2>, A_sel : UInt<1>, B_sel : UInt<1>, alu_op : UInt<4>, br_type : UInt<3>, data_re : UInt<1>, st_type : UInt<2>, ld_type : UInt<3>, csr_cmd : UInt<2>}
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
    node T_560 = sub-wrap(UInt<14>(8192), UInt<3>(4))
    reg pc : UInt<14>
    on-reset pc := T_560
    node T_561 = eq(ctrl.pc_sel, UInt<1>(1))
    node T_562 = bit-or(T_561, brCond.taken)
    node T_563 = add-wrap(pc, UInt<3>(4))
    node iaddr = mux(T_562, alu.sum, T_563)
    node T_564 = eq(ctrl.inst_type, UInt<1>(1))
    node T_565 = bit-or(T_564, brCond.taken)
    node inst_425 = mux(T_565, UInt<32>(19), icache.dout)
    icache.addr := iaddr
    icache.re := ctrl.inst_re
    node T_566 = eq(dcache.we, UInt<1>(0))
    node T_567 = bit-not(T_566)
    node T_568 = bit-not(T_567)
    node T_569 = bit-and(icache.re, T_568)
    node T_570 = mux(T_569, iaddr, pc)
    pc := T_570
    node T_571 = bit-not(stall)
    when T_571 : 
      fe_pc := pc
      fe_inst := inst_425
    ctrl.inst_424 := fe_inst
    ctrl.stall := stall
    node rd_addr = bits(fe_inst, 11, 7)
    node rs1_addr = bits(fe_inst, 19, 15)
    node rs2_addr = bits(fe_inst, 24, 20)
    regFile.raddr1 := rs1_addr
    regFile.raddr2 := rs2_addr
    immGen.inst_252 := fe_inst
    immGen.sel := ctrl.imm_sel
    node T_572 = eq(rs1_addr, UInt<1>(0))
    node rs1NotZero = bit-not(T_572)
    node T_573 = eq(rs2_addr, UInt<1>(0))
    node rs2NotZero = bit-not(T_573)
    node T_574 = eq(ctrl.wb_sel, UInt<2>(0))
    node alutype = bit-and(ctrl.wb_en, T_574)
    node ex_rd_addr = bits(ew_inst, 11, 7)
    node T_575 = bit-and(alutype, rs1NotZero)
    node T_576 = eq(rs1_addr, ex_rd_addr)
    node T_577 = bit-and(T_575, T_576)
    node rs1 = mux(T_577, ew_alu, regFile.rdata1)
    node T_578 = bit-and(alutype, rs2NotZero)
    node T_579 = eq(rs2_addr, ex_rd_addr)
    node T_580 = bit-and(T_578, T_579)
    node rs2 = mux(T_580, ew_alu, regFile.rdata2)
    node T_581 = eq(ctrl.A_sel, UInt<1>(0))
    node T_582 = mux(T_581, rs1, fe_pc)
    alu.A := T_582
    node T_583 = eq(ctrl.B_sel, UInt<1>(0))
    node T_584 = mux(T_583, rs2, immGen.out)
    alu.B := T_584
    alu.alu_op := ctrl.alu_op
    brCond.rs1 := rs1
    brCond.rs2 := rs2
    brCond.br_type := ctrl.br_type
    node T_585 = bit(alu.sum, 1)
    node T_586 = dshl(T_585, UInt<3>(4))
    node T_587 = bit(alu.sum, 0)
    node T_588 = dshl(T_587, UInt<2>(3))
    node woffset = bit-or(T_586, T_588)
    dcache.re := ctrl.data_re
    node T_589 = mux(stall, ew_alu, alu.sum)
    dcache.addr := T_589
    node T_590 = bits(alu.sum, 1, 0)
    node T_591 = dshl(UInt<2>(3), T_590)
    node T_592 = bits(alu.sum, 1, 0)
    node T_593 = dshl(UInt<1>(1), T_592)
    node T_594 = eq(UInt<2>(2), ctrl.st_type)
    node T_595 = mux(T_594, T_593, UInt<4>(0))
    node T_596 = eq(UInt<2>(1), ctrl.st_type)
    node T_597 = mux(T_596, T_591, T_595)
    node T_598 = eq(UInt<2>(0), ctrl.st_type)
    node T_599 = mux(T_598, UInt<4>(15), T_597)
    node T_600 = mux(stall, UInt<4>(0), T_599)
    dcache.we := T_600
    node T_601 = dshl(rs2, woffset)
    dcache.din := T_601
    node T_602 = bit-not(stall)
    when T_602 : 
      ew_pc := fe_pc
      ew_inst := fe_inst
      ew_alu := alu.out
    node T_603 = bit(ew_alu, 1)
    node T_604 = dshl(T_603, UInt<3>(4))
    node T_605 = bit(ew_alu, 0)
    node T_606 = dshl(T_605, UInt<2>(3))
    node loffset = bit-or(T_604, T_606)
    node lshift = dshr(dcache.dout, loffset)
    node T_607 = as-SInt(dcache.dout)
    node T_608 = bits(lshift, 15, 0)
    node T_609 = as-SInt(T_608)
    node T_610 = bits(lshift, 7, 0)
    node T_611 = as-SInt(T_610)
    node T_612 = bits(lshift, 15, 0)
    node T_613 = convert(T_612)
    node T_614 = bits(lshift, 7, 0)
    node T_615 = convert(T_614)
    node T_616 = eq(UInt<3>(4), ctrl.ld_type)
    node T_617 = mux(T_616, T_615, T_607)
    node T_618 = eq(UInt<3>(3), ctrl.ld_type)
    node T_619 = mux(T_618, T_613, T_617)
    node T_620 = eq(UInt<3>(2), ctrl.ld_type)
    node T_621 = mux(T_620, T_611, T_619)
    node T_622 = eq(UInt<3>(1), ctrl.ld_type)
    node T_623 = mux(T_622, T_609, T_621)
    node load = as-UInt(T_623)
    inst csr of CSR
    csr.host <> host
    csr.src := ew_alu
    node T_624 = bits(ew_inst, 31, 20)
    csr.addr := T_624
    csr.cmd := ctrl.csr_cmd
    node T_625 = add-wrap(ew_pc, UInt<3>(4))
    node T_626 = eq(UInt<2>(3), ctrl.wb_sel)
    node T_627 = mux(T_626, csr.data, ew_alu)
    node T_628 = eq(UInt<2>(2), ctrl.wb_sel)
    node T_629 = mux(T_628, T_625, T_627)
    node T_630 = eq(UInt<2>(1), ctrl.wb_sel)
    node regWrite = mux(T_630, load, T_629)
    regFile.wen := ctrl.wb_en
    regFile.waddr := ex_rd_addr
    regFile.wdata := regWrite
