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
    node T_433 = addw(A, B)
    node T_434 = subw(A, B)
    node T_435 = asSInt(A)
    node T_436 = dshr(T_435, shamt)
    node T_437 = asUInt(T_436)
    node T_438 = dshr(A, shamt)
    node T_439 = dshl(A, shamt)
    node T_440 = bits(T_439, 31, 0)
    node T_441 = asSInt(A)
    node T_442 = asSInt(B)
    node T_443 = lt(T_441, T_442)
    node T_444 = asUInt(T_443)
    node T_445 = lt(A, B)
    node T_446 = asUInt(T_445)
    node T_447 = and(A, B)
    node T_448 = or(A, B)
    node T_449 = xor(A, B)
    node T_450 = eq(UInt<4>(10), alu_op)
    node T_451 = mux(T_450, A, B)
    node T_452 = eq(UInt<4>(4), alu_op)
    node T_453 = mux(T_452, T_449, T_451)
    node T_454 = eq(UInt<4>(3), alu_op)
    node T_455 = mux(T_454, T_448, T_453)
    node T_456 = eq(UInt<4>(2), alu_op)
    node T_457 = mux(T_456, T_447, T_455)
    node T_458 = eq(UInt<4>(7), alu_op)
    node T_459 = mux(T_458, T_446, T_457)
    node T_460 = eq(UInt<4>(5), alu_op)
    node T_461 = mux(T_460, T_444, T_459)
    node T_462 = eq(UInt<4>(6), alu_op)
    node T_463 = mux(T_462, T_440, T_461)
    node T_464 = eq(UInt<4>(8), alu_op)
    node T_465 = mux(T_464, T_438, T_463)
    node T_466 = eq(UInt<4>(9), alu_op)
    node T_467 = mux(T_466, T_437, T_465)
    node T_468 = eq(UInt<4>(1), alu_op)
    node T_469 = mux(T_468, T_434, T_467)
    node T_470 = eq(UInt<4>(0), alu_op)
    node oot = mux(T_470, T_433, T_469)
    node T_471 = bits(oot, 31, 0)
    out := T_471
    node T_472 = bit(alu_op, 0)
    node T_473 = subw(UInt<1>(0), B)
    node T_474 = mux(T_472, T_473, B)
    node T_475 = addw(A, T_474)
    sum := T_475
  module BrCond : 
    input rs1 : UInt<32>
    input rs2 : UInt<32>
    output taken : UInt<1>
    input br_type : UInt<3>
    
    node eq = eq(rs1, rs2)
    node neq = not(eq)
    node T_476 = asSInt(rs1)
    node T_477 = asSInt(rs2)
    node lt = lt(T_476, T_477)
    node ge = not(lt)
    node ltu = lt(rs1, rs2)
    node geu = not(ltu)
    node T_478 = eq(br_type, UInt<3>(2))
    node T_479 = and(T_478, eq)
    node T_480 = eq(br_type, UInt<3>(6))
    node T_481 = and(T_480, neq)
    node T_482 = or(T_479, T_481)
    node T_483 = eq(br_type, UInt<3>(1))
    node T_484 = and(T_483, lt)
    node T_485 = or(T_482, T_484)
    node T_486 = eq(br_type, UInt<3>(5))
    node T_487 = and(T_486, ge)
    node T_488 = or(T_485, T_487)
    node T_489 = eq(br_type, UInt<3>(0))
    node T_490 = and(T_489, ltu)
    node T_491 = or(T_488, T_490)
    node T_492 = eq(br_type, UInt<3>(4))
    node T_493 = and(T_492, geu)
    node T_494 = or(T_491, T_493)
    taken := T_494
  module RegFile : 
    input clk : Clock
    input raddr1 : UInt<5>
    input raddr2 : UInt<5>
    output rdata1 : UInt<32>
    output rdata2 : UInt<32>
    input wen : UInt<1>
    input waddr : UInt<5>
    input wdata : UInt<32>
    
    cmem regs : UInt<32>[32], clk
    node T_495 = eq(raddr1, UInt<1>(0))
    node T_496 = not(T_495)
    infer accessor T_497 = regs[raddr1]
    node T_498 = mux(T_496, T_497, UInt<1>(0))
    rdata1 := T_498
    node T_499 = eq(raddr2, UInt<1>(0))
    node T_500 = not(T_499)
    infer accessor T_501 = regs[raddr2]
    node T_502 = mux(T_500, T_501, UInt<1>(0))
    rdata2 := T_502
    node T_503 = eq(waddr, UInt<1>(0))
    node T_504 = not(T_503)
    node T_505 = and(wen, T_504)
    when T_505 : 
      infer accessor T_506 = regs[waddr]
      T_506 := wdata
  module ImmGenWire : 
    output out : UInt<32>
    input inst : UInt<32>
    input sel : UInt<3>
    
    node T_507 = bits(inst, 31, 20)
    node Iimm = asSInt(T_507)
    node T_508 = bits(inst, 31, 25)
    node T_509 = bits(inst, 11, 7)
    node T_510 = cat(T_508, T_509)
    node Simm = asSInt(T_510)
    node T_511 = bit(inst, 31)
    node T_512 = bit(inst, 7)
    node T_513 = bits(inst, 30, 25)
    node T_514 = bits(inst, 11, 8)
    node T_515 = cat(T_511, T_512)
    node T_516 = cat(T_514, UInt<1>(0))
    node T_517 = cat(T_513, T_516)
    node T_518 = cat(T_515, T_517)
    node Bimm = asSInt(T_518)
    node T_519 = bits(inst, 31, 12)
    node T_520 = cat(T_519, UInt<12>(0))
    node Uimm = asSInt(T_520)
    node T_521 = bit(inst, 31)
    node T_522 = bits(inst, 19, 12)
    node T_523 = bit(inst, 20)
    node T_524 = bits(inst, 30, 25)
    node T_525 = bits(inst, 24, 21)
    node T_526 = cat(T_522, T_523)
    node T_527 = cat(T_521, T_526)
    node T_528 = cat(T_525, UInt<1>(0))
    node T_529 = cat(T_524, T_528)
    node T_530 = cat(T_527, T_529)
    node Jimm = asSInt(T_530)
    node T_531 = bits(inst, 19, 15)
    node T_532 = pad(T_531, 32)
    node Zimm = asSInt(T_532)
    node T_533 = eq(UInt<3>(3), sel)
    node T_534 = mux(T_533, Jimm, Zimm)
    node T_535 = eq(UInt<3>(2), sel)
    node T_536 = mux(T_535, Uimm, T_534)
    node T_537 = eq(UInt<3>(4), sel)
    node T_538 = mux(T_537, Bimm, T_536)
    node T_539 = eq(UInt<3>(1), sel)
    node T_540 = mux(T_539, Simm, T_538)
    node T_541 = eq(UInt<3>(0), sel)
    node T_542 = mux(T_541, Iimm, T_540)
    node T_543 = asUInt(T_542)
    out := T_543
  module CSR : 
    input clk : Clock
    input reset : UInt<1>
    output host : {status : UInt<32>, tohost : UInt<32>, flip hid : UInt<1>}
    input src : UInt<32>
    input cmd : UInt<2>
    output data : UInt<32>
    input addr : UInt<12>
    
    reg reg_tohost : UInt<32>, clk, reset
    onreset reg_tohost := UInt<32>(0)
    reg reg_status : UInt<32>, clk, reset
    onreset reg_status := UInt<32>(0)
    host.tohost := reg_tohost
    host.status := reg_status
    node T_544 = eq(UInt<12>(1291), addr)
    node T_545 = mux(T_544, host.hid, UInt<1>(0))
    node T_546 = eq(UInt<12>(1290), addr)
    node T_547 = mux(T_546, reg_status, T_545)
    node T_548 = eq(UInt<12>(1310), addr)
    node T_549 = mux(T_548, reg_tohost, T_547)
    data := T_549
    node T_550 = eq(cmd, UInt<2>(1))
    when T_550 : 
      node T_551 = eq(addr, UInt<12>(1310))
      when T_551 : reg_tohost := src
      node T_552 = eq(addr, UInt<12>(1290))
      when T_552 : reg_status := src
    node T_553 = eq(cmd, UInt<2>(2))
    node T_554 = neq(src, UInt<1>(0))
    node T_555 = and(T_553, T_554)
    when T_555 : 
      node T_556 = eq(addr, UInt<12>(1310))
      when T_556 : 
        node T_557 = dshl(UInt<1>(1), bits(src,5,0))
        node T_558 = or(data, T_557)
        reg_tohost := T_558
      node T_559 = eq(addr, UInt<12>(1290))
      when T_559 : 
        node T_560 = dshl(UInt<1>(1), bits(src,5,0))
        node T_561 = or(data, T_560)
        reg_status := T_561
    node T_562 = eq(cmd, UInt<2>(3))
    node T_563 = neq(src, UInt<1>(0))
    node T_564 = and(T_562, T_563)
    when T_564 : 
      node T_565 = eq(addr, UInt<12>(1310))
      when T_565 : 
        node T_566 = dshl(UInt<1>(0), bits(src,5,0))
        node T_567 = and(data, T_566)
        reg_tohost := T_567
      node T_568 = eq(addr, UInt<12>(1290))
      when T_568 : 
        node T_569 = dshl(UInt<1>(0), bits(src,5,0))
        node T_570 = and(data, T_569)
        reg_status := T_570
  module Datapath : 
    input clk : Clock
    input reset : UInt<1>
    output host : {status : UInt<32>, tohost : UInt<32>, flip hid : UInt<1>}
    input ctrl : {flip inst : UInt<32>, pc_sel : UInt<1>, inst_type : UInt<1>, inst_re : UInt<1>, flip stall : UInt<1>, imm_sel : UInt<3>, wb_en : UInt<1>, wb_sel : UInt<2>, A_sel : UInt<1>, B_sel : UInt<1>, alu_op : UInt<4>, br_type : UInt<3>, data_re : UInt<1>, st_type : UInt<2>, ld_type : UInt<3>, csr_cmd : UInt<2>}
    output icache : {re : UInt<1>, flip dout : UInt<32>, we : UInt<4>, addr : UInt<32>, din : UInt<32>}
    output dcache : {re : UInt<1>, flip dout : UInt<32>, we : UInt<4>, addr : UInt<32>, din : UInt<32>}
    input stall : UInt<1>
    
    inst alu of ALU
    inst brCond of BrCond
    inst regFile of RegFile
    regFile.clk := clk
    inst immGen of ImmGenWire
    reg fe_inst : UInt<32>, clk, reset
    onreset fe_inst := UInt<32>(0)
    reg fe_pc : UInt, clk, reset
    reg ew_inst : UInt<32>, clk, reset
    onreset ew_inst := UInt<32>(0)
    reg ew_pc : UInt, clk, reset
    reg ew_alu : UInt, clk, reset
    node T_571 = subw(UInt<14>(8192), UInt<32>(4))
    reg pc : UInt<32>, clk, reset
    onreset pc := T_571
    node T_572 = eq(ctrl.pc_sel, UInt<1>(1))
    node T_573 = or(T_572, brCond.taken)
    node T_574 = addw(pc, UInt<3>(4))
    node iaddr = mux(T_573, alu.sum, T_574)
    node T_575 = eq(ctrl.inst_type, UInt<1>(1))
    node T_576 = or(T_575, brCond.taken)
    node inst = mux(T_576, UInt<32>(19), icache.dout)
    icache.we := UInt<1>(0)
    icache.din := UInt<1>(0)
    icache.addr := iaddr
    icache.re := ctrl.inst_re
    node T_577 = eq(dcache.we, UInt<1>(0))
    node T_578 = not(T_577)
    node T_579 = not(T_578)
    node T_580 = and(icache.re, T_579)
    node T_581 = mux(T_580, iaddr, pc)
    pc := T_581
    node T_582 = not(stall)
    when T_582 : 
      fe_pc := pc
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
    node T_583 = eq(rs1_addr, UInt<1>(0))
    node rs1NotZero = not(T_583)
    node T_584 = eq(rs2_addr, UInt<1>(0))
    node rs2NotZero = not(T_584)
    node T_585 = eq(ctrl.wb_sel, UInt<2>(0))
    node alutype = and(ctrl.wb_en, T_585)
    node ex_rd_addr = bits(ew_inst, 11, 7)
    node T_586 = and(alutype, rs1NotZero)
    node T_587 = eq(rs1_addr, ex_rd_addr)
    node T_588 = and(T_586, T_587)
    node rs1 = mux(T_588, ew_alu, regFile.rdata1)
    node T_589 = and(alutype, rs2NotZero)
    node T_590 = eq(rs2_addr, ex_rd_addr)
    node T_591 = and(T_589, T_590)
    node rs2 = mux(T_591, ew_alu, regFile.rdata2)
    node T_592 = eq(ctrl.A_sel, UInt<1>(0))
    node T_593 = mux(T_592, rs1, fe_pc)
    alu.A := T_593
    node T_594 = eq(ctrl.B_sel, UInt<1>(0))
    node T_595 = mux(T_594, rs2, immGen.out)
    alu.B := T_595
    alu.alu_op := ctrl.alu_op
    brCond.rs1 := rs1
    brCond.rs2 := rs2
    brCond.br_type := ctrl.br_type
    node T_596 = bit(alu.sum, 1)
    node T_597 = dshl(T_596, UInt<3>(4))
    node T_598 = bit(alu.sum, 0)
    node T_599 = dshl(T_598, UInt<2>(3))
    node woffset = or(T_597, T_599)
    dcache.re := ctrl.data_re
    node T_600 = mux(stall, ew_alu, alu.sum)
    dcache.addr := T_600
    node T_601 = bits(alu.sum, 1, 0)
    node T_602 = dshl(UInt<2>(3), T_601)
    node T_603 = bits(T_602, 3, 0)
    node T_604 = bits(alu.sum, 1, 0)
    node T_605 = dshl(UInt<1>(1), T_604)
    node T_606 = bits(T_605, 3, 0)
    node T_607 = eq(UInt<2>(2), ctrl.st_type)
    node T_608 = mux(T_607, T_606, UInt<4>(0))
    node T_609 = eq(UInt<2>(1), ctrl.st_type)
    node T_610 = mux(T_609, T_603, T_608)
    node T_611 = eq(UInt<2>(0), ctrl.st_type)
    node T_612 = mux(T_611, UInt<4>(15), T_610)
    node T_613 = mux(stall, UInt<4>(0), T_612)
    dcache.we := T_613
    node T_614 = dshl(rs2, woffset)
    node T_615 = bits(T_614, 31, 0)
    dcache.din := T_615
    node T_616 = not(stall)
    when T_616 : 
      ew_pc := fe_pc
      ew_inst := fe_inst
      ew_alu := alu.out
    node T_617 = bit(ew_alu, 1)
    node T_618 = dshl(T_617, UInt<3>(4))
    node T_619 = bit(ew_alu, 0)
    node T_620 = dshl(T_619, UInt<2>(3))
    node loffset = or(T_618, T_620)
    node lshift = dshr(dcache.dout, loffset)
    node T_621 = bits(lshift, 15, 0)
    node T_622 = asSInt(T_621)
    node T_623 = pad(T_622, 32)
    node T_624 = asUInt(T_623)
    node T_625 = bits(lshift, 7, 0)
    node T_626 = asSInt(T_625)
    node T_627 = pad(T_626, 32)
    node T_628 = asUInt(T_627)
    node T_629 = bits(lshift, 15, 0)
    node T_630 = bits(lshift, 7, 0)
    node T_631 = eq(UInt<3>(4), ctrl.ld_type)
    node T_632 = mux(T_631, T_630, dcache.dout)
    node T_633 = eq(UInt<3>(3), ctrl.ld_type)
    node T_634 = mux(T_633, T_629, T_632)
    node T_635 = eq(UInt<3>(2), ctrl.ld_type)
    node T_636 = mux(T_635, T_628, T_634)
    node T_637 = eq(UInt<3>(1), ctrl.ld_type)
    node load = mux(T_637, T_624, T_636)
    inst csr of CSR
    csr.reset := reset
    csr.clk := clk
    host := csr.host
    csr.src := ew_alu
    node T_638 = bits(ew_inst, 31, 20)
    csr.addr := T_638
    csr.cmd := ctrl.csr_cmd
    node T_639 = addw(ew_pc, UInt<3>(4))
    node T_640 = eq(UInt<2>(3), ctrl.wb_sel)
    node T_641 = mux(T_640, csr.data, ew_alu)
    node T_642 = eq(UInt<2>(2), ctrl.wb_sel)
    node T_643 = mux(T_642, T_639, T_641)
    node T_644 = eq(UInt<2>(1), ctrl.wb_sel)
    node regWrite = mux(T_644, load, T_643)
    regFile.wen := ctrl.wb_en
    regFile.waddr := ex_rd_addr
    regFile.wdata := regWrite
