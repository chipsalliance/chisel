; RUN: firrtl -i %s -o %s.v -X verilog -p c | tee %s.out | FileCheck %s
;CHECK: Done!

circuit Tile : 
  module ALU : 
    input B : UInt<32>
    output out : UInt<32>
    output sum : UInt<32>
    input A : UInt<32>
    input alu_op : UInt<4>
    
    node shamt = bits(B, 4, 0)
    node T_1554 = add-wrap(A, B)
    node T_1555 = sub-wrap(A, B)
    node T_1556 = as-SInt(A)
    node T_1557 = dshr(T_1556, shamt)
    node T_1558 = as-UInt(T_1557)
    node T_1559 = dshr(A, shamt)
    node T_1560 = dshl(A, shamt)
    node T_1561 = bits(T_1560, 31, 0)
    node T_1562 = as-SInt(A)
    node T_1563 = as-SInt(B)
    node T_1564 = lt(T_1562, T_1563)
    node T_1565 = as-UInt(T_1564)
    node T_1566 = lt(A, B)
    node T_1567 = as-UInt(T_1566)
    node T_1568 = bit-and(A, B)
    node T_1569 = bit-or(A, B)
    node T_1570 = bit-xor(A, B)
    node T_1571 = eq(UInt<4>(10), alu_op)
    node T_1572 = mux(T_1571, A, B)
    node T_1573 = eq(UInt<4>(4), alu_op)
    node T_1574 = mux(T_1573, T_1570, T_1572)
    node T_1575 = eq(UInt<4>(3), alu_op)
    node T_1576 = mux(T_1575, T_1569, T_1574)
    node T_1577 = eq(UInt<4>(2), alu_op)
    node T_1578 = mux(T_1577, T_1568, T_1576)
    node T_1579 = eq(UInt<4>(7), alu_op)
    node T_1580 = mux(T_1579, T_1567, T_1578)
    node T_1581 = eq(UInt<4>(5), alu_op)
    node T_1582 = mux(T_1581, T_1565, T_1580)
    node T_1583 = eq(UInt<4>(6), alu_op)
    node T_1584 = mux(T_1583, T_1561, T_1582)
    node T_1585 = eq(UInt<4>(8), alu_op)
    node T_1586 = mux(T_1585, T_1559, T_1584)
    node T_1587 = eq(UInt<4>(9), alu_op)
    node T_1588 = mux(T_1587, T_1558, T_1586)
    node T_1589 = eq(UInt<4>(1), alu_op)
    node T_1590 = mux(T_1589, T_1555, T_1588)
    node T_1591 = eq(UInt<4>(0), alu_op)
    node oot = mux(T_1591, T_1554, T_1590)
    node T_1592 = bits(oot, 31, 0)
    out := T_1592
    node T_1593 = bit(alu_op, 0)
    node T_1594 = sub-wrap(UInt<1>(0), B)
    node T_1595 = mux(T_1593, T_1594, B)
    node T_1596 = add-wrap(A, T_1595)
    sum := T_1596
  module BrCond : 
    input br_type : UInt<3>
    input rs1 : UInt<32>
    input rs2 : UInt<32>
    output taken : UInt<1>
    
    node eq = eq(rs1, rs2)
    node neq = bit-not(eq)
    node T_1597 = as-SInt(rs1)
    node T_1598 = as-SInt(rs2)
    node lt = lt(T_1597, T_1598)
    node ge = bit-not(lt)
    node ltu = lt(rs1, rs2)
    node geu = bit-not(ltu)
    node T_1599 = eq(br_type, UInt<3>(2))
    node T_1600 = bit-and(T_1599, eq)
    node T_1601 = eq(br_type, UInt<3>(6))
    node T_1602 = bit-and(T_1601, neq)
    node T_1603 = bit-or(T_1600, T_1602)
    node T_1604 = eq(br_type, UInt<3>(1))
    node T_1605 = bit-and(T_1604, lt)
    node T_1606 = bit-or(T_1603, T_1605)
    node T_1607 = eq(br_type, UInt<3>(5))
    node T_1608 = bit-and(T_1607, ge)
    node T_1609 = bit-or(T_1606, T_1608)
    node T_1610 = eq(br_type, UInt<3>(0))
    node T_1611 = bit-and(T_1610, ltu)
    node T_1612 = bit-or(T_1609, T_1611)
    node T_1613 = eq(br_type, UInt<3>(4))
    node T_1614 = bit-and(T_1613, geu)
    node T_1615 = bit-or(T_1612, T_1614)
    taken := T_1615
  module RegFile : 
    input raddr1 : UInt<5>
    input raddr2 : UInt<5>
    output rdata1 : UInt<32>
    output rdata2 : UInt<32>
    input wen : UInt<1>
    input waddr : UInt<5>
    input wdata : UInt<32>
    
    cmem regs : UInt<32>[32]
    node T_1616 = eq(raddr1, UInt<1>(0))
    node T_1617 = bit-not(T_1616)
    infer accessor T_1618 = regs[raddr1]
    node T_1619 = mux(T_1617, T_1618, UInt<1>(0))
    rdata1 := T_1619
    node T_1620 = eq(raddr2, UInt<1>(0))
    node T_1621 = bit-not(T_1620)
    infer accessor T_1622 = regs[raddr2]
    node T_1623 = mux(T_1621, T_1622, UInt<1>(0))
    rdata2 := T_1623
    node T_1624 = eq(waddr, UInt<1>(0))
    node T_1625 = bit-not(T_1624)
    node T_1626 = bit-and(wen, T_1625)
    when T_1626 : 
      infer accessor T_1627 = regs[waddr]
      T_1627 := wdata
  module ImmGenWire : 
    output out : UInt<32>
    input inst : UInt<32>
    input sel : UInt<3>
    
    node T_1628 = bits(inst, 31, 20)
    node Iimm = as-SInt(T_1628)
    node T_1629 = bits(inst, 31, 25)
    node T_1630 = bits(inst, 11, 7)
    node T_1631 = cat(T_1629, T_1630)
    node Simm = as-SInt(T_1631)
    node T_1632 = bit(inst, 31)
    node T_1633 = bit(inst, 7)
    node T_1634 = bits(inst, 30, 25)
    node T_1635 = bits(inst, 11, 8)
    node T_1636 = cat(T_1632, T_1633)
    node T_1637 = cat(T_1635, UInt<1>(0))
    node T_1638 = cat(T_1634, T_1637)
    node T_1639 = cat(T_1636, T_1638)
    node Bimm = as-SInt(T_1639)
    node T_1640 = bits(inst, 31, 12)
    node T_1641 = cat(T_1640, UInt<12>(0))
    node Uimm = as-SInt(T_1641)
    node T_1642 = bit(inst, 31)
    node T_1643 = bits(inst, 19, 12)
    node T_1644 = bit(inst, 20)
    node T_1645 = bits(inst, 30, 25)
    node T_1646 = bits(inst, 24, 21)
    node T_1647 = cat(T_1643, T_1644)
    node T_1648 = cat(T_1642, T_1647)
    node T_1649 = cat(T_1646, UInt<1>(0))
    node T_1650 = cat(T_1645, T_1649)
    node T_1651 = cat(T_1648, T_1650)
    node Jimm = as-SInt(T_1651)
    node T_1652 = bits(inst, 19, 15)
    node T_1653 = pad(T_1652, 32)
    node Zimm = as-SInt(T_1653)
    node T_1654 = eq(UInt<3>(3), sel)
    node T_1655 = mux(T_1654, Jimm, Zimm)
    node T_1656 = eq(UInt<3>(2), sel)
    node T_1657 = mux(T_1656, Uimm, T_1655)
    node T_1658 = eq(UInt<3>(4), sel)
    node T_1659 = mux(T_1658, Bimm, T_1657)
    node T_1660 = eq(UInt<3>(1), sel)
    node T_1661 = mux(T_1660, Simm, T_1659)
    node T_1662 = eq(UInt<3>(0), sel)
    node T_1663 = mux(T_1662, Iimm, T_1661)
    node T_1664 = as-UInt(T_1663)
    out := T_1664
  module CSR : 
    output host : {status : UInt<32>, flip hid : UInt<1>, tohost : UInt<32>}
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
    node T_1665 = eq(UInt<12>(1291), addr)
    node T_1666 = mux(T_1665, host.hid, UInt<1>(0))
    node T_1667 = eq(UInt<12>(1290), addr)
    node T_1668 = mux(T_1667, reg_status, T_1666)
    node T_1669 = eq(UInt<12>(1310), addr)
    node T_1670 = mux(T_1669, reg_tohost, T_1668)
    data := T_1670
    node T_1671 = eq(cmd, UInt<2>(1))
    when T_1671 : 
      node T_1672 = eq(addr, UInt<12>(1310))
      when T_1672 : reg_tohost := src
      node T_1673 = eq(addr, UInt<12>(1290))
      when T_1673 : reg_status := src
    node T_1674 = eq(cmd, UInt<2>(2))
    node T_1675 = neq(src, UInt<1>(0))
    node T_1676 = bit-and(T_1674, T_1675)
    when T_1676 : 
      node T_1677 = eq(addr, UInt<12>(1310))
      when T_1677 : 
        node T_1678 = dshl(UInt<1>(1), bits(src,5,0))
        node T_1679 = bit-or(data, T_1678)
        reg_tohost := T_1679
      node T_1680 = eq(addr, UInt<12>(1290))
      when T_1680 : 
        node T_1681 = dshl(UInt<1>(1), bits(src,5,0))
        node T_1682 = bit-or(data, T_1681)
        reg_status := T_1682
    node T_1683 = eq(cmd, UInt<2>(3))
    node T_1684 = neq(src, UInt<1>(0))
    node T_1685 = bit-and(T_1683, T_1684)
    when T_1685 : 
      node T_1686 = eq(addr, UInt<12>(1310))
      when T_1686 : 
        node T_1687 = dshl(UInt<1>(0), bits(src,5,0))
        node T_1688 = bit-and(data, T_1687)
        reg_tohost := T_1688
      node T_1689 = eq(addr, UInt<12>(1290))
      when T_1689 : 
        node T_1690 = dshl(UInt<1>(0), bits(src,5,0))
        node T_1691 = bit-and(data, T_1690)
        reg_status := T_1691
  module Datapath : 
    output host : {status : UInt<32>, flip hid : UInt<1>, tohost : UInt<32>}
    input stall : UInt<1>
    output icache : {re : UInt<1>, flip dout : UInt<32>, we : UInt<4>, addr : UInt<32>, din : UInt<32>}
    output dcache : {re : UInt<1>, flip dout : UInt<32>, we : UInt<4>, addr : UInt<32>, din : UInt<32>}
    input ctrl : {flip inst : UInt<32>, flip stall : UInt<1>, inst_re : UInt<1>, inst_type : UInt<1>, pc_sel : UInt<1>, A_sel : UInt<1>, B_sel : UInt<1>, imm_sel : UInt<3>, alu_op : UInt<4>, br_type : UInt<3>, data_re : UInt<1>, st_type : UInt<2>, ld_type : UInt<3>, wb_sel : UInt<2>, wb_en : UInt<1>, csr_cmd : UInt<2>}
    
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
    node T_1692 = sub-wrap(UInt<14>(8192), UInt<32>(4))
    reg pc : UInt<32>
    on-reset pc := T_1692
    node T_1693 = eq(ctrl.pc_sel, UInt<1>(1))
    node T_1694 = bit-or(T_1693, brCond.taken)
    node T_1695 = add-wrap(pc, UInt<3>(4))
    node iaddr = mux(T_1694, alu.sum, T_1695)
    node T_1696 = eq(ctrl.inst_type, UInt<1>(1))
    node T_1697 = bit-or(T_1696, brCond.taken)
    node inst = mux(T_1697, UInt<32>(19), icache.dout)
    icache.we := UInt<1>(0)
    icache.din := UInt<1>(0)
    icache.addr := iaddr
    icache.re := ctrl.inst_re
    node T_1698 = eq(dcache.we, UInt<1>(0))
    node T_1699 = bit-not(T_1698)
    node T_1700 = bit-not(T_1699)
    node T_1701 = bit-and(icache.re, T_1700)
    node T_1702 = mux(T_1701, iaddr, pc)
    pc := T_1702
    node T_1703 = bit-not(stall)
    when T_1703 : 
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
    node T_1704 = eq(rs1_addr, UInt<1>(0))
    node rs1NotZero = bit-not(T_1704)
    node T_1705 = eq(rs2_addr, UInt<1>(0))
    node rs2NotZero = bit-not(T_1705)
    node T_1706 = eq(ctrl.wb_sel, UInt<2>(0))
    node alutype = bit-and(ctrl.wb_en, T_1706)
    node ex_rd_addr = bits(ew_inst, 11, 7)
    node T_1707 = bit-and(alutype, rs1NotZero)
    node T_1708 = eq(rs1_addr, ex_rd_addr)
    node T_1709 = bit-and(T_1707, T_1708)
    node rs1 = mux(T_1709, ew_alu, regFile.rdata1)
    node T_1710 = bit-and(alutype, rs2NotZero)
    node T_1711 = eq(rs2_addr, ex_rd_addr)
    node T_1712 = bit-and(T_1710, T_1711)
    node rs2 = mux(T_1712, ew_alu, regFile.rdata2)
    node T_1713 = eq(ctrl.A_sel, UInt<1>(0))
    node T_1714 = mux(T_1713, rs1, fe_pc)
    alu.A := T_1714
    node T_1715 = eq(ctrl.B_sel, UInt<1>(0))
    node T_1716 = mux(T_1715, rs2, immGen.out)
    alu.B := T_1716
    alu.alu_op := ctrl.alu_op
    brCond.rs1 := rs1
    brCond.rs2 := rs2
    brCond.br_type := ctrl.br_type
    node T_1717 = bit(alu.sum, 1)
    node T_1718 = dshl(T_1717, UInt<3>(4))
    node T_1719 = bit(alu.sum, 0)
    node T_1720 = dshl(T_1719, UInt<2>(3))
    node woffset = bit-or(T_1718, T_1720)
    dcache.re := ctrl.data_re
    node T_1721 = mux(stall, ew_alu, alu.sum)
    dcache.addr := T_1721
    node T_1722 = bits(alu.sum, 1, 0)
    node T_1723 = dshl(UInt<2>(3), T_1722)
    node T_1724 = bits(T_1723, 3, 0)
    node T_1725 = bits(alu.sum, 1, 0)
    node T_1726 = dshl(UInt<1>(1), T_1725)
    node T_1727 = bits(T_1726, 3, 0)
    node T_1728 = eq(UInt<2>(2), ctrl.st_type)
    node T_1729 = mux(T_1728, T_1727, UInt<4>(0))
    node T_1730 = eq(UInt<2>(1), ctrl.st_type)
    node T_1731 = mux(T_1730, T_1724, T_1729)
    node T_1732 = eq(UInt<2>(0), ctrl.st_type)
    node T_1733 = mux(T_1732, UInt<4>(15), T_1731)
    node T_1734 = mux(stall, UInt<4>(0), T_1733)
    dcache.we := T_1734
    node T_1735 = dshl(rs2, woffset)
    node T_1736 = bits(T_1735, 31, 0)
    dcache.din := T_1736
    node T_1737 = bit-not(stall)
    when T_1737 : 
      ew_pc := fe_pc
      ew_inst := fe_inst
      ew_alu := alu.out
    node T_1738 = bit(ew_alu, 1)
    node T_1739 = dshl(T_1738, UInt<3>(4))
    node T_1740 = bit(ew_alu, 0)
    node T_1741 = dshl(T_1740, UInt<2>(3))
    node loffset = bit-or(T_1739, T_1741)
    node lshift = dshr(dcache.dout, loffset)
    node T_1742 = bits(lshift, 15, 0)
    node T_1743 = as-SInt(T_1742)
    node T_1744 = pad(T_1743, 32)
    node T_1745 = as-UInt(T_1744)
    node T_1746 = bits(lshift, 7, 0)
    node T_1747 = as-SInt(T_1746)
    node T_1748 = pad(T_1747, 32)
    node T_1749 = as-UInt(T_1748)
    node T_1750 = bits(lshift, 15, 0)
    node T_1751 = bits(lshift, 7, 0)
    node T_1752 = eq(UInt<3>(4), ctrl.ld_type)
    node T_1753 = mux(T_1752, T_1751, dcache.dout)
    node T_1754 = eq(UInt<3>(3), ctrl.ld_type)
    node T_1755 = mux(T_1754, T_1750, T_1753)
    node T_1756 = eq(UInt<3>(2), ctrl.ld_type)
    node T_1757 = mux(T_1756, T_1749, T_1755)
    node T_1758 = eq(UInt<3>(1), ctrl.ld_type)
    node load = mux(T_1758, T_1745, T_1757)
    inst csr of CSR
    host := csr.host
    csr.src := ew_alu
    node T_1759 = bits(ew_inst, 31, 20)
    csr.addr := T_1759
    csr.cmd := ctrl.csr_cmd
    node T_1760 = add-wrap(ew_pc, UInt<3>(4))
    node T_1761 = eq(UInt<2>(3), ctrl.wb_sel)
    node T_1762 = mux(T_1761, csr.data, ew_alu)
    node T_1763 = eq(UInt<2>(2), ctrl.wb_sel)
    node T_1764 = mux(T_1763, T_1760, T_1762)
    node T_1765 = eq(UInt<2>(1), ctrl.wb_sel)
    node regWrite = mux(T_1765, load, T_1764)
    regFile.wen := ctrl.wb_en
    regFile.waddr := ex_rd_addr
    regFile.wdata := regWrite
  module Control : 
    output ctrl : {flip inst : UInt<32>, flip stall : UInt<1>, inst_re : UInt<1>, inst_type : UInt<1>, pc_sel : UInt<1>, A_sel : UInt<1>, B_sel : UInt<1>, imm_sel : UInt<3>, alu_op : UInt<4>, br_type : UInt<3>, data_re : UInt<1>, st_type : UInt<2>, ld_type : UInt<3>, wb_sel : UInt<2>, wb_en : UInt<1>, csr_cmd : UInt<2>}
    
    node T_1766 = bit-and(UInt<7>(127), ctrl.inst)
    node T_1767 = eq(T_1766, UInt<6>(55))
    node T_1768 = bit-and(UInt<7>(127), ctrl.inst)
    node T_1769 = eq(T_1768, UInt<5>(23))
    node T_1770 = bit-and(UInt<7>(127), ctrl.inst)
    node T_1771 = eq(T_1770, UInt<7>(111))
    node T_1772 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1773 = eq(T_1772, UInt<7>(103))
    node T_1774 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1775 = eq(T_1774, UInt<7>(99))
    node T_1776 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1777 = eq(T_1776, UInt<13>(4195))
    node T_1778 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1779 = eq(T_1778, UInt<15>(16483))
    node T_1780 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1781 = eq(T_1780, UInt<15>(20579))
    node T_1782 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1783 = eq(T_1782, UInt<15>(24675))
    node T_1784 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1785 = eq(T_1784, UInt<15>(28771))
    node T_1786 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1787 = eq(T_1786, UInt<2>(3))
    node T_1788 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1789 = eq(T_1788, UInt<13>(4099))
    node T_1790 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1791 = eq(T_1790, UInt<14>(8195))
    node T_1792 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1793 = eq(T_1792, UInt<15>(16387))
    node T_1794 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1795 = eq(T_1794, UInt<15>(20483))
    node T_1796 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1797 = eq(T_1796, UInt<6>(35))
    node T_1798 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1799 = eq(T_1798, UInt<13>(4131))
    node T_1800 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1801 = eq(T_1800, UInt<14>(8227))
    node T_1802 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1803 = eq(T_1802, UInt<5>(19))
    node T_1804 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1805 = eq(T_1804, UInt<14>(8211))
    node T_1806 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1807 = eq(T_1806, UInt<14>(12307))
    node T_1808 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1809 = eq(T_1808, UInt<15>(16403))
    node T_1810 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1811 = eq(T_1810, UInt<15>(24595))
    node T_1812 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1813 = eq(T_1812, UInt<15>(28691))
    node T_1814 = bit-and(UInt<32>(4261441663), ctrl.inst)
    node T_1815 = eq(T_1814, UInt<13>(4115))
    node T_1816 = bit-and(UInt<32>(4261441663), ctrl.inst)
    node T_1817 = eq(T_1816, UInt<15>(20499))
    node T_1818 = bit-and(UInt<32>(4261441663), ctrl.inst)
    node T_1819 = eq(T_1818, UInt<31>(1073762323))
    node T_1820 = bit-and(UInt<32>(4261441663), ctrl.inst)
    node T_1821 = eq(T_1820, UInt<6>(51))
    node T_1822 = bit-and(UInt<32>(4261441663), ctrl.inst)
    node T_1823 = eq(T_1822, UInt<31>(1073741875))
    node T_1824 = bit-and(UInt<32>(4261441663), ctrl.inst)
    node T_1825 = eq(T_1824, UInt<13>(4147))
    node T_1826 = bit-and(UInt<32>(4261441663), ctrl.inst)
    node T_1827 = eq(T_1826, UInt<14>(8243))
    node T_1828 = bit-and(UInt<32>(4261441663), ctrl.inst)
    node T_1829 = eq(T_1828, UInt<14>(12339))
    node T_1830 = bit-and(UInt<32>(4261441663), ctrl.inst)
    node T_1831 = eq(T_1830, UInt<15>(16435))
    node T_1832 = bit-and(UInt<32>(4261441663), ctrl.inst)
    node T_1833 = eq(T_1832, UInt<15>(20531))
    node T_1834 = bit-and(UInt<32>(4261441663), ctrl.inst)
    node T_1835 = eq(T_1834, UInt<31>(1073762355))
    node T_1836 = bit-and(UInt<32>(4261441663), ctrl.inst)
    node T_1837 = eq(T_1836, UInt<15>(24627))
    node T_1838 = bit-and(UInt<32>(4261441663), ctrl.inst)
    node T_1839 = eq(T_1838, UInt<15>(28723))
    node T_1840 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1841 = eq(T_1840, UInt<13>(4211))
    node T_1842 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1843 = eq(T_1842, UInt<14>(8307))
    node T_1844 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1845 = eq(T_1844, UInt<14>(12403))
    node T_1846 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1847 = eq(T_1846, UInt<15>(20595))
    node T_1848 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1849 = eq(T_1848, UInt<15>(24691))
    node T_1850 = bit-and(UInt<15>(28799), ctrl.inst)
    node T_1851 = eq(T_1850, UInt<15>(28787))
    node T_1852 = mux(T_1851, UInt<1>(0), UInt<1>(0))
    node T_1853 = mux(T_1849, UInt<1>(0), T_1852)
    node T_1854 = mux(T_1847, UInt<1>(0), T_1853)
    node T_1855 = mux(T_1845, UInt<1>(0), T_1854)
    node T_1856 = mux(T_1843, UInt<1>(0), T_1855)
    node T_1857 = mux(T_1841, UInt<1>(0), T_1856)
    node T_1858 = mux(T_1839, UInt<1>(0), T_1857)
    node T_1859 = mux(T_1837, UInt<1>(0), T_1858)
    node T_1860 = mux(T_1835, UInt<1>(0), T_1859)
    node T_1861 = mux(T_1833, UInt<1>(0), T_1860)
    node T_1862 = mux(T_1831, UInt<1>(0), T_1861)
    node T_1863 = mux(T_1829, UInt<1>(0), T_1862)
    node T_1864 = mux(T_1827, UInt<1>(0), T_1863)
    node T_1865 = mux(T_1825, UInt<1>(0), T_1864)
    node T_1866 = mux(T_1823, UInt<1>(0), T_1865)
    node T_1867 = mux(T_1821, UInt<1>(0), T_1866)
    node T_1868 = mux(T_1819, UInt<1>(0), T_1867)
    node T_1869 = mux(T_1817, UInt<1>(0), T_1868)
    node T_1870 = mux(T_1815, UInt<1>(0), T_1869)
    node T_1871 = mux(T_1813, UInt<1>(0), T_1870)
    node T_1872 = mux(T_1811, UInt<1>(0), T_1871)
    node T_1873 = mux(T_1809, UInt<1>(0), T_1872)
    node T_1874 = mux(T_1807, UInt<1>(0), T_1873)
    node T_1875 = mux(T_1805, UInt<1>(0), T_1874)
    node T_1876 = mux(T_1803, UInt<1>(0), T_1875)
    node T_1877 = mux(T_1801, UInt<1>(0), T_1876)
    node T_1878 = mux(T_1799, UInt<1>(0), T_1877)
    node T_1879 = mux(T_1797, UInt<1>(0), T_1878)
    node T_1880 = mux(T_1795, UInt<1>(0), T_1879)
    node T_1881 = mux(T_1793, UInt<1>(0), T_1880)
    node T_1882 = mux(T_1791, UInt<1>(0), T_1881)
    node T_1883 = mux(T_1789, UInt<1>(0), T_1882)
    node T_1884 = mux(T_1787, UInt<1>(0), T_1883)
    node T_1885 = mux(T_1785, UInt<1>(0), T_1884)
    node T_1886 = mux(T_1783, UInt<1>(0), T_1885)
    node T_1887 = mux(T_1781, UInt<1>(0), T_1886)
    node T_1888 = mux(T_1779, UInt<1>(0), T_1887)
    node T_1889 = mux(T_1777, UInt<1>(0), T_1888)
    node T_1890 = mux(T_1775, UInt<1>(0), T_1889)
    node T_1891 = mux(T_1773, UInt<1>(1), T_1890)
    node T_1892 = mux(T_1771, UInt<1>(1), T_1891)
    node T_1893 = mux(T_1769, UInt<1>(0), T_1892)
    node T_1894 = mux(T_1767, UInt<1>(0), T_1893)
    node T_1895 = mux(T_1851, UInt<1>(1), UInt<1>(1))
    node T_1896 = mux(T_1849, UInt<1>(1), T_1895)
    node T_1897 = mux(T_1847, UInt<1>(1), T_1896)
    node T_1898 = mux(T_1845, UInt<1>(0), T_1897)
    node T_1899 = mux(T_1843, UInt<1>(0), T_1898)
    node T_1900 = mux(T_1841, UInt<1>(0), T_1899)
    node T_1901 = mux(T_1839, UInt<1>(0), T_1900)
    node T_1902 = mux(T_1837, UInt<1>(0), T_1901)
    node T_1903 = mux(T_1835, UInt<1>(0), T_1902)
    node T_1904 = mux(T_1833, UInt<1>(0), T_1903)
    node T_1905 = mux(T_1831, UInt<1>(0), T_1904)
    node T_1906 = mux(T_1829, UInt<1>(0), T_1905)
    node T_1907 = mux(T_1827, UInt<1>(0), T_1906)
    node T_1908 = mux(T_1825, UInt<1>(0), T_1907)
    node T_1909 = mux(T_1823, UInt<1>(0), T_1908)
    node T_1910 = mux(T_1821, UInt<1>(0), T_1909)
    node T_1911 = mux(T_1819, UInt<1>(0), T_1910)
    node T_1912 = mux(T_1817, UInt<1>(0), T_1911)
    node T_1913 = mux(T_1815, UInt<1>(0), T_1912)
    node T_1914 = mux(T_1813, UInt<1>(0), T_1913)
    node T_1915 = mux(T_1811, UInt<1>(0), T_1914)
    node T_1916 = mux(T_1809, UInt<1>(0), T_1915)
    node T_1917 = mux(T_1807, UInt<1>(0), T_1916)
    node T_1918 = mux(T_1805, UInt<1>(0), T_1917)
    node T_1919 = mux(T_1803, UInt<1>(0), T_1918)
    node T_1920 = mux(T_1801, UInt<1>(0), T_1919)
    node T_1921 = mux(T_1799, UInt<1>(0), T_1920)
    node T_1922 = mux(T_1797, UInt<1>(0), T_1921)
    node T_1923 = mux(T_1795, UInt<1>(0), T_1922)
    node T_1924 = mux(T_1793, UInt<1>(0), T_1923)
    node T_1925 = mux(T_1791, UInt<1>(0), T_1924)
    node T_1926 = mux(T_1789, UInt<1>(0), T_1925)
    node T_1927 = mux(T_1787, UInt<1>(0), T_1926)
    node T_1928 = mux(T_1785, UInt<1>(1), T_1927)
    node T_1929 = mux(T_1783, UInt<1>(1), T_1928)
    node T_1930 = mux(T_1781, UInt<1>(1), T_1929)
    node T_1931 = mux(T_1779, UInt<1>(1), T_1930)
    node T_1932 = mux(T_1777, UInt<1>(1), T_1931)
    node T_1933 = mux(T_1775, UInt<1>(1), T_1932)
    node T_1934 = mux(T_1773, UInt<1>(0), T_1933)
    node T_1935 = mux(T_1771, UInt<1>(1), T_1934)
    node T_1936 = mux(T_1769, UInt<1>(1), T_1935)
    node T_1937 = mux(T_1767, UInt<1>(1), T_1936)
    node T_1938 = mux(T_1851, UInt<1>(1), UInt<1>(0))
    node T_1939 = mux(T_1849, UInt<1>(1), T_1938)
    node T_1940 = mux(T_1847, UInt<1>(1), T_1939)
    node T_1941 = mux(T_1845, UInt<1>(0), T_1940)
    node T_1942 = mux(T_1843, UInt<1>(0), T_1941)
    node T_1943 = mux(T_1841, UInt<1>(0), T_1942)
    node T_1944 = mux(T_1839, UInt<1>(0), T_1943)
    node T_1945 = mux(T_1837, UInt<1>(0), T_1944)
    node T_1946 = mux(T_1835, UInt<1>(0), T_1945)
    node T_1947 = mux(T_1833, UInt<1>(0), T_1946)
    node T_1948 = mux(T_1831, UInt<1>(0), T_1947)
    node T_1949 = mux(T_1829, UInt<1>(0), T_1948)
    node T_1950 = mux(T_1827, UInt<1>(0), T_1949)
    node T_1951 = mux(T_1825, UInt<1>(0), T_1950)
    node T_1952 = mux(T_1823, UInt<1>(0), T_1951)
    node T_1953 = mux(T_1821, UInt<1>(0), T_1952)
    node T_1954 = mux(T_1819, UInt<1>(1), T_1953)
    node T_1955 = mux(T_1817, UInt<1>(1), T_1954)
    node T_1956 = mux(T_1815, UInt<1>(1), T_1955)
    node T_1957 = mux(T_1813, UInt<1>(1), T_1956)
    node T_1958 = mux(T_1811, UInt<1>(1), T_1957)
    node T_1959 = mux(T_1809, UInt<1>(1), T_1958)
    node T_1960 = mux(T_1807, UInt<1>(1), T_1959)
    node T_1961 = mux(T_1805, UInt<1>(1), T_1960)
    node T_1962 = mux(T_1803, UInt<1>(1), T_1961)
    node T_1963 = mux(T_1801, UInt<1>(1), T_1962)
    node T_1964 = mux(T_1799, UInt<1>(1), T_1963)
    node T_1965 = mux(T_1797, UInt<1>(1), T_1964)
    node T_1966 = mux(T_1795, UInt<1>(1), T_1965)
    node T_1967 = mux(T_1793, UInt<1>(1), T_1966)
    node T_1968 = mux(T_1791, UInt<1>(1), T_1967)
    node T_1969 = mux(T_1789, UInt<1>(1), T_1968)
    node T_1970 = mux(T_1787, UInt<1>(1), T_1969)
    node T_1971 = mux(T_1785, UInt<1>(1), T_1970)
    node T_1972 = mux(T_1783, UInt<1>(1), T_1971)
    node T_1973 = mux(T_1781, UInt<1>(1), T_1972)
    node T_1974 = mux(T_1779, UInt<1>(1), T_1973)
    node T_1975 = mux(T_1777, UInt<1>(1), T_1974)
    node T_1976 = mux(T_1775, UInt<1>(1), T_1975)
    node T_1977 = mux(T_1773, UInt<1>(1), T_1976)
    node T_1978 = mux(T_1771, UInt<1>(1), T_1977)
    node T_1979 = mux(T_1769, UInt<1>(1), T_1978)
    node T_1980 = mux(T_1767, UInt<1>(1), T_1979)
    node T_1981 = mux(T_1851, UInt<3>(5), UInt<3>(7))
    node T_1982 = mux(T_1849, UInt<3>(5), T_1981)
    node T_1983 = mux(T_1847, UInt<3>(5), T_1982)
    node T_1984 = mux(T_1845, UInt<3>(5), T_1983)
    node T_1985 = mux(T_1843, UInt<3>(5), T_1984)
    node T_1986 = mux(T_1841, UInt<3>(5), T_1985)
    node T_1987 = mux(T_1839, UInt<3>(7), T_1986)
    node T_1988 = mux(T_1837, UInt<3>(7), T_1987)
    node T_1989 = mux(T_1835, UInt<3>(7), T_1988)
    node T_1990 = mux(T_1833, UInt<3>(7), T_1989)
    node T_1991 = mux(T_1831, UInt<3>(7), T_1990)
    node T_1992 = mux(T_1829, UInt<3>(7), T_1991)
    node T_1993 = mux(T_1827, UInt<3>(7), T_1992)
    node T_1994 = mux(T_1825, UInt<3>(7), T_1993)
    node T_1995 = mux(T_1823, UInt<3>(7), T_1994)
    node T_1996 = mux(T_1821, UInt<3>(7), T_1995)
    node T_1997 = mux(T_1819, UInt<3>(0), T_1996)
    node T_1998 = mux(T_1817, UInt<3>(0), T_1997)
    node T_1999 = mux(T_1815, UInt<3>(0), T_1998)
    node T_2000 = mux(T_1813, UInt<3>(0), T_1999)
    node T_2001 = mux(T_1811, UInt<3>(0), T_2000)
    node T_2002 = mux(T_1809, UInt<3>(0), T_2001)
    node T_2003 = mux(T_1807, UInt<3>(0), T_2002)
    node T_2004 = mux(T_1805, UInt<3>(0), T_2003)
    node T_2005 = mux(T_1803, UInt<3>(0), T_2004)
    node T_2006 = mux(T_1801, UInt<3>(1), T_2005)
    node T_2007 = mux(T_1799, UInt<3>(1), T_2006)
    node T_2008 = mux(T_1797, UInt<3>(1), T_2007)
    node T_2009 = mux(T_1795, UInt<3>(0), T_2008)
    node T_2010 = mux(T_1793, UInt<3>(0), T_2009)
    node T_2011 = mux(T_1791, UInt<3>(0), T_2010)
    node T_2012 = mux(T_1789, UInt<3>(0), T_2011)
    node T_2013 = mux(T_1787, UInt<3>(0), T_2012)
    node T_2014 = mux(T_1785, UInt<3>(4), T_2013)
    node T_2015 = mux(T_1783, UInt<3>(4), T_2014)
    node T_2016 = mux(T_1781, UInt<3>(4), T_2015)
    node T_2017 = mux(T_1779, UInt<3>(4), T_2016)
    node T_2018 = mux(T_1777, UInt<3>(4), T_2017)
    node T_2019 = mux(T_1775, UInt<3>(4), T_2018)
    node T_2020 = mux(T_1773, UInt<3>(0), T_2019)
    node T_2021 = mux(T_1771, UInt<3>(3), T_2020)
    node T_2022 = mux(T_1769, UInt<3>(2), T_2021)
    node T_2023 = mux(T_1767, UInt<3>(2), T_2022)
    node T_2024 = mux(T_1851, UInt<4>(11), UInt<4>(15))
    node T_2025 = mux(T_1849, UInt<4>(11), T_2024)
    node T_2026 = mux(T_1847, UInt<4>(11), T_2025)
    node T_2027 = mux(T_1845, UInt<4>(10), T_2026)
    node T_2028 = mux(T_1843, UInt<4>(10), T_2027)
    node T_2029 = mux(T_1841, UInt<4>(10), T_2028)
    node T_2030 = mux(T_1839, UInt<4>(2), T_2029)
    node T_2031 = mux(T_1837, UInt<4>(3), T_2030)
    node T_2032 = mux(T_1835, UInt<4>(9), T_2031)
    node T_2033 = mux(T_1833, UInt<4>(8), T_2032)
    node T_2034 = mux(T_1831, UInt<4>(4), T_2033)
    node T_2035 = mux(T_1829, UInt<4>(7), T_2034)
    node T_2036 = mux(T_1827, UInt<4>(5), T_2035)
    node T_2037 = mux(T_1825, UInt<4>(6), T_2036)
    node T_2038 = mux(T_1823, UInt<4>(1), T_2037)
    node T_2039 = mux(T_1821, UInt<4>(0), T_2038)
    node T_2040 = mux(T_1819, UInt<4>(9), T_2039)
    node T_2041 = mux(T_1817, UInt<4>(8), T_2040)
    node T_2042 = mux(T_1815, UInt<4>(6), T_2041)
    node T_2043 = mux(T_1813, UInt<4>(2), T_2042)
    node T_2044 = mux(T_1811, UInt<4>(3), T_2043)
    node T_2045 = mux(T_1809, UInt<4>(4), T_2044)
    node T_2046 = mux(T_1807, UInt<4>(7), T_2045)
    node T_2047 = mux(T_1805, UInt<4>(5), T_2046)
    node T_2048 = mux(T_1803, UInt<4>(0), T_2047)
    node T_2049 = mux(T_1801, UInt<4>(0), T_2048)
    node T_2050 = mux(T_1799, UInt<4>(0), T_2049)
    node T_2051 = mux(T_1797, UInt<4>(0), T_2050)
    node T_2052 = mux(T_1795, UInt<4>(0), T_2051)
    node T_2053 = mux(T_1793, UInt<4>(0), T_2052)
    node T_2054 = mux(T_1791, UInt<4>(0), T_2053)
    node T_2055 = mux(T_1789, UInt<4>(0), T_2054)
    node T_2056 = mux(T_1787, UInt<4>(0), T_2055)
    node T_2057 = mux(T_1785, UInt<4>(0), T_2056)
    node T_2058 = mux(T_1783, UInt<4>(0), T_2057)
    node T_2059 = mux(T_1781, UInt<4>(0), T_2058)
    node T_2060 = mux(T_1779, UInt<4>(0), T_2059)
    node T_2061 = mux(T_1777, UInt<4>(0), T_2060)
    node T_2062 = mux(T_1775, UInt<4>(0), T_2061)
    node T_2063 = mux(T_1773, UInt<4>(0), T_2062)
    node T_2064 = mux(T_1771, UInt<4>(0), T_2063)
    node T_2065 = mux(T_1769, UInt<4>(0), T_2064)
    node T_2066 = mux(T_1767, UInt<4>(11), T_2065)
    node T_2067 = mux(T_1851, UInt<3>(7), UInt<3>(7))
    node T_2068 = mux(T_1849, UInt<3>(7), T_2067)
    node T_2069 = mux(T_1847, UInt<3>(7), T_2068)
    node T_2070 = mux(T_1845, UInt<3>(7), T_2069)
    node T_2071 = mux(T_1843, UInt<3>(7), T_2070)
    node T_2072 = mux(T_1841, UInt<3>(7), T_2071)
    node T_2073 = mux(T_1839, UInt<3>(7), T_2072)
    node T_2074 = mux(T_1837, UInt<3>(7), T_2073)
    node T_2075 = mux(T_1835, UInt<3>(7), T_2074)
    node T_2076 = mux(T_1833, UInt<3>(7), T_2075)
    node T_2077 = mux(T_1831, UInt<3>(7), T_2076)
    node T_2078 = mux(T_1829, UInt<3>(7), T_2077)
    node T_2079 = mux(T_1827, UInt<3>(7), T_2078)
    node T_2080 = mux(T_1825, UInt<3>(7), T_2079)
    node T_2081 = mux(T_1823, UInt<3>(7), T_2080)
    node T_2082 = mux(T_1821, UInt<3>(7), T_2081)
    node T_2083 = mux(T_1819, UInt<3>(7), T_2082)
    node T_2084 = mux(T_1817, UInt<3>(7), T_2083)
    node T_2085 = mux(T_1815, UInt<3>(7), T_2084)
    node T_2086 = mux(T_1813, UInt<3>(7), T_2085)
    node T_2087 = mux(T_1811, UInt<3>(7), T_2086)
    node T_2088 = mux(T_1809, UInt<3>(7), T_2087)
    node T_2089 = mux(T_1807, UInt<3>(7), T_2088)
    node T_2090 = mux(T_1805, UInt<3>(7), T_2089)
    node T_2091 = mux(T_1803, UInt<3>(7), T_2090)
    node T_2092 = mux(T_1801, UInt<3>(7), T_2091)
    node T_2093 = mux(T_1799, UInt<3>(7), T_2092)
    node T_2094 = mux(T_1797, UInt<3>(7), T_2093)
    node T_2095 = mux(T_1795, UInt<3>(7), T_2094)
    node T_2096 = mux(T_1793, UInt<3>(7), T_2095)
    node T_2097 = mux(T_1791, UInt<3>(7), T_2096)
    node T_2098 = mux(T_1789, UInt<3>(7), T_2097)
    node T_2099 = mux(T_1787, UInt<3>(7), T_2098)
    node T_2100 = mux(T_1785, UInt<3>(4), T_2099)
    node T_2101 = mux(T_1783, UInt<3>(0), T_2100)
    node T_2102 = mux(T_1781, UInt<3>(5), T_2101)
    node T_2103 = mux(T_1779, UInt<3>(1), T_2102)
    node T_2104 = mux(T_1777, UInt<3>(6), T_2103)
    node T_2105 = mux(T_1775, UInt<3>(2), T_2104)
    node T_2106 = mux(T_1773, UInt<3>(7), T_2105)
    node T_2107 = mux(T_1771, UInt<3>(7), T_2106)
    node T_2108 = mux(T_1769, UInt<3>(7), T_2107)
    node T_2109 = mux(T_1767, UInt<3>(7), T_2108)
    node T_2110 = mux(T_1851, UInt<1>(0), UInt<1>(0))
    node T_2111 = mux(T_1849, UInt<1>(0), T_2110)
    node T_2112 = mux(T_1847, UInt<1>(0), T_2111)
    node T_2113 = mux(T_1845, UInt<1>(0), T_2112)
    node T_2114 = mux(T_1843, UInt<1>(0), T_2113)
    node T_2115 = mux(T_1841, UInt<1>(0), T_2114)
    node T_2116 = mux(T_1839, UInt<1>(0), T_2115)
    node T_2117 = mux(T_1837, UInt<1>(0), T_2116)
    node T_2118 = mux(T_1835, UInt<1>(0), T_2117)
    node T_2119 = mux(T_1833, UInt<1>(0), T_2118)
    node T_2120 = mux(T_1831, UInt<1>(0), T_2119)
    node T_2121 = mux(T_1829, UInt<1>(0), T_2120)
    node T_2122 = mux(T_1827, UInt<1>(0), T_2121)
    node T_2123 = mux(T_1825, UInt<1>(0), T_2122)
    node T_2124 = mux(T_1823, UInt<1>(0), T_2123)
    node T_2125 = mux(T_1821, UInt<1>(0), T_2124)
    node T_2126 = mux(T_1819, UInt<1>(0), T_2125)
    node T_2127 = mux(T_1817, UInt<1>(0), T_2126)
    node T_2128 = mux(T_1815, UInt<1>(0), T_2127)
    node T_2129 = mux(T_1813, UInt<1>(0), T_2128)
    node T_2130 = mux(T_1811, UInt<1>(0), T_2129)
    node T_2131 = mux(T_1809, UInt<1>(0), T_2130)
    node T_2132 = mux(T_1807, UInt<1>(0), T_2131)
    node T_2133 = mux(T_1805, UInt<1>(0), T_2132)
    node T_2134 = mux(T_1803, UInt<1>(0), T_2133)
    node T_2135 = mux(T_1801, UInt<1>(0), T_2134)
    node T_2136 = mux(T_1799, UInt<1>(0), T_2135)
    node T_2137 = mux(T_1797, UInt<1>(0), T_2136)
    node T_2138 = mux(T_1795, UInt<1>(0), T_2137)
    node T_2139 = mux(T_1793, UInt<1>(0), T_2138)
    node T_2140 = mux(T_1791, UInt<1>(0), T_2139)
    node T_2141 = mux(T_1789, UInt<1>(0), T_2140)
    node T_2142 = mux(T_1787, UInt<1>(0), T_2141)
    node T_2143 = mux(T_1785, UInt<1>(0), T_2142)
    node T_2144 = mux(T_1783, UInt<1>(0), T_2143)
    node T_2145 = mux(T_1781, UInt<1>(0), T_2144)
    node T_2146 = mux(T_1779, UInt<1>(0), T_2145)
    node T_2147 = mux(T_1777, UInt<1>(0), T_2146)
    node T_2148 = mux(T_1775, UInt<1>(0), T_2147)
    node T_2149 = mux(T_1773, UInt<1>(1), T_2148)
    node T_2150 = mux(T_1771, UInt<1>(1), T_2149)
    node T_2151 = mux(T_1769, UInt<1>(0), T_2150)
    node T_2152 = mux(T_1767, UInt<1>(0), T_2151)
    node T_2153 = mux(T_1851, UInt<2>(3), UInt<2>(3))
    node T_2154 = mux(T_1849, UInt<2>(3), T_2153)
    node T_2155 = mux(T_1847, UInt<2>(3), T_2154)
    node T_2156 = mux(T_1845, UInt<2>(3), T_2155)
    node T_2157 = mux(T_1843, UInt<2>(3), T_2156)
    node T_2158 = mux(T_1841, UInt<2>(3), T_2157)
    node T_2159 = mux(T_1839, UInt<2>(3), T_2158)
    node T_2160 = mux(T_1837, UInt<2>(3), T_2159)
    node T_2161 = mux(T_1835, UInt<2>(3), T_2160)
    node T_2162 = mux(T_1833, UInt<2>(3), T_2161)
    node T_2163 = mux(T_1831, UInt<2>(3), T_2162)
    node T_2164 = mux(T_1829, UInt<2>(3), T_2163)
    node T_2165 = mux(T_1827, UInt<2>(3), T_2164)
    node T_2166 = mux(T_1825, UInt<2>(3), T_2165)
    node T_2167 = mux(T_1823, UInt<2>(3), T_2166)
    node T_2168 = mux(T_1821, UInt<2>(3), T_2167)
    node T_2169 = mux(T_1819, UInt<2>(3), T_2168)
    node T_2170 = mux(T_1817, UInt<2>(3), T_2169)
    node T_2171 = mux(T_1815, UInt<2>(3), T_2170)
    node T_2172 = mux(T_1813, UInt<2>(3), T_2171)
    node T_2173 = mux(T_1811, UInt<2>(3), T_2172)
    node T_2174 = mux(T_1809, UInt<2>(3), T_2173)
    node T_2175 = mux(T_1807, UInt<2>(3), T_2174)
    node T_2176 = mux(T_1805, UInt<2>(3), T_2175)
    node T_2177 = mux(T_1803, UInt<2>(3), T_2176)
    node T_2178 = mux(T_1801, UInt<2>(0), T_2177)
    node T_2179 = mux(T_1799, UInt<2>(1), T_2178)
    node T_2180 = mux(T_1797, UInt<2>(2), T_2179)
    node T_2181 = mux(T_1795, UInt<2>(3), T_2180)
    node T_2182 = mux(T_1793, UInt<2>(3), T_2181)
    node T_2183 = mux(T_1791, UInt<2>(3), T_2182)
    node T_2184 = mux(T_1789, UInt<2>(3), T_2183)
    node T_2185 = mux(T_1787, UInt<2>(3), T_2184)
    node T_2186 = mux(T_1785, UInt<2>(3), T_2185)
    node T_2187 = mux(T_1783, UInt<2>(3), T_2186)
    node T_2188 = mux(T_1781, UInt<2>(3), T_2187)
    node T_2189 = mux(T_1779, UInt<2>(3), T_2188)
    node T_2190 = mux(T_1777, UInt<2>(3), T_2189)
    node T_2191 = mux(T_1775, UInt<2>(3), T_2190)
    node T_2192 = mux(T_1773, UInt<2>(3), T_2191)
    node T_2193 = mux(T_1771, UInt<2>(3), T_2192)
    node T_2194 = mux(T_1769, UInt<2>(3), T_2193)
    node T_2195 = mux(T_1767, UInt<2>(3), T_2194)
    node T_2196 = mux(T_1851, UInt<3>(7), UInt<3>(7))
    node T_2197 = mux(T_1849, UInt<3>(7), T_2196)
    node T_2198 = mux(T_1847, UInt<3>(7), T_2197)
    node T_2199 = mux(T_1845, UInt<3>(7), T_2198)
    node T_2200 = mux(T_1843, UInt<3>(7), T_2199)
    node T_2201 = mux(T_1841, UInt<3>(7), T_2200)
    node T_2202 = mux(T_1839, UInt<3>(7), T_2201)
    node T_2203 = mux(T_1837, UInt<3>(7), T_2202)
    node T_2204 = mux(T_1835, UInt<3>(7), T_2203)
    node T_2205 = mux(T_1833, UInt<3>(7), T_2204)
    node T_2206 = mux(T_1831, UInt<3>(7), T_2205)
    node T_2207 = mux(T_1829, UInt<3>(7), T_2206)
    node T_2208 = mux(T_1827, UInt<3>(7), T_2207)
    node T_2209 = mux(T_1825, UInt<3>(7), T_2208)
    node T_2210 = mux(T_1823, UInt<3>(7), T_2209)
    node T_2211 = mux(T_1821, UInt<3>(7), T_2210)
    node T_2212 = mux(T_1819, UInt<3>(7), T_2211)
    node T_2213 = mux(T_1817, UInt<3>(7), T_2212)
    node T_2214 = mux(T_1815, UInt<3>(7), T_2213)
    node T_2215 = mux(T_1813, UInt<3>(7), T_2214)
    node T_2216 = mux(T_1811, UInt<3>(7), T_2215)
    node T_2217 = mux(T_1809, UInt<3>(7), T_2216)
    node T_2218 = mux(T_1807, UInt<3>(7), T_2217)
    node T_2219 = mux(T_1805, UInt<3>(7), T_2218)
    node T_2220 = mux(T_1803, UInt<3>(7), T_2219)
    node T_2221 = mux(T_1801, UInt<3>(7), T_2220)
    node T_2222 = mux(T_1799, UInt<3>(7), T_2221)
    node T_2223 = mux(T_1797, UInt<3>(7), T_2222)
    node T_2224 = mux(T_1795, UInt<3>(3), T_2223)
    node T_2225 = mux(T_1793, UInt<3>(4), T_2224)
    node T_2226 = mux(T_1791, UInt<3>(0), T_2225)
    node T_2227 = mux(T_1789, UInt<3>(1), T_2226)
    node T_2228 = mux(T_1787, UInt<3>(2), T_2227)
    node T_2229 = mux(T_1785, UInt<3>(7), T_2228)
    node T_2230 = mux(T_1783, UInt<3>(7), T_2229)
    node T_2231 = mux(T_1781, UInt<3>(7), T_2230)
    node T_2232 = mux(T_1779, UInt<3>(7), T_2231)
    node T_2233 = mux(T_1777, UInt<3>(7), T_2232)
    node T_2234 = mux(T_1775, UInt<3>(7), T_2233)
    node T_2235 = mux(T_1773, UInt<3>(7), T_2234)
    node T_2236 = mux(T_1771, UInt<3>(7), T_2235)
    node T_2237 = mux(T_1769, UInt<3>(7), T_2236)
    node T_2238 = mux(T_1767, UInt<3>(7), T_2237)
    node T_2239 = mux(T_1851, UInt<2>(3), UInt<2>(0))
    node T_2240 = mux(T_1849, UInt<2>(3), T_2239)
    node T_2241 = mux(T_1847, UInt<2>(3), T_2240)
    node T_2242 = mux(T_1845, UInt<2>(3), T_2241)
    node T_2243 = mux(T_1843, UInt<2>(3), T_2242)
    node T_2244 = mux(T_1841, UInt<2>(3), T_2243)
    node T_2245 = mux(T_1839, UInt<2>(0), T_2244)
    node T_2246 = mux(T_1837, UInt<2>(0), T_2245)
    node T_2247 = mux(T_1835, UInt<2>(0), T_2246)
    node T_2248 = mux(T_1833, UInt<2>(0), T_2247)
    node T_2249 = mux(T_1831, UInt<2>(0), T_2248)
    node T_2250 = mux(T_1829, UInt<2>(0), T_2249)
    node T_2251 = mux(T_1827, UInt<2>(0), T_2250)
    node T_2252 = mux(T_1825, UInt<2>(0), T_2251)
    node T_2253 = mux(T_1823, UInt<2>(0), T_2252)
    node T_2254 = mux(T_1821, UInt<2>(0), T_2253)
    node T_2255 = mux(T_1819, UInt<2>(0), T_2254)
    node T_2256 = mux(T_1817, UInt<2>(0), T_2255)
    node T_2257 = mux(T_1815, UInt<2>(0), T_2256)
    node T_2258 = mux(T_1813, UInt<2>(0), T_2257)
    node T_2259 = mux(T_1811, UInt<2>(0), T_2258)
    node T_2260 = mux(T_1809, UInt<2>(0), T_2259)
    node T_2261 = mux(T_1807, UInt<2>(0), T_2260)
    node T_2262 = mux(T_1805, UInt<2>(0), T_2261)
    node T_2263 = mux(T_1803, UInt<2>(0), T_2262)
    node T_2264 = mux(T_1801, UInt<2>(0), T_2263)
    node T_2265 = mux(T_1799, UInt<2>(0), T_2264)
    node T_2266 = mux(T_1797, UInt<2>(0), T_2265)
    node T_2267 = mux(T_1795, UInt<2>(1), T_2266)
    node T_2268 = mux(T_1793, UInt<2>(1), T_2267)
    node T_2269 = mux(T_1791, UInt<2>(1), T_2268)
    node T_2270 = mux(T_1789, UInt<2>(1), T_2269)
    node T_2271 = mux(T_1787, UInt<2>(1), T_2270)
    node T_2272 = mux(T_1785, UInt<2>(0), T_2271)
    node T_2273 = mux(T_1783, UInt<2>(0), T_2272)
    node T_2274 = mux(T_1781, UInt<2>(0), T_2273)
    node T_2275 = mux(T_1779, UInt<2>(0), T_2274)
    node T_2276 = mux(T_1777, UInt<2>(0), T_2275)
    node T_2277 = mux(T_1775, UInt<2>(0), T_2276)
    node T_2278 = mux(T_1773, UInt<2>(2), T_2277)
    node T_2279 = mux(T_1771, UInt<2>(2), T_2278)
    node T_2280 = mux(T_1769, UInt<2>(0), T_2279)
    node T_2281 = mux(T_1767, UInt<2>(0), T_2280)
    node T_2282 = mux(T_1851, UInt<1>(0), UInt<1>(0))
    node T_2283 = mux(T_1849, UInt<1>(0), T_2282)
    node T_2284 = mux(T_1847, UInt<1>(0), T_2283)
    node T_2285 = mux(T_1845, UInt<1>(0), T_2284)
    node T_2286 = mux(T_1843, UInt<1>(0), T_2285)
    node T_2287 = mux(T_1841, UInt<1>(0), T_2286)
    node T_2288 = mux(T_1839, UInt<1>(1), T_2287)
    node T_2289 = mux(T_1837, UInt<1>(1), T_2288)
    node T_2290 = mux(T_1835, UInt<1>(1), T_2289)
    node T_2291 = mux(T_1833, UInt<1>(1), T_2290)
    node T_2292 = mux(T_1831, UInt<1>(1), T_2291)
    node T_2293 = mux(T_1829, UInt<1>(1), T_2292)
    node T_2294 = mux(T_1827, UInt<1>(1), T_2293)
    node T_2295 = mux(T_1825, UInt<1>(1), T_2294)
    node T_2296 = mux(T_1823, UInt<1>(1), T_2295)
    node T_2297 = mux(T_1821, UInt<1>(1), T_2296)
    node T_2298 = mux(T_1819, UInt<1>(1), T_2297)
    node T_2299 = mux(T_1817, UInt<1>(1), T_2298)
    node T_2300 = mux(T_1815, UInt<1>(1), T_2299)
    node T_2301 = mux(T_1813, UInt<1>(1), T_2300)
    node T_2302 = mux(T_1811, UInt<1>(1), T_2301)
    node T_2303 = mux(T_1809, UInt<1>(1), T_2302)
    node T_2304 = mux(T_1807, UInt<1>(1), T_2303)
    node T_2305 = mux(T_1805, UInt<1>(1), T_2304)
    node T_2306 = mux(T_1803, UInt<1>(1), T_2305)
    node T_2307 = mux(T_1801, UInt<1>(0), T_2306)
    node T_2308 = mux(T_1799, UInt<1>(0), T_2307)
    node T_2309 = mux(T_1797, UInt<1>(0), T_2308)
    node T_2310 = mux(T_1795, UInt<1>(1), T_2309)
    node T_2311 = mux(T_1793, UInt<1>(1), T_2310)
    node T_2312 = mux(T_1791, UInt<1>(1), T_2311)
    node T_2313 = mux(T_1789, UInt<1>(1), T_2312)
    node T_2314 = mux(T_1787, UInt<1>(1), T_2313)
    node T_2315 = mux(T_1785, UInt<1>(0), T_2314)
    node T_2316 = mux(T_1783, UInt<1>(0), T_2315)
    node T_2317 = mux(T_1781, UInt<1>(0), T_2316)
    node T_2318 = mux(T_1779, UInt<1>(0), T_2317)
    node T_2319 = mux(T_1777, UInt<1>(0), T_2318)
    node T_2320 = mux(T_1775, UInt<1>(0), T_2319)
    node T_2321 = mux(T_1773, UInt<1>(1), T_2320)
    node T_2322 = mux(T_1771, UInt<1>(1), T_2321)
    node T_2323 = mux(T_1769, UInt<1>(1), T_2322)
    node T_2324 = mux(T_1767, UInt<1>(1), T_2323)
    node T_2325 = mux(T_1851, UInt<2>(3), UInt<2>(0))
    node T_2326 = mux(T_1849, UInt<2>(2), T_2325)
    node T_2327 = mux(T_1847, UInt<2>(1), T_2326)
    node T_2328 = mux(T_1845, UInt<2>(3), T_2327)
    node T_2329 = mux(T_1843, UInt<2>(2), T_2328)
    node T_2330 = mux(T_1841, UInt<2>(1), T_2329)
    node T_2331 = mux(T_1839, UInt<2>(0), T_2330)
    node T_2332 = mux(T_1837, UInt<2>(0), T_2331)
    node T_2333 = mux(T_1835, UInt<2>(0), T_2332)
    node T_2334 = mux(T_1833, UInt<2>(0), T_2333)
    node T_2335 = mux(T_1831, UInt<2>(0), T_2334)
    node T_2336 = mux(T_1829, UInt<2>(0), T_2335)
    node T_2337 = mux(T_1827, UInt<2>(0), T_2336)
    node T_2338 = mux(T_1825, UInt<2>(0), T_2337)
    node T_2339 = mux(T_1823, UInt<2>(0), T_2338)
    node T_2340 = mux(T_1821, UInt<2>(0), T_2339)
    node T_2341 = mux(T_1819, UInt<2>(0), T_2340)
    node T_2342 = mux(T_1817, UInt<2>(0), T_2341)
    node T_2343 = mux(T_1815, UInt<2>(0), T_2342)
    node T_2344 = mux(T_1813, UInt<2>(0), T_2343)
    node T_2345 = mux(T_1811, UInt<2>(0), T_2344)
    node T_2346 = mux(T_1809, UInt<2>(0), T_2345)
    node T_2347 = mux(T_1807, UInt<2>(0), T_2346)
    node T_2348 = mux(T_1805, UInt<2>(0), T_2347)
    node T_2349 = mux(T_1803, UInt<2>(0), T_2348)
    node T_2350 = mux(T_1801, UInt<2>(0), T_2349)
    node T_2351 = mux(T_1799, UInt<2>(0), T_2350)
    node T_2352 = mux(T_1797, UInt<2>(0), T_2351)
    node T_2353 = mux(T_1795, UInt<2>(0), T_2352)
    node T_2354 = mux(T_1793, UInt<2>(0), T_2353)
    node T_2355 = mux(T_1791, UInt<2>(0), T_2354)
    node T_2356 = mux(T_1789, UInt<2>(0), T_2355)
    node T_2357 = mux(T_1787, UInt<2>(0), T_2356)
    node T_2358 = mux(T_1785, UInt<2>(0), T_2357)
    node T_2359 = mux(T_1783, UInt<2>(0), T_2358)
    node T_2360 = mux(T_1781, UInt<2>(0), T_2359)
    node T_2361 = mux(T_1779, UInt<2>(0), T_2360)
    node T_2362 = mux(T_1777, UInt<2>(0), T_2361)
    node T_2363 = mux(T_1775, UInt<2>(0), T_2362)
    node T_2364 = mux(T_1773, UInt<2>(0), T_2363)
    node T_2365 = mux(T_1771, UInt<2>(0), T_2364)
    node T_2366 = mux(T_1769, UInt<2>(0), T_2365)
    node T_2367 = mux(T_1767, UInt<2>(0), T_2366)
    node rs1_addr = bits(ctrl.inst, 19, 15)
    node rs2_addr = bits(ctrl.inst, 24, 20)
    reg st_type : UInt<2>
    reg ld_type : UInt<3>
    reg wb_sel : UInt<2>
    node T_2368 = bit(T_2324, 0)
    reg wb_en : UInt<1>
    reg csr_cmd : UInt<2>
    ctrl.pc_sel := T_1894
    node T_2369 = bit-not(ctrl.stall)
    node T_2370 = bit-not(ctrl.data_re)
    node T_2371 = bit-and(T_2369, T_2370)
    ctrl.inst_re := T_2371
    node T_2372 = neq(T_2238, UInt<3>(7))
    node T_2373 = bit(T_2152, 0)
    node T_2374 = bit-or(T_2372, T_2373)
    node T_2375 = mux(T_2374, UInt<1>(1), UInt<1>(0))
    ctrl.inst_type := T_2375
    ctrl.A_sel := T_1937
    ctrl.B_sel := T_1980
    ctrl.imm_sel := T_2023
    ctrl.alu_op := T_2066
    ctrl.br_type := T_2109
    ctrl.st_type := T_2195
    node T_2376 = bit-not(ctrl.stall)
    when T_2376 : 
      st_type := ctrl.st_type
      ld_type := T_2238
      wb_sel := T_2281
      node T_2377 = bit(T_2324, 0)
      wb_en := T_2377
      csr_cmd := T_2367
    node T_2378 = neq(ctrl.ld_type, UInt<3>(7))
    node T_2379 = neq(T_2238, UInt<3>(7))
    node T_2380 = mux(ctrl.stall, T_2378, T_2379)
    ctrl.data_re := T_2380
    ctrl.ld_type := ld_type
    ctrl.wb_en := wb_en
    ctrl.wb_sel := wb_sel
    ctrl.csr_cmd := csr_cmd
  module Core : 
    output host : {status : UInt<32>, flip hid : UInt<1>, tohost : UInt<32>}
    input stall : UInt<1>
    output icache : {re : UInt<1>, flip dout : UInt<32>, we : UInt<4>, addr : UInt<32>, din : UInt<32>}
    output dcache : {re : UInt<1>, flip dout : UInt<32>, we : UInt<4>, addr : UInt<32>, din : UInt<32>}
    
    inst dpath of Datapath
    inst ctrl of Control
    host := dpath.host
    icache := dpath.icache
    dcache := dpath.dcache
    dpath.ctrl := ctrl.ctrl
    dpath.stall := stall
  module Queue : 
    output count : UInt<3>
    input enq : {valid : UInt<1>, flip ready : UInt<1>, bits : {mask : UInt<4>, tag : UInt<5>, rw : UInt<1>, addr : UInt<32>}}
    output deq : {valid : UInt<1>, flip ready : UInt<1>, bits : {mask : UInt<4>, tag : UInt<5>, rw : UInt<1>, addr : UInt<32>}}
    
    cmem ram : {mask : UInt<4>, tag : UInt<5>, rw : UInt<1>, addr : UInt<32>}[4]
    reg T_2381 : UInt<2>
    on-reset T_2381 := UInt<2>(0)
    reg T_2382 : UInt<2>
    on-reset T_2382 := UInt<2>(0)
    reg maybe_full : UInt<1>
    on-reset maybe_full := UInt<1>(0)
    node ptr_match = eq(T_2381, T_2382)
    node T_2383 = bit-not(maybe_full)
    node empty = bit-and(ptr_match, T_2383)
    node full = bit-and(ptr_match, maybe_full)
    node maybe_flow = bit-and(UInt<1>(0), empty)
    node do_flow = bit-and(maybe_flow, deq.ready)
    node T_2384 = bit-and(enq.ready, enq.valid)
    node T_2385 = bit-not(do_flow)
    node do_enq = bit-and(T_2384, T_2385)
    node T_2386 = bit-and(deq.ready, deq.valid)
    node T_2387 = bit-not(do_flow)
    node do_deq = bit-and(T_2386, T_2387)
    when do_enq : 
      infer accessor T_2388 = ram[T_2381]
      T_2388 := enq.bits
      node T_2389 = eq(T_2381, UInt<2>(3))
      node T_2390 = bit-and(UInt<1>(0), T_2389)
      node T_2391 = add-wrap(T_2381, UInt<1>(1))
      node T_2392 = mux(T_2390, UInt<1>(0), T_2391)
      T_2381 := T_2392
    when do_deq : 
      node T_2393 = eq(T_2382, UInt<2>(3))
      node T_2394 = bit-and(UInt<1>(0), T_2393)
      node T_2395 = add-wrap(T_2382, UInt<1>(1))
      node T_2396 = mux(T_2394, UInt<1>(0), T_2395)
      T_2382 := T_2396
    node T_2397 = neq(do_enq, do_deq)
    when T_2397 : maybe_full := do_enq
    node T_2398 = bit-not(empty)
    node T_2399 = bit-and(UInt<1>(0), enq.valid)
    node T_2400 = bit-or(T_2398, T_2399)
    deq.valid := T_2400
    node T_2401 = bit-not(full)
    node T_2402 = bit-and(UInt<1>(0), deq.ready)
    node T_2403 = bit-or(T_2401, T_2402)
    enq.ready := T_2403
    infer accessor T_2404 = ram[T_2382]
    wire T_2405 : {mask : UInt<4>, tag : UInt<5>, rw : UInt<1>, addr : UInt<32>}
    node T_2406 = mux(maybe_flow, enq.bits.mask, T_2404.mask)
    T_2405.mask := T_2406
    node T_2407 = mux(maybe_flow, enq.bits.tag, T_2404.tag)
    T_2405.tag := T_2407
    node T_2408 = mux(maybe_flow, enq.bits.rw, T_2404.rw)
    T_2405.rw := T_2408
    node T_2409 = mux(maybe_flow, enq.bits.addr, T_2404.addr)
    T_2405.addr := T_2409
    deq.bits := T_2405
    node ptr_diff = sub-wrap(T_2381, T_2382)
    node T_2410 = bit-and(maybe_full, ptr_match)
    node T_2411 = cat(T_2410, ptr_diff)
    count := T_2411
  module Queue_1490 : 
    output count : UInt<3>
    input enq : {valid : UInt<1>, flip ready : UInt<1>, bits : {data : UInt<32>}}
    output deq : {valid : UInt<1>, flip ready : UInt<1>, bits : {data : UInt<32>}}
    
    cmem ram : {data : UInt<32>}[4]
    reg T_2412 : UInt<2>
    on-reset T_2412 := UInt<2>(0)
    reg T_2413 : UInt<2>
    on-reset T_2413 := UInt<2>(0)
    reg maybe_full : UInt<1>
    on-reset maybe_full := UInt<1>(0)
    node ptr_match = eq(T_2412, T_2413)
    node T_2414 = bit-not(maybe_full)
    node empty = bit-and(ptr_match, T_2414)
    node full = bit-and(ptr_match, maybe_full)
    node maybe_flow = bit-and(UInt<1>(0), empty)
    node do_flow = bit-and(maybe_flow, deq.ready)
    node T_2415 = bit-and(enq.ready, enq.valid)
    node T_2416 = bit-not(do_flow)
    node do_enq = bit-and(T_2415, T_2416)
    node T_2417 = bit-and(deq.ready, deq.valid)
    node T_2418 = bit-not(do_flow)
    node do_deq = bit-and(T_2417, T_2418)
    when do_enq : 
      infer accessor T_2419 = ram[T_2412]
      T_2419 := enq.bits
      node T_2420 = eq(T_2412, UInt<2>(3))
      node T_2421 = bit-and(UInt<1>(0), T_2420)
      node T_2422 = add-wrap(T_2412, UInt<1>(1))
      node T_2423 = mux(T_2421, UInt<1>(0), T_2422)
      T_2412 := T_2423
    when do_deq : 
      node T_2424 = eq(T_2413, UInt<2>(3))
      node T_2425 = bit-and(UInt<1>(0), T_2424)
      node T_2426 = add-wrap(T_2413, UInt<1>(1))
      node T_2427 = mux(T_2425, UInt<1>(0), T_2426)
      T_2413 := T_2427
    node T_2428 = neq(do_enq, do_deq)
    when T_2428 : maybe_full := do_enq
    node T_2429 = bit-not(empty)
    node T_2430 = bit-and(UInt<1>(0), enq.valid)
    node T_2431 = bit-or(T_2429, T_2430)
    deq.valid := T_2431
    node T_2432 = bit-not(full)
    node T_2433 = bit-and(UInt<1>(0), deq.ready)
    node T_2434 = bit-or(T_2432, T_2433)
    enq.ready := T_2434
    infer accessor T_2435 = ram[T_2413]
    wire T_2436 : {data : UInt<32>}
    node T_2437 = mux(maybe_flow, enq.bits.data, T_2435.data)
    T_2436.data := T_2437
    deq.bits := T_2436
    node ptr_diff = sub-wrap(T_2412, T_2413)
    node T_2438 = bit-and(maybe_full, ptr_match)
    node T_2439 = cat(T_2438, ptr_diff)
    count := T_2439
  module Memory : 
    output memory : {req_cmd : {valid : UInt<1>, flip ready : UInt<1>, bits : {mask : UInt<4>, tag : UInt<5>, rw : UInt<1>, addr : UInt<32>}}, req_data : {valid : UInt<1>, flip ready : UInt<1>, bits : {data : UInt<32>}}, flip resp : {valid : UInt<1>, flip ready : UInt<1>, bits : {tag : UInt<5>, data : UInt<32>}}}
    output stall : UInt<1>
    input icache : {re : UInt<1>, flip dout : UInt<32>, we : UInt<4>, addr : UInt<32>, din : UInt<32>}
    input dcache : {re : UInt<1>, flip dout : UInt<32>, we : UInt<4>, addr : UInt<32>, din : UInt<32>}
    
    inst memReqCmdQueue of Queue
    inst memReqDataQueue of Queue_1490
    reg state : UInt<1>
    on-reset state := UInt<1>(0)
    reg tag : UInt<5>
    on-reset tag := UInt<5>(0)
    node T_2440 = eq(state, UInt<1>(0))
    node T_2441 = bit-or(icache.re, dcache.re)
    node T_2442 = eq(dcache.we, UInt<1>(0))
    node T_2443 = bit-not(T_2442)
    node T_2444 = bit-or(T_2441, T_2443)
    node cpuReq = bit-and(T_2440, T_2444)
    node T_2445 = bits(icache.addr, 31, 2)
    node iaddr = cat(T_2445, UInt<2>(0))
    node T_2446 = bits(dcache.addr, 31, 2)
    node daddr = cat(T_2446, UInt<2>(0))
    reg idata : UInt
    reg ddata : UInt
    reg ire : UInt<1>
    reg dre : UInt<1>
    icache.dout := idata
    dcache.dout := ddata
    memory.req_cmd := memReqCmdQueue.deq
    memory.req_data := memReqDataQueue.deq
    memory.resp.ready := UInt<1>(0)
    node T_2447 = eq(state, UInt<1>(1))
    node T_2448 = bit-not(memReqCmdQueue.enq.ready)
    node T_2449 = bit-or(T_2447, T_2448)
    node T_2450 = bit-not(memReqDataQueue.enq.ready)
    node T_2451 = bit-or(T_2449, T_2450)
    stall := T_2451
    node T_2452 = eq(dcache.we, UInt<1>(0))
    node T_2453 = bit-not(T_2452)
    memReqCmdQueue.enq.bits.rw := T_2453
    memReqCmdQueue.enq.bits.tag := tag
    node T_2454 = eq(dcache.we, UInt<1>(0))
    node T_2455 = bit-not(T_2454)
    node T_2456 = bit-not(icache.re)
    node T_2457 = bit-or(T_2455, T_2456)
    node T_2458 = mux(T_2457, daddr, iaddr)
    memReqCmdQueue.enq.bits.addr := T_2458
    memReqCmdQueue.enq.bits.mask := dcache.we
    node T_2459 = bit-and(memReqDataQueue.enq.ready, cpuReq)
    memReqCmdQueue.enq.valid := T_2459
    memReqDataQueue.enq.bits.data := dcache.din
    node T_2460 = bit-and(memReqCmdQueue.enq.ready, cpuReq)
    node T_2461 = eq(dcache.we, UInt<1>(0))
    node T_2462 = bit-not(T_2461)
    node T_2463 = bit-and(T_2460, T_2462)
    memReqDataQueue.enq.valid := T_2463
    node T_2464 = eq(UInt<1>(0), state)
    when T_2464 : 
      node T_2465 = bit-or(icache.re, dcache.re)
      node T_2466 = eq(dcache.we, UInt<1>(0))
      node T_2467 = bit-not(T_2466)
      node T_2468 = bit-not(T_2467)
      node T_2469 = bit-and(T_2465, T_2468)
      node T_2470 = bit-and(T_2469, memReqCmdQueue.enq.ready)
      when T_2470 : 
        ire := icache.re
        dre := dcache.re
        state := UInt<1>(1)
    node T_2471 = eq(UInt<1>(1), state)
    when T_2471 : 
      memory.resp.ready := UInt<1>(1)
      node T_2472 = eq(memory.resp.bits.tag, tag)
      node T_2473 = bit-and(memory.resp.valid, T_2472)
      when T_2473 : 
        state := UInt<1>(0)
        node T_2474 = add-wrap(tag, UInt<1>(1))
        tag := T_2474
        memory.resp.ready := UInt<1>(0)
        when ire : idata := memory.resp.bits.data
        when dre : ddata := memory.resp.bits.data
  module Tile : 
    output htif : {host : {status : UInt<32>, flip hid : UInt<1>, tohost : UInt<32>}}
    output memory : {req_cmd : {valid : UInt<1>, flip ready : UInt<1>, bits : {mask : UInt<4>, tag : UInt<5>, rw : UInt<1>, addr : UInt<32>}}, req_data : {valid : UInt<1>, flip ready : UInt<1>, bits : {data : UInt<32>}}, flip resp : {valid : UInt<1>, flip ready : UInt<1>, bits : {tag : UInt<5>, data : UInt<32>}}}
    
    inst core of Core
    inst memmod of Memory
    htif.host := core.host
    memory := memmod.memory
    core.stall := memmod.stall
    memmod.icache := core.icache
    memmod.dcache := core.dcache
