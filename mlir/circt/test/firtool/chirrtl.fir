; RUN: firtool %s | FileCheck %s

; This is testing that CHIRRTL enable inference is working as intended.  If the
; no-op wires and nodes are not optimized away, then both ports should always
; be enabled.  If they are accidentally removed before the lower-chirrtl pass,
; then they won't be enabled.

circuit test: %[[{
    "class": "sifive.enterprise.firrtl.MarkDUTAnnotation",
    "target":"~test|test"
  }]]
  module test:
    input p: UInt<1>
    input addr1: UInt<4>
    input addr2: UInt<4>
    input addr3: UInt<4>
    input clock: Clock
    input data: UInt<8>
    output out0: UInt<8>
    output out1: UInt<8>

    ; CHECK: testmem_16x8 testmem_ext (
    smem testmem : UInt<8>[16], undefined

    ; CHECK: .R0_en (1'h1)
    node _T_0 = addr1
    when p:
      read mport testport0 = testmem[_T_0], clock
    out0 <= testport0

    ; CHECK: .R1_en (1'h1),
    wire _T_1: UInt<4>
    _T_1 <= addr2
    when p:
      read mport testport1 = testmem[_T_1], clock
    out1 <= testport1

    node writeAddr = addr3
    when p:
      write mport testport2 = testmem[writeAddr], clock
    testport2 <= data
