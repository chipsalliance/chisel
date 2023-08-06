; RUN: firtool %s --format=fir --ir-fir | FileCheck %s

; Temporary wires should not be introduced by type lowering, and if they are,
; they should be cleaned up by canonicalize.

; CHECK-LABEL: firrtl.module @Issue794
; CHECK-SAME: (in %clock: !firrtl.clock,
; CHECK:       %[[memory_0:.+]] = firrtl.reg %clock {{.*}}: !firrtl.clock, !firrtl.uint<8>
; CHECK:       firrtl.mux(%[[v14:.+]], %wData_0, %[[memory_0]])
; CHECK:       firrtl.mux(%[[v19:.+]], %wData_1, %[[v5:.+]])
; CHECK:       firrtl.strictconnect %[[memory_0]], %[[v22:.+]]
circuit Issue794: %[[{
    "class": "sifive.enterprise.firrtl.MarkDUTAnnotation",
    "target":"~Issue794|Issue794"
  }, 
  {
    "class": "sifive.enterprise.firrtl.ConvertMemToRegOfVecAnnotation$"
  }]]
  module Issue794:
    input clock: Clock
    input rAddr: UInt<2>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wAddr: UInt<2>[2]
    input wEn: UInt<1>[2]
    input wMask: UInt<1>[2]
    input wData: UInt<8>[2]

    mem memory:
      data-type => UInt<8>
      depth => 4
      reader => r
      writer => w0,w1
      read-latency => 0
      write-latency => 1
      read-under-write => undefined

    memory.r.clk <= clock
    memory.r.en <= rEn
    memory.r.addr <= rAddr
    rData <= memory.r.data

    memory.w0.clk <= clock
    memory.w0.en <= wEn[0]
    memory.w0.addr <= wAddr[0]
    memory.w0.mask <= wMask[0]
    memory.w0.data <= wData[0]

    memory.w1.clk <= clock
    memory.w1.en <= wEn[1]
    memory.w1.addr <= wAddr[1]
    memory.w1.mask <= wMask[1]
    memory.w1.data <= wData[1]
