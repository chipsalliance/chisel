; RUN: firtool %s --format=fir --ir-sv | FileCheck %s --check-prefixes=CHECK,COMMON --implicit-check-not sv.attributes
; RUN: firtool %s --format=fir --ir-sv -disable-mem-randomization | FileCheck %s --check-prefix COMMON --implicit-check-not RANDOMIZE_MEM
; RUN: firtool %s --format=fir --ir-sv -disable-reg-randomization | FileCheck %s --check-prefix COMMON --implicit-check-not RANDOMIZE_REG
; RUN: firtool %s --format=fir --ir-sv -disable-mem-randomization --disable-reg-randomization | FileCheck %s --check-prefix COMMON --implicit-check-not RANDOMIZE_MEM --implicit-check-not RANDOMIZE_REG
; RUN: firtool %s --format=fir --ir-sv -disable-all-randomization | FileCheck %s --check-prefix COMMON --implicit-check-not RANDOMIZE_MEM --implicit-check-not RANDOMIZE_REG

circuit Qux: %[[{
    "class": "sifive.enterprise.firrtl.MarkDUTAnnotation",
    "target":"~Qux|Qux"
  }, 
  {
    "class": "sifive.enterprise.firrtl.ConvertMemToRegOfVecAnnotation$"
  }]]
  module Qux: 
    input clock: Clock
    input rAddr: UInt<2>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wAddr: UInt<2>
    input wEn: UInt<1>
    input wMask: UInt<1>
    input wData: UInt<8>
    input rwEn: UInt<1>
    input rwMode: UInt<1>
    input rwAddr: UInt<2>
    input rwMask: UInt<1>
    input rwDataIn: UInt<8>
    output rwDataOut: UInt<8>

    mem memory:
      data-type => UInt<8>
      depth => 4
      reader => r
      writer => w
      readwriter => rw
      read-latency => 1
      write-latency => 1
      read-under-write => undefined

    ; All of these are unified together
    memory.r.clk <= clock
    memory.r.en <= rEn
    memory.r.addr <= rAddr
    ; This is split
    rData <= memory.r.data

    ; All of these are unified together
    memory.w.clk <= clock
    memory.w.en <= wEn
    memory.w.addr <= wAddr
    ; These two are split
    memory.w.mask <= wMask
    memory.w.data <= wData

    ; All of these are unified together
    memory.rw.clk <= clock
    memory.rw.en <= rwEn
    memory.rw.addr <= rwAddr
    memory.rw.wmode <= rwMode
    ; All of these are split
    memory.rw.wmask <= rwMask
    memory.rw.wdata <= rwDataIn
    rwDataOut <= memory.rw.rdata



; This test is quite fragile, both as written, and in that it depends on
; multiple passes.  It should be replaced with a narrower test.

;COMMON-LABEL: hw.module @Qux
;CHECK:    %[[memory_0:.+]] = sv.reg
;CHECK:    %[[memory_1:.+]] = sv.reg
;CHECK:    %[[memory_2:.+]] = sv.reg
;CHECK:    %[[memory_3:.+]] = sv.reg
;CHECK:    %[[addr:.+]] = sv.reg
;CHECK:    %[[v4:.+]] = sv.read_inout %[[addr]]
;CHECK:    %[[v5:.+]] = hw.array_create
;CHECK:    %[[v6:.+]] = hw.array_get %[[v5]][%[[v4]]]
;CHECK:    %[[v7:.+]] = hw.array_get %[[v5]][%rwAddr]
;CHECK:    %8 = comb.icmp bin eq %rwAddr, %c0_i2 : i2
;CHECK:    %9 = comb.and bin %rwEn, %rwMode, %rwMask, %8 : i1
;CHECK:    %10 = comb.icmp bin eq %rwAddr, %c1_i2 : i2
;CHECK:    %11 = comb.and bin %rwEn, %rwMode, %rwMask, %10 : i1
;CHECK:    %12 = comb.icmp bin eq %rwAddr, %c-2_i2 : i2
;CHECK:    %13 = comb.and bin %rwEn, %rwMode, %rwMask, %12 : i1
;CHECK:    %14 = comb.icmp bin eq %rwAddr, %c-1_i2 : i2
;CHECK:    %15 = comb.and bin %rwEn, %rwMode, %rwMask, %14 : i1
;CHECK:    %16 = comb.icmp bin eq %wAddr, %c0_i2 : i2
;CHECK:    %17 = comb.and bin %wEn, %wMask, %16 : i1
;CHECK:    %18 = comb.icmp bin eq %wAddr, %c1_i2 : i2
;CHECK:    %19 = comb.and bin %wEn, %wMask, %18 : i1
;CHECK:    %20 = comb.icmp bin eq %wAddr, %c-2_i2 : i2
;CHECK:    %21 = comb.and bin %wEn, %wMask, %20 : i1
;CHECK:    %22 = comb.icmp bin eq %wAddr, %c-1_i2 : i2
;CHECK:    %23 = comb.and bin %wEn, %wMask, %22 : i1
;CHECK:    sv.always posedge %clock {
;CHECK:      sv.if %17 {
;CHECK:        sv.passign %memory_0, %wData : i8
;CHECK:      } else {
;CHECK:        sv.if %9 {
;CHECK:          sv.passign %memory_0, %rwDataIn : i8
;CHECK:        }
;CHECK:      }
;CHECK:      sv.if %19 {
;CHECK:        sv.passign %memory_1, %wData : i8
;CHECK:      } else {
;CHECK:        sv.if %11 {
;CHECK:          sv.passign %memory_1, %rwDataIn : i8
;CHECK:        }
;CHECK:      }
;CHECK:      sv.if %21 {
;CHECK:        sv.passign %memory_2, %wData : i8
;CHECK:      } else {
;CHECK:        sv.if %13 {
;CHECK:          sv.passign %memory_2, %rwDataIn : i8
;CHECK:        }
;CHECK:      }
;CHECK:      sv.if %23 {
;CHECK:        sv.passign %memory_3, %wData : i8
;CHECK:      } else {
;CHECK:        sv.if %15 {
;CHECK:          sv.passign %memory_3, %rwDataIn : i8
;CHECK:        }
;CHECK:      }
;CHECK:      sv.passign %addr, %rAddr : i2
;CHECK:    }
;CHECK:    hw.output %[[v6]], %[[v7]]
