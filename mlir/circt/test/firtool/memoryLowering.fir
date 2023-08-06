; RUN: firtool %s --annotation-file %s.anno.json -repl-seq-mem -repl-seq-mem-file="test.conf" |  FileCheck %s

circuit test:
  module test:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>[4]
    input wMask: UInt<1>[4]
    input wData: UInt<8>[4]

    mem memory:
      data-type => UInt<8>[4]
      depth => 16
      reader => r
      writer => w
      read-latency => 1
      write-latency => 1
      read-under-write => undefined

    ; All of these are unified together
    memory.r.clk <= clock
    memory.r.en <= rEn
    memory.r.addr <= rAddr
    rData <= memory.r.data

    memory.w.clk <= clock
    memory.w.en <= rEn
    memory.w.addr <= rAddr
    ; These two are split
    memory.w.mask <= wMask
    memory.w.data <= wData

; CHECK: external module memory_ext

; CHECK-LABEL: FILE "metadata{{[/\]}}seq_mems.json"
; CHECK:      [
; CHECK-NEXT:   {
; CHECK-NEXT:     "module_name": "memory_ext",
; CHECK-NEXT:     "depth": 16,
; CHECK-NEXT:     "width": 32,
; CHECK-NEXT:     "masked": true,
; CHECK-NEXT:     "read": 1,
; CHECK-NEXT:     "write": 1,
; CHECK-NEXT:     "readwrite": 0,
; CHECK-NEXT:     "mask_granularity": 8,
; CHECK-NEXT:     "extra_ports": [],
; CHECK-NEXT:     "hierarchy": [
; CHECK-NEXT:       "test.memory.memory_ext"
; CHECK-NEXT:     ]
; CHECK-NEXT:   }
; CHECK-NEXT: ]

; CHECK: test.conf

; CHECK: name memory_ext depth 16 width 32 ports mwrite,read mask_gran 8
