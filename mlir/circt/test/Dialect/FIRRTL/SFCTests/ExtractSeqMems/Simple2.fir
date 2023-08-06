; UNSUPPORTED: system-windows
;   See https://github.com/llvm/circt/issues/4128
; RUN: rm -rf %t
; RUN: firtool --repl-seq-mem --repl-seq-mem-file=mems.conf --split-verilog -o=%t %s
; RUN: FileCheck %s --check-prefix=TESTHARNESS < %t/TestHarness.sv
; RUN: FileCheck %s --check-prefix=DUTMODULE < %t/DUTModule.sv
; RUN: FileCheck %s --check-prefix=MEM < %t/mem.sv
; RUN: FileCheck %s --check-prefix=MEMS-CONF < %t/mems.conf
; RUN: FileCheck %s --check-prefix=SEQMEMS-TXT < %t/SeqMems.txt

; Extracted from test/scala/firrtl/ExtractSeqMems.scala

circuit TestHarness : %[[
  {
    "class":"sifive.enterprise.firrtl.ExtractSeqMemsFileAnnotation",
    "filename":"SeqMems.txt"
  },
  {
    "class":"sifive.enterprise.firrtl.MarkDUTAnnotation",
    "target":"TestHarness.DUTModule"
  }
]]
  ; DUTMODULE: module DUTModule
  ; DUTMODULE: mem mem
  ; DUTMODULE:   .mem_wiring_0_R0_data
  ; DUTMODULE:   .mem_wiring_0_R0_addr
  ; DUTMODULE:   .mem_wiring_0_R0_clk
  ; DUTMODULE:   .mem_wiring_0_W0_addr
  ; DUTMODULE:   .mem_wiring_0_W0_en
  ; DUTMODULE:   .mem_wiring_0_W0_clk
  ; DUTMODULE:   .mem_wiring_0_W0_data
  module DUTModule :
    input clock : Clock
    input reset : Reset
    output io : { flip addr : UInt<3>, flip dataIn : UInt<8>, flip wen : UInt<1>, dataOut : UInt<8>}

    smem mem : UInt<8> [8]
    infer mport read = mem[io.addr], clock
    io.dataOut <= read
    when io.wen :
      infer mport write = mem[io.addr], clock
      write <= io.dataIn

  ; TESTHARNESS: module TestHarness
  ; TESTHARNESS: DUTModule dut
  ; TESTHARNESS: mem_ext mem_ext
  module TestHarness :
    input clock : Clock
    input reset : UInt<1>
    output io : { flip addr : UInt<3>, flip dataIn : UInt<8>, flip wen : UInt<1>, dataOut : UInt<8>}

    inst dut of DUTModule
    dut.clock <= clock
    dut.reset <= reset
    io.dataOut <= dut.io.dataOut
    dut.io.wen <= io.wen
    dut.io.dataIn <= io.dataIn
    dut.io.addr <= io.addr

  ; MEM: module mem
  ; MEM-NOT: mem_ext mem_ext

  ; MEMS-CONF: name mem_ext depth 8 width 8 ports write,read
  ; SEQMEMS-TXT: mem_wiring_0 -> DUTModule.mem
