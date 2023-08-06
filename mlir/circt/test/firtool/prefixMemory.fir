; RUN: firtool %s --ir-fir | FileCheck %s --check-prefix=SIM-FIR --check-prefix=CHECK-FIR
; RUN: firtool %s --ir-sv | FileCheck %s --check-prefix=SIM-HW --check-prefix=CHECK-HW
; RUN: firtool %s --repl-seq-mem --repl-seq-mem-file=test.txt --ir-fir | FileCheck %s --check-prefix=REPL-FIR --check-prefix=CHECK-FIR
; RUN: firtool %s --repl-seq-mem --repl-seq-mem-file=test.txt --ir-sv | FileCheck %s --check-prefix=REPL-HW --check-prefix=CHECK-HW

circuit Foo : %[[
  {
    "class":"sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
    "prefix":"prefix1_",
    "inclusive":true,
    "target":"~Foo|Bar"
  },
  {
    "class":"sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
    "prefix":"prefix2_",
    "inclusive":true,
    "target":"~Foo|Baz"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Foo|Bar>mem"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Foo|Baz>mem"
  }
]]
  ; CHECK-FIR-LABEL: firrtl.module private @prefix1_Bar
  ; CHECK-HW-LABEL: hw.module private @prefix1_Bar
  module Bar :
    input clock : Clock
    input reset : Reset
    input readAddr : UInt<3>
    output readData : UInt<32>
    input writeEn : UInt<1>
    input writeAddr : UInt<3>
    input writeData : UInt<32>

    ; REPL-FIR:     firrtl.instance mem sym @{{[^ ]+}} @prefix1_mem
    ; REPL-HW:      hw.instance "mem" sym @{{[^ ]+}} @prefix1_mem
    ; SIM-FIR:      firrtl.mem
    ; SIM-FIR-SAME:   name = "mem"
    ; SIM-FIR-SAME:   prefix = "prefix1_"
    ; SIM-HW:       hw.instance "mem_ext" @prefix1_mem
    mem mem :
      data-type => UInt<1>
      depth => 8
      read-latency => 1
      write-latency => 1
      reader => readData_MPORT
      writer => MPORT
    mem.readData_MPORT.addr is invalid
    mem.readData_MPORT.clk is invalid
    mem.readData_MPORT.en <= UInt<1>("h0")
    mem.MPORT.addr is invalid
    mem.MPORT.clk is invalid
    mem.MPORT.en <= UInt<1>("h0")
    mem.MPORT.data is invalid
    mem.MPORT.mask is invalid
    mem.readData_MPORT.addr <= readAddr
    mem.readData_MPORT.clk <= clock
    readData <= mem.readData_MPORT.data
    when writeEn :
      mem.MPORT.addr <= writeAddr
      mem.MPORT.clk <= clock
      mem.MPORT.en <= UInt<1>("h1")
      mem.MPORT.mask <= UInt<1>("h0")
      mem.MPORT.data <= writeData
      mem.MPORT.mask <= UInt<1>("h1")

  ; CHECK-FIR-LABEL: firrtl.module private @prefix2_Baz
  ; CHECK-HW-LABEL: hw.module private @prefix2_Baz
  module Baz :
    input clock : Clock
    input reset : Reset
    input readAddr : UInt<3>
    output readData : UInt<32>
    input writeEn : UInt<1>
    input writeAddr : UInt<3>
    input writeData : UInt<32>

    ; REPL-FIR:     firrtl.instance mem sym @{{[^ ]+}} @prefix2_mem
    ; REPL-HW:      hw.instance "mem" sym @{{[^ ]+}} @prefix2_mem
    ; SIM-FIR:      firrtl.mem
    ; SIM-FIR-SAME:   name = "mem"
    ; SIM-FIR-SAME:   prefix = "prefix2_"
    ; SIM-HW:       hw.instance "mem_ext" @prefix2_mem
    mem mem :
      data-type => UInt<1>
      depth => 8
      read-latency => 1
      write-latency => 1
      reader => readData_MPORT
      writer => MPORT
    mem.readData_MPORT.addr is invalid
    mem.readData_MPORT.clk is invalid
    mem.readData_MPORT.en <= UInt<1>("h0")
    mem.MPORT.addr is invalid
    mem.MPORT.clk is invalid
    mem.MPORT.en <= UInt<1>("h0")
    mem.MPORT.data is invalid
    mem.MPORT.mask is invalid
    mem.readData_MPORT.addr <= readAddr
    mem.readData_MPORT.clk <= clock
    readData <= mem.readData_MPORT.data
    when writeEn :
      mem.MPORT.addr <= writeAddr
      mem.MPORT.clk <= clock
      mem.MPORT.en <= UInt<1>("h1")
      mem.MPORT.mask <= UInt<1>("h0")
      mem.MPORT.data <= writeData
      mem.MPORT.mask <= UInt<1>("h1")

  module Foo :
    input clock : Clock
    input reset : UInt<1>
    input readAddr : UInt<3>
    output readData : UInt<32>
    input writeEn : UInt<1>
    input writeAddr : UInt<3>
    input writeData : UInt<32>

    inst bar of Bar
    bar.clock <= clock
    bar.reset <= reset
    bar.readAddr <= readAddr
    bar.writeEn <= writeEn
    bar.writeAddr <= writeAddr
    bar.writeData <= writeData
    inst baz of Baz
    baz.clock <= clock
    baz.reset <= reset
    baz.readAddr <= readAddr
    baz.writeEn <= writeEn
    baz.writeAddr <= writeAddr
    baz.writeData <= writeData
    node _readData_T = xor(bar.readData, baz.readData)
    readData <= _readData_T

; REPL-FIR: firrtl.memmodule private @prefix1_mem
; REPL-FIR: firrtl.memmodule private @prefix2_mem

; REPL-HW: hw.module.extern @prefix1_mem
; REPL-HW: hw.module.extern @prefix2_mem
