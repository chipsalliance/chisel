; RUN: firtool --verilog %s | FileCheck %s

circuit Top : %[[
  {
    "class": "sifive.enterprise.firrtl.MarkDUTAnnotation",
    "target":"~Top|DUTModule"
  },
  {
    "class":"firrtl.transforms.DontTouchAnnotation",
    "target":"~Top|Top>memTap"
  },
  {"class": "sifive.enterprise.firrtl.ConvertMemToRegOfVecAnnotation$"},
  {
    "class":"sifive.enterprise.grandcentral.MemTapAnnotation",
    "source":"~Top|DUTModule>rf",
    "sink":[
      "~Top|Top>memTap[0]",
      "~Top|Top>memTap[1]",
      "~Top|Top>memTap[2]",
      "~Top|Top>memTap[3]",
      "~Top|Top>memTap[4]",
      "~Top|Top>memTap[5]",
      "~Top|Top>memTap[6]",
      "~Top|Top>memTap[7]"
    ]
  }
]]
  module DUTModule :
    input clock : Clock
    input reset : Reset
    output io : { flip addr : UInt<3>, flip dataIn : UInt<8>, flip wen : UInt<1>, dataOut : UInt<8>}

    cmem rf : UInt<8> [8]
    infer mport read = rf[io.addr], clock
    io.dataOut <= read
    when io.wen :
      infer mport write = rf[io.addr], clock
      write <= io.dataIn

  module Top :
    input clock : Clock
    input reset : UInt<1>
    output io : { flip addr : UInt<3>, flip dataIn : UInt<8>, flip wen : UInt<1>, dataOut : UInt<8>}

    inst dut of DUTModule
    dut.clock <= clock
    dut.reset <= reset
    wire memTap : UInt<8>[8]
    memTap is invalid
    io.dataOut <= dut.io.dataOut
    dut.io.wen <= io.wen
    dut.io.dataIn <= io.dataIn
    dut.io.addr <= io.addr

; CHECK:      module Top(
; CHECK-NOT:  module
; CHECK:        wire [7:0] memTap_0 = Top.dut.rf_0_probe;
; CHECK-NEXT:   wire [7:0] memTap_1 = Top.dut.rf_1_probe;
; CHECK-NEXT:   wire [7:0] memTap_2 = Top.dut.rf_2_probe;
; CHECK-NEXT:   wire [7:0] memTap_3 = Top.dut.rf_3_probe;
; CHECK-NEXT:   wire [7:0] memTap_4 = Top.dut.rf_4_probe;
; CHECK-NEXT:   wire [7:0] memTap_5 = Top.dut.rf_5_probe;
; CHECK-NEXT:   wire [7:0] memTap_6 = Top.dut.rf_6_probe;
; CHECK-NEXT:   wire [7:0] memTap_7 = Top.dut.rf_7_probe;
; CHECK:      endmodule
