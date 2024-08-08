// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" --java-opt="--enable-native-access=ALL-UNNAMED" --java-opt="--enable-preview" --java-opt="-Djava.library.path=%JAVALIBRARYPATH" %s | FileCheck %s -check-prefix=FIRRTL
// SPDX-License-Identifier: Apache-2.0

import chisel3._
import chisel3.experimental.{Analog, attach}
import chisel3.util.SRAM

// FIRRTL-LABEL: public module Attach :
// FIRRTL-NEXT:   input clock : Clock
// FIRRTL-NEXT:   input reset : UInt<1>
class Attach extends Module {
  // FIRRTL-NEXT: output o : Analog<1>
  val o = IO(Analog(1.W))
  // FIRRTL-NEXT: output i : Analog<1>
  val i = IO(Analog(1.W))

  // FIRRTL:      attach(i, o)
  attach(i, o)
}

println(lit.utility.panamaconverter.firrtlString(new Attach))

// FIRRTL-LABEL: public module Cond :
// FIRRTL-NEXT:   input clock : Clock
// FIRRTL-NEXT:   input reset : UInt<1>
class Cond extends Module {
  // FIRRTL-NEXT: input i : UInt<2>
  val i = IO(Input(UInt(2.W)))
  // FIRRTL-NEXT: output o : UInt<3>
  val o = IO(Output(UInt(3.W)))

  // FIRRTL: node _T = eq(i, UInt<1>(1))
  // FIRRTL-NEXT: when _T :
  when(i === "b01".U) {
    // FIRRTL-NEXT: connect o, pad(UInt<1>(0), 3)
    o := 0.U
    // FIRRTL-NEXT: else :
    // FIRRTL-NEXT:   node _T_1 = eq(i, UInt<2>(2))
    // FIRRTL-NEXT:   when _T_1 :
  }.elsewhen(i === "b10".U) {
    // FIRRTL-NEXT:   connect o, pad(UInt<1>(1), 3)
    o := 1.U
    // FIRRTL-NEXT:   else :
  }.otherwise {
    // FIRRTL-NEXT:     connect o, UInt<3>(4)
    o := 4.U
  }
}

println(lit.utility.panamaconverter.firrtlString(new Cond))

// FIRRTL-LABEL: public module Mem :
// FIRRTL-NEXT:   input clock : Clock
// FIRRTL-NEXT:   input reset : UInt<1>
class Mem extends Module {
  // FIRRTL: output r : { flip enable : UInt<1>,
  val r = IO(new Bundle {
    val enable = Input(Bool())
    // FIRRTL-NEXT:       flip address : UInt<10>,
    val address = Input(UInt(10.W))
    // FIRRTL-NEXT:            data : UInt<32> }
    val data = Output(UInt(32.W))
  })
  // FIRRTL: output w : { flip enable : UInt<1>,
  val w = IO(new Bundle {
    val enable = Input(Bool())
    // FIRRTL-NEXT:       flip address : UInt<10>,
    val address = Input(UInt(10.W))
    // FIRRTL-NEXT:       flip data : UInt<32>
    val data = Input(UInt(32.W))
  })

  // FIRRTL: smem mem : UInt<32>[1024]
  val mem = SyncReadMem(1024, UInt(32.W))

  // FIRRTL: invalidate r.data
  r.data := DontCare
  // FIRRTL: when r.enable :
  when (r.enable) {
    // FIRRTL-NEXT: infer mport wrPort = mem[r.address], clock
    val wrPort = mem(r.address)
    // FIRRTL-NEXT: connect r.data, wrPort
    r.data := wrPort
    // NOTE: PanamaConverter emits empty 'else :' here in FIRRTL, which is not
    //       valid when fed into firtool. But direct verilog lowering seems fine.
  }

  // FIRRTL: when w.enable :
  when (w.enable) {
    // FIRRTL-NEXT: infer mport wrPort_1 = mem[w.address], clock
    val wrPort = mem(w.address)
    // FIRRTL-NEXT: connect wrPort_1, w.data
    wrPort := w.data
  }
}

println(lit.utility.panamaconverter.firrtlString(new Mem))

// FIRRTL-LABEL: public module Sram :
// FIRRTL-NEXT:   input clock : Clock
// FIRRTL-NEXT:   input reset : UInt<1>
class Sram extends Module {
  // FIRRTL:      wire mem
  // FIRRTL:      mem mem_sram
  // FIRRTL:      connect mem_sram.R0.addr, mem.readPorts[0].address
  // FIRRTL-NEXT: connect mem_sram.R0.clk, clock
  // FIRRTL-NEXT: connect mem.readPorts[0].data, mem_sram.R0.data
  // FIRRTL-NEXT: connect mem_sram.R0.en, mem.readPorts[0].enable
  // FIRRTL-NEXT: connect mem_sram.R1.addr, mem.readPorts[1].address
  // FIRRTL-NEXT: connect mem_sram.R1.clk, clock
  // FIRRTL-NEXT: connect mem.readPorts[1].data, mem_sram.R1.data
  // FIRRTL-NEXT: connect mem_sram.R1.en, mem.readPorts[1].enable
  // FIRRTL-NEXT: connect mem_sram.W0.addr, mem.writePorts[0].address
  // FIRRTL-NEXT: connect mem_sram.W0.clk, clock
  // FIRRTL-NEXT: connect mem_sram.W0.data, mem.writePorts[0].data
  // FIRRTL-NEXT: connect mem_sram.W0.en, mem.writePorts[0].enable
  // FIRRTL-NEXT: connect mem_sram.W0.mask, UInt<1>(1)
  // FIRRTL-NEXT: connect mem_sram.W1.addr, mem.writePorts[1].address
  // FIRRTL-NEXT: connect mem_sram.W1.clk, clock
  // FIRRTL-NEXT: connect mem_sram.W1.data, mem.writePorts[1].data
  // FIRRTL-NEXT: connect mem_sram.W1.en, mem.writePorts[1].enable
  // FIRRTL-NEXT: connect mem_sram.W1.mask, UInt<1>(1)
  // FIRRTL-NEXT: connect mem_sram.RW0.addr, mem.readwritePorts[0].address
  // FIRRTL-NEXT: connect mem_sram.RW0.clk, clock
  // FIRRTL-NEXT: connect mem_sram.RW0.en, mem.readwritePorts[0].enable
  // FIRRTL-NEXT: connect mem.readwritePorts[0].readData, mem_sram.RW0.rdata
  // FIRRTL-NEXT: connect mem_sram.RW0.wdata, mem.readwritePorts[0].writeData
  // FIRRTL-NEXT: connect mem_sram.RW0.wmode, mem.readwritePorts[0].isWrite
  // FIRRTL-NEXT: connect mem_sram.RW0.wmask, UInt<1>(1)
  // FIRRTL-NEXT: connect mem_sram.RW1.addr, mem.readwritePorts[1].address
  // FIRRTL-NEXT: connect mem_sram.RW1.clk, clock
  // FIRRTL-NEXT: connect mem_sram.RW1.en, mem.readwritePorts[1].enable
  // FIRRTL-NEXT: connect mem.readwritePorts[1].readData, mem_sram.RW1.rdata
  // FIRRTL-NEXT: connect mem_sram.RW1.wdata, mem.readwritePorts[1].writeData
  // FIRRTL-NEXT: connect mem_sram.RW1.wmode, mem.readwritePorts[1].isWrite
  // FIRRTL-NEXT: connect mem_sram.RW1.wmask, UInt<1>(1)
  // FIRRTL-NEXT: connect mem_sram.RW2.addr, mem.readwritePorts[2].address
  // FIRRTL-NEXT: connect mem_sram.RW2.clk, clock
  // FIRRTL-NEXT: connect mem_sram.RW2.en, mem.readwritePorts[2].enable
  // FIRRTL-NEXT: connect mem.readwritePorts[2].readData, mem_sram.RW2.rdata
  // FIRRTL-NEXT: connect mem_sram.RW2.wdata, mem.readwritePorts[2].writeData
  // FIRRTL-NEXT: connect mem_sram.RW2.wmode, mem.readwritePorts[2].isWrite
  // FIRRTL-NEXT: connect mem_sram.RW2.wmask, UInt<1>(1)
  val mem = SRAM(1024, UInt(8.W), 2, 2, 3)

  // FIRRTL-NEXT: connect mem.readPorts[0].address, pad(UInt<7>(100), 10)
  // FIRRTL-NEXT: connect mem.readPorts[0].enable, UInt<1>(1)
  mem.readPorts(0).address := 100.U
  mem.readPorts(0).enable := true.B

  // FIRRTL-NEXT: wire foo : UInt<8>
  // FIRRTL-NEXT: connect foo, mem.readPorts[0].data
  val foo = WireInit(UInt(8.W), mem.readPorts(0).data)

  // FIRRTL-NEXT: connect mem.writePorts[1].address, pad(UInt<3>(5), 10)
  // FIRRTL-NEXT: connect mem.writePorts[1].enable, UInt<1>(1)
  // FIRRTL-NEXT: connect mem.writePorts[1].data, pad(UInt<4>(12), 8)
  mem.writePorts(1).address := 5.U
  mem.writePorts(1).enable := true.B
  mem.writePorts(1).data := 12.U

  // FIRRTL-NEXT: connect mem.readwritePorts[2].address, pad(UInt<3>(5), 10)
  // FIRRTL-NEXT: connect mem.readwritePorts[2].enable, UInt<1>(1)
  // FIRRTL-NEXT: connect mem.readwritePorts[2].isWrite, UInt<1>(1)
  // FIRRTL-NEXT: connect mem.readwritePorts[2].writeData, pad(UInt<7>(100), 8)
  mem.readwritePorts(2).address := 5.U
  mem.readwritePorts(2).enable := true.B
  mem.readwritePorts(2).isWrite := true.B
  mem.readwritePorts(2).writeData := 100.U

  // FIRRTL-NEXT: connect mem.readwritePorts[2].address, pad(UInt<3>(5), 10)
  // FIRRTL-NEXT: connect mem.readwritePorts[2].enable, UInt<1>(1)
  // FIRRTL-NEXT: connect mem.readwritePorts[2].isWrite, UInt<1>(0)
  mem.readwritePorts(2).address := 5.U
  mem.readwritePorts(2).enable := true.B
  mem.readwritePorts(2).isWrite := false.B

  // FIRRTL-NEXT: wire bar : UInt<8>
  // FIRRTL-NEXT: connect bar, mem.readwritePorts[2].readData
  val bar = WireInit(UInt(8.W), mem.readwritePorts(2).readData)
}

println(lit.utility.panamaconverter.firrtlString(new Sram))

// FIRRTL-LABEL: public module WireAndReg :
// FIRRTL-NEXT:   input clock : Clock
// FIRRTL-NEXT:   input reset : UInt<1>
class WireAndReg extends Module {
  // FIRRTL-NEXT: input r : UInt<1>
  val r = IO(Input(Bool()))
  // FIRRTL-NEXT: output o : UInt<2>
  val o = IO(Output(UInt(2.W)))

  // FIRRTL: regreset o_next : UInt<1>, clock, reset, UInt<1>(0)
  val o_next = RegInit(false.B)
  // FIRRTL: reg flip : UInt<1>, clock
  val flip = Reg(Bool())
  // FIRRTL: wire magic : SInt<8>
  val magic = Wire(SInt(8.W))

  // FIRRTL:      bits(magic, 7, 7)
  // FIRRTL-NEXT: and(o_next
  // FIRRTL-NEXT: connect o
  o := o_next && magic(7)
  // FIRRTL:      connect o_next, flip
  o_next := flip
  // FIRRTL:      xor(flip, r)
  flip := flip ^ r
  // FIRRTL:      connect magic, pad(SInt<7>(-42), 8)
  magic := -42.S
}

println(lit.utility.panamaconverter.firrtlString(new WireAndReg))
