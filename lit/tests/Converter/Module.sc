// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" %s | FileCheck %s
// SPDX-License-Identifier: Apache-2.0

import chisel3._
import chisel3.experimental.{Analog, attach}
import chisel3.util.SRAM

// CHECK-LABEL: public module Attach :
// CHECK-NEXT:   input clock : Clock
// CHECK-NEXT:   input reset : UInt<1>
class Attach extends Module {
  // CHECK-NEXT: output o : Analog<1>
  val o = IO(Analog(1.W))
  // CHECK-NEXT: output i : Analog<1>
  val i = IO(Analog(1.W))

  // CHECK:      attach (i, o)
  attach(i, o)
}

println(circt.stage.ChiselStage.emitCHIRRTL(new Attach))

// CHECK-LABEL: public module Cond :
// CHECK-NEXT:   input clock : Clock
// CHECK-NEXT:   input reset : UInt<1>
class Cond extends Module {
  // CHECK-NEXT: input i : UInt<2>
  val i = IO(Input(UInt(2.W)))
  // CHECK-NEXT: output o : UInt<3>
  val o = IO(Output(UInt(3.W)))

  // CHECK: node _T = eq(i, UInt<1>(0h1))
  // CHECK-NEXT: when _T :
  when(i === "b01".U) {
    // CHECK-NEXT: connect o, UInt<1>(0h0)
    o := 0.U
    // CHECK-NEXT: else :
    // CHECK-NEXT:   node _T_1 = eq(i, UInt<2>(0h2))
    // CHECK-NEXT:   when _T_1 :
  }.elsewhen(i === "b10".U) {
    // CHECK-NEXT:   connect o, UInt<1>(0h1)
    o := 1.U
    // CHECK-NEXT:   else :
  }.otherwise {
    // CHECK-NEXT:     connect o, UInt<3>(0h4)
    o := 4.U
  }
}

println(circt.stage.ChiselStage.emitCHIRRTL(new Cond))

// CHECK-LABEL: public module Mem :
// CHECK-NEXT:   input clock : Clock
// CHECK-NEXT:   input reset : UInt<1>
class Mem extends Module {
  // CHECK: output r : { flip enable : UInt<1>,
  val r = IO(new Bundle {
    val enable = Input(Bool())
    // CHECK-SAME:       flip address : UInt<10>,
    val address = Input(UInt(10.W))
    // CHECK-SAME:            data : UInt<32>}
    val data = Output(UInt(32.W))
  })
  // CHECK: output w : { flip enable : UInt<1>,
  val w = IO(new Bundle {
    val enable = Input(Bool())
    // CHECK-SAME:       flip address : UInt<10>,
    val address = Input(UInt(10.W))
    // CHECK-SAME:       flip data : UInt<32>
    val data = Input(UInt(32.W))}
  })

  // CHECK: smem mem : UInt<32>[1024]
  val mem = SyncReadMem(1024, UInt(32.W))

  // CHECK: invalidate r.data
  r.data := DontCare
  // CHECK: when r.enable :
  when (r.enable) {
    // CHECK-NEXT: infer mport wrPort = mem[r.address], clock
    val wrPort = mem(r.address)
    // CHECK-NEXT: connect r.data, wrPort
    r.data := wrPort
    // NOTE: PanamaConverter emits empty 'else :' here in CHECK, which is not
    //       valid when fed into firtool. But direct verilog lowering seems fine.
  }

  // CHECK: when w.enable :
  when (w.enable) {
    // CHECK-NEXT: infer mport wrPort_1 = mem[w.address], clock
    val wrPort = mem(w.address)
    // CHECK-NEXT: connect wrPort_1, w.data
    wrPort := w.data
  }
}

println(circt.stage.ChiselStage.emitCHIRRTL(new Mem))

// CHECK-LABEL: public module Sram :
// CHECK-NEXT:   input clock : Clock
// CHECK-NEXT:   input reset : UInt<1>
class Sram extends Module {
  // CHECK:      wire mem
  // CHECK:      mem mem_sram
  // CHECK:      connect mem_sram.R0.addr, mem.readPorts[0].address
  // CHECK-NEXT: connect mem_sram.R0.clk, clock
  // CHECK-NEXT: connect mem.readPorts[0].data, mem_sram.R0.data
  // CHECK-NEXT: connect mem_sram.R0.en, mem.readPorts[0].enable
  // CHECK-NEXT: connect mem_sram.R1.addr, mem.readPorts[1].address
  // CHECK-NEXT: connect mem_sram.R1.clk, clock
  // CHECK-NEXT: connect mem.readPorts[1].data, mem_sram.R1.data
  // CHECK-NEXT: connect mem_sram.R1.en, mem.readPorts[1].enable
  // CHECK-NEXT: connect mem_sram.W0.addr, mem.writePorts[0].address
  // CHECK-NEXT: connect mem_sram.W0.clk, clock
  // CHECK-NEXT: connect mem_sram.W0.data, mem.writePorts[0].data
  // CHECK-NEXT: connect mem_sram.W0.en, mem.writePorts[0].enable
  // CHECK-NEXT: connect mem_sram.W0.mask, UInt<1>(0h1)
  // CHECK-NEXT: connect mem_sram.W1.addr, mem.writePorts[1].address
  // CHECK-NEXT: connect mem_sram.W1.clk, clock
  // CHECK-NEXT: connect mem_sram.W1.data, mem.writePorts[1].data
  // CHECK-NEXT: connect mem_sram.W1.en, mem.writePorts[1].enable
  // CHECK-NEXT: connect mem_sram.W1.mask, UInt<1>(0h1)
  // CHECK-NEXT: connect mem_sram.RW0.addr, mem.readwritePorts[0].address
  // CHECK-NEXT: connect mem_sram.RW0.clk, clock
  // CHECK-NEXT: connect mem_sram.RW0.en, mem.readwritePorts[0].enable
  // CHECK-NEXT: connect mem.readwritePorts[0].readData, mem_sram.RW0.rdata
  // CHECK-NEXT: connect mem_sram.RW0.wdata, mem.readwritePorts[0].writeData
  // CHECK-NEXT: connect mem_sram.RW0.wmode, mem.readwritePorts[0].isWrite
  // CHECK-NEXT: connect mem_sram.RW0.wmask, UInt<1>(1)
  // CHECK-NEXT: connect mem_sram.RW1.addr, mem.readwritePorts[1].address
  // CHECK-NEXT: connect mem_sram.RW1.clk, clock
  // CHECK-NEXT: connect mem_sram.RW1.en, mem.readwritePorts[1].enable
  // CHECK-NEXT: connect mem.readwritePorts[1].readData, mem_sram.RW1.rdata
  // CHECK-NEXT: connect mem_sram.RW1.wdata, mem.readwritePorts[1].writeData
  // CHECK-NEXT: connect mem_sram.RW1.wmode, mem.readwritePorts[1].isWrite
  // CHECK-NEXT: connect mem_sram.RW1.wmask, UInt<1>(1)
  // CHECK-NEXT: connect mem_sram.RW2.addr, mem.readwritePorts[2].address
  // CHECK-NEXT: connect mem_sram.RW2.clk, clock
  // CHECK-NEXT: connect mem_sram.RW2.en, mem.readwritePorts[2].enable
  // CHECK-NEXT: connect mem.readwritePorts[2].readData, mem_sram.RW2.rdata
  // CHECK-NEXT: connect mem_sram.RW2.wdata, mem.readwritePorts[2].writeData
  // CHECK-NEXT: connect mem_sram.RW2.wmode, mem.readwritePorts[2].isWrite
  // CHECK-NEXT: connect mem_sram.RW2.wmask, UInt<1>(1)
  val mem = SRAM(1024, UInt(8.W), 2, 2, 3)

  // CHECK-NEXT: connect mem.readPorts[0].address, UInt<7>(0h100)
  // CHECK-NEXT: connect mem.readPorts[0].enable, UInt<1>(1)
  mem.readPorts(0).address := 100.U
  mem.readPorts(0).enable := true.B

  // CHECK-NEXT: wire foo : UInt<8>
  // CHECK-NEXT: connect foo, mem.readPorts[0].data
  val foo = WireInit(UInt(8.W), mem.readPorts(0).data)

  // CHECK-NEXT: connect mem.writePorts[1].address, UInt<3>(0h5)
  // CHECK-NEXT: connect mem.writePorts[1].enable, UInt<1>(1)
  // CHECK-NEXT: connect mem.writePorts[1].data, UInt<4>(0h12)
  mem.writePorts(1).address := 5.U
  mem.writePorts(1).enable := true.B
  mem.writePorts(1).data := 12.U

  // CHECK-NEXT: connect mem.readwritePorts[2].address, UInt<3>(0h5)
  // CHECK-NEXT: connect mem.readwritePorts[2].enable, UInt<1>(1)
  // CHECK-NEXT: connect mem.readwritePorts[2].isWrite, UInt<1>(1)
  // CHECK-NEXT: connect mem.readwritePorts[2].writeData, UInt<7>(0h100)
  mem.readwritePorts(2).address := 5.U
  mem.readwritePorts(2).enable := true.B
  mem.readwritePorts(2).isWrite := true.B
  mem.readwritePorts(2).writeData := 100.U

  // CHECK-NEXT: connect mem.readwritePorts[2].address, UInt<3>(0h5)
  // CHECK-NEXT: connect mem.readwritePorts[2].enable, UInt<1>(1)
  // CHECK-NEXT: connect mem.readwritePorts[2].isWrite, UInt<1>(0)
  mem.readwritePorts(2).address := 5.U
  mem.readwritePorts(2).enable := true.B
  mem.readwritePorts(2).isWrite := false.B

  // CHECK-NEXT: wire bar : UInt<8>
  // CHECK-NEXT: connect bar, mem.readwritePorts[2].readData
  val bar = WireInit(UInt(8.W), mem.readwritePorts(2).readData)
}

println(circt.stage.ChiselStage.emitCHIRRTL(new Sram))

// CHECK-LABEL: public module WireAndReg :
// CHECK-NEXT:   input clock : Clock
// CHECK-NEXT:   input reset : UInt<1>
class WireAndReg extends Module {
  // CHECK-NEXT: input r : UInt<1>
  val r = IO(Input(Bool()))
  // CHECK-NEXT: output o : UInt<2>
  val o = IO(Output(UInt(2.W)))

  // CHECK: regreset o_next : UInt<1>, clock, reset, UInt<1>(0h0)
  val o_next = RegInit(false.B)
  // CHECK: reg flip : UInt<1>, clock
  val flip = Reg(Bool())
  // CHECK: wire magic : SInt<8>
  val magic = Wire(SInt(8.W))

  // CHECK:      bits(magic, 7, 7)
  // CHECK-NEXT: and(o_next
  // CHECK-NEXT: connect o
  o := o_next && magic(7)
  // CHECK:      connect o_next, flip
  o_next := flip
  // CHECK:      xor(flip, r)
  flip := flip ^ r
  // CHECK:      connect magic, pad(SInt<7>(-42), 8)
  magic := -42.S
}

println(circt.stage.ChiselStage.emitCHIRRTL(new WireAndReg))
