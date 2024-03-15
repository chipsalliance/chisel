// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" --java-opt="--enable-native-access=ALL-UNNAMED" --java-opt="--enable-preview" --java-opt="-Djava.library.path=%JAVALIBRARYPATH" %s | FileCheck %s -check-prefix=FIRRTL
// SPDX-License-Identifier: Apache-2.0

import chisel3._
import chisel3.experimental.{Analog, attach}

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
