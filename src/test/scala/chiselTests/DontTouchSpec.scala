// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage

class HasDeadCodeChild(withDontTouch: Boolean) extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(32.W))
    val b = Output(UInt(32.W))
    val c = Output(Vec(2, UInt(32.W)))
  })
  io.b := io.a
  io.c := DontCare
  if (withDontTouch) {
    dontTouch(io.c)
  }
}

class HasDeadCode(withDontTouch: Boolean) extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(32.W))
    val b = Output(UInt(32.W))
  })
  val inst = Module(new HasDeadCodeChild(withDontTouch))
  inst.io.a := io.a
  io.b := inst.io.b
  val dead = WireDefault(io.a + 1.U)
  if (withDontTouch) {
    dontTouch(dead)
  }
}

class DontTouchSpec extends ChiselFlatSpec with Utils {
  val deadSignals = List(
    "io_c_0",
    "io_c_1",
    "dead"
  )
  "Dead code" should "be removed by default" in {
    val verilog = compile(new HasDeadCode(false))
    for (signal <- deadSignals) {
      (verilog should not).include(signal)
    }
  }
  it should "NOT be removed if marked dontTouch" in {
    val verilog = compile(new HasDeadCode(true))
    for (signal <- deadSignals) {
      verilog should include(signal)
    }
  }
  "Dont touch" should "only work on bound hardware" in {
    a[chisel3.BindingException] should be thrownBy extractCause[BindingException] {
      ChiselStage.emitCHIRRTL(new Module {
        val io = IO(new Bundle {})
        dontTouch(new Bundle { val a = UInt(32.W) })
      })
    }
  }
}
