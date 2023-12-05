// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage

class HasDeadCodeChildLeaves(withDontTouch: Boolean) extends Module {
  val io = IO(new Bundle {
    val a = Input(new Bundle {val a1 = UInt(32.W); val a2 = UInt(32.W)})
    val b = Output(new Bundle {val b1 = UInt(32.W); val b2 = UInt(32.W)})
  })

  io.b.b1 := io.a.a1
  io.b.b2 := DontCare
  if (withDontTouch) {
    dontTouchLeaves(io.a)
  }
}

class HasDeadCodeLeaves(withDontTouch: Boolean) extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(32.W))
    val b = Output(UInt(32.W))
  })
  val inst = Module(new HasDeadCodeChildLeaves(withDontTouch))
  inst.io.a.a1 := io.a
  inst.io.a.a2 := io.a
  val tmp = inst.io.b.b1 + inst.io.b.b2
  if (withDontTouch)
    dontTouchLeaves(tmp)
  io.b := tmp
}

class DontTouchLeavesSpec extends ChiselFlatSpec with Utils {
  val deadSignals = List(
    "io_a_a2",
    "tmp"
  )
  "Dead code" should "be removed by default" in {
    val verilog = compile(new HasDeadCodeLeaves(false))
    for (signal <- deadSignals) {
      (verilog should not).include(signal)
    }
  }
  it should "NOT be removed if marked dontTouchLeaves" in {
    val verilog = compile(new HasDeadCodeLeaves(true))
    for (signal <- deadSignals) {
      verilog should include(signal)
    }
  }
  "Dont touch leaves" should "only work on bound hardware" in {
    a[chisel3.BindingException] should be thrownBy extractCause[BindingException] {
      ChiselStage.emitCHIRRTL(new Module {
        val io = IO(new Bundle {})
        dontTouchLeaves(new Bundle { val a = UInt(32.W) })
      })
    }
  }
}
