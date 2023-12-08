// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage
import chiselTests.experimental.hierarchy.Utils

import firrtl.transforms.DontTouchAnnotation

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

class HasDeadCodeChildLeaves(withDontTouchAgg: Boolean) extends Module {
  val io = IO(new Bundle {
    val a = Input(new Bundle { val a1 = UInt(32.W); val a2 = UInt(32.W) })
    val b = Output(new Bundle { val b1 = UInt(32.W); val b2 = UInt(32.W) })
  })

  io.b.b1 := io.a.a1
  io.b.b2 := DontCare
  dontTouch(io.a, withDontTouchAgg)
}

class HasDeadCodeLeaves(withDontTouchAgg: Boolean) extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(32.W))
    val b = Output(UInt(32.W))
  })
  val inst = Module(new HasDeadCodeChildLeaves(withDontTouchAgg))
  inst.io.a.a1 := io.a
  inst.io.a.a2 := io.a
  val tmp = inst.io.b.b1 + inst.io.b.b2
  dontTouch(tmp, withDontTouchAgg)
  io.b := tmp
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

  "fields" should "be marked don't touch by default" in {
    val (_, annos) = getFirrtlAndAnnos(new HasDeadCodeLeaves(false))
    annos should contain(DontTouchAnnotation("~HasDeadCodeLeaves|HasDeadCodeChildLeaves>io.a.a1".rt))
    annos should not contain (DontTouchAnnotation("~HasDeadCodeLeaves|HasDeadCodeChildLeaves>io.a".rt))
  }
  "aggregates" should "be marked if marked markAgg is true" in {
    val (_, annos) = getFirrtlAndAnnos(new HasDeadCodeLeaves(true))
    annos should not contain (DontTouchAnnotation("~HasDeadCodeLeaves|HasDeadCodeChildLeaves>io.a.a1".rt))
    annos should contain(DontTouchAnnotation("~HasDeadCodeLeaves|HasDeadCodeChildLeaves>io.a".rt))
  }
}
