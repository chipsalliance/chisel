// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.probe.{Probe, ProbeValue}
import chisel3.properties.Property
import chiselTests.experimental.hierarchy.Utils
import circt.stage.ChiselStage

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

class HasDeadCodeChildLeaves() extends Module {
  val io = IO(new Bundle {
    val a = Input(new Bundle { val a1 = UInt(32.W); val a2 = UInt(32.W) })
    val b = Output(new Bundle { val b1 = UInt(32.W); val b2 = UInt(32.W) })
  })

  io.b.b1 := io.a.a1
  io.b.b2 := DontCare
  dontTouch(io.a)
}

class HasDeadCodeLeaves() extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(32.W))
    val b = Output(UInt(32.W))
  })
  val inst = Module(new HasDeadCodeChildLeaves())
  inst.io.a.a1 := io.a
  inst.io.a.a2 := io.a
  val tmp = inst.io.b.b1 + inst.io.b.b2
  dontTouch(tmp)
  io.b := tmp
}

class HasProbesAndProperties() extends Module {
  val io = IO(new Bundle {
    val a = Output(UInt(32.W))
    val probe = Output(Probe(UInt(32.W)))
    val prop = Output(Property[Int]())
  })
  io.a := DontCare
  io.probe := probe.ProbeValue(io.a)
  io.prop := Property(5)

  dontTouch(io)
}

object OptTest {
  def apply(reset: Option[Bool]): Unit = {
    reset.map(dontTouch.apply)
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

  "fields" should "be marked don't touch by default" in {
    val (_, annos) = getFirrtlAndAnnos(new HasDeadCodeLeaves())
    annos should contain(DontTouchAnnotation("~HasDeadCodeLeaves|HasDeadCodeChildLeaves>io.a.a1".rt))
    annos should not contain (DontTouchAnnotation("~HasDeadCodeLeaves|HasDeadCodeChildLeaves>io.a".rt))
  }

  "probes and properties" should "NOT be marked dontTouch" in {
    val (_, annos) = getFirrtlAndAnnos(new HasProbesAndProperties())
    // Check for DontTouch on io.a but not on the probe or property leaves.
    annos should contain(DontTouchAnnotation("~HasProbesAndProperties|HasProbesAndProperties>io.a".rt))
    annos should not contain (DontTouchAnnotation("~HasProbesAndProperties|HasProbesAndProperties>io.probe".rt))
    annos should not contain (DontTouchAnnotation("~HasProbesAndProperties|HasProbesAndProperties>io.prop".rt))

    // Ensure can compile the result.
    compile(new HasProbesAndProperties())
  }
}
