// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.probe.{Probe, ProbeValue}
import chisel3.properties.Property
import circt.stage.ChiselStage
import firrtl.transforms.DontTouchAnnotation
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

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

class DontTouchSpec extends AnyFlatSpec with Matchers with FileCheck {
  val deadSignals = List(
    "io_c_0",
    "io_c_1",
    "dead"
  )
  "Dead code" should "be removed by default" in {
    val verilog = ChiselStage.emitSystemVerilog(new HasDeadCode(false))
    for (signal <- deadSignals) {
      (verilog should not).include(signal)
    }
  }
  it should "NOT be removed if marked dontTouch" in {
    val verilog = ChiselStage.emitSystemVerilog(new HasDeadCode(true))
    for (signal <- deadSignals) {
      verilog should include(signal)
    }
  }
  "Dont touch" should "only work on bound hardware" in {
    a[BindingException] should be thrownBy {
      ChiselStage.emitCHIRRTL(new Module {
        val io = IO(new Bundle {})
        dontTouch(new Bundle { val a = UInt(32.W) })
      })
    }
  }
  it should "not work on literals" in {
    val e = the[chisel3.ExpectedAnnotatableException] thrownBy {
      ChiselStage.emitCHIRRTL(new Module {
        dontTouch(123.U)
      })
    }
    e.getMessage should include("must not be a literal")
  }

  "fields" should "be marked don't touch by default" in {
    generateFirrtlAndFileCheck(
      new HasDeadCodeLeaves(),
      "--implicit-check-not",
      """"target":"~HasDeadCodeLeaves|HasDeadCodeChildLeaves>io.a""""
    )(
      """|CHECK:      "class":"firrtl.transforms.DontTouchAnnotation",
         |CHECK-NEXT: "target":"~HasDeadCodeLeaves|HasDeadCodeChildLeaves>io.a.a2"
         |CHECK:      "class":"firrtl.transforms.DontTouchAnnotation",
         |CHECK-NEXT: "target":"~HasDeadCodeLeaves|HasDeadCodeChildLeaves>io.a.a1"
         |""".stripMargin
    )
  }

  "probes and properties" should "NOT be marked dontTouch" in {
    generateFirrtlAndFileCheck(
      new HasProbesAndProperties(),
      "--implicit-check-not",
      """"target":"~HasProbesAndProperties|HasProbesAndProperties>io.probe"""",
      "--implicit-check-not",
      """"target":"~HasProbesAndProperties|HasProbesAndProperties>io.prop""""
    )(
      """|CHECK:      "class":"firrtl.transforms.DontTouchAnnotation",
         |CHECK-NEXT: "target":"~HasProbesAndProperties|HasProbesAndProperties>io.a"
         |""".stripMargin
    )

    // Ensure can compile the result.
    ChiselStage.emitSystemVerilog(new HasProbesAndProperties())
  }
}
