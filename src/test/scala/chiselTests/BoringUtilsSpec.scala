// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.util.Counter
import chisel3.testers._
import chisel3.experimental.{BaseModule, ChiselAnnotation}
import chisel3.probe._
import chisel3.util.experimental.BoringUtils
import firrtl.annotations.Annotation
import firrtl.transforms.DontTouchAnnotation

abstract class ShouldntAssertTester(cyclesToWait: BigInt = 4) extends BasicTester {
  val dut: BaseModule
  val (_, done) = Counter(true.B, 2)
  when(done) { stop() }
}

class BoringUtilsSpec extends ChiselFlatSpec with ChiselRunners with Utils with MatchesAndOmits {
  val args = Array("--throw-on-first-error", "--full-stacktrace")

  class BoringInverter extends Module {
    val io = IO(new Bundle {})
    val a = Wire(UInt(1.W))
    val notA = Wire(UInt(1.W))
    val b = Wire(UInt(1.W))
    a := 0.U
    notA := ~a
    b := a
    chisel3.assert(b === 1.U)
    BoringUtils.addSource(notA, "x")
    BoringUtils.addSink(b, "x")
  }

  behavior.of("BoringUtils.{addSink, addSource}")

  it should "connect two wires within a module" in {
    runTester(
      new ShouldntAssertTester { val dut = Module(new BoringInverter) },
      annotations = TesterDriver.verilatorOnly
    ) should be(true)
  }

  trait WireX { this: BaseModule =>
    val x = Wire(UInt(4.W))
    chisel3.experimental.annotate(new ChiselAnnotation {
      def toFirrtl: Annotation = DontTouchAnnotation(x.toNamed)
    })
  }

  class Source extends RawModule with WireX {
    val in = IO(Input(UInt()))
    x := in
  }

  class Sink extends RawModule with WireX {
    val out = IO(Output(UInt()))
    x := 0.U // Default value. Output is zero unless we bore...
    out := x
  }

  class Top(val width: Int) extends Module {
    /* From the perspective of deduplication, all sources are identical and all sinks are identical. */
    val sources = Seq.fill(3)(Module(new Source))
    val sinks = Seq.fill(6)(Module(new Sink))

    /* Sources are differentiated by their input connections only. */
    sources.zip(Seq(0, 1, 2)).map { case (a, b) => a.in := b.U }

    /* Sinks are differentiated by their post-boring outputs. */
    sinks.zip(Seq(0, 1, 1, 2, 2, 2)).map { case (a, b) => chisel3.assert(a.out === b.U) }
  }

  /** This is testing a complicated wiring pattern and exercising
    * the necessity of disabling deduplication for sources and sinks.
    * Without disabling deduplication, this test will fail.
    */
  class TopTester extends ShouldntAssertTester {
    val dut = Module(new Top(4))
    BoringUtils.bore(dut.sources(1).x, Seq(dut.sinks(1).x, dut.sinks(2).x))
    BoringUtils.bore(dut.sources(2).x, Seq(dut.sinks(3).x, dut.sinks(4).x, dut.sinks(5).x))
  }

  class TopTesterFail extends ShouldntAssertTester {
    val dut = Module(new Top(4))
    BoringUtils.addSource(dut.sources(1).x, "foo", disableDedup = true)
    BoringUtils.addSink(dut.sinks(1).x, "foo", disableDedup = true)
    BoringUtils.addSink(dut.sinks(2).x, "foo", disableDedup = true)

    BoringUtils.addSource(dut.sources(2).x, "bar", disableDedup = true)
    BoringUtils.addSink(dut.sinks(3).x, "bar", disableDedup = true)
    BoringUtils.addSink(dut.sinks(4).x, "bar", disableDedup = true)
    BoringUtils.addSink(dut.sinks(5).x, "bar", disableDedup = true)
  }

  behavior.of("BoringUtils.bore")

  it should "connect across modules using BoringUtils.bore" in {
    runTester(new TopTester, annotations = TesterDriver.verilatorOnly) should be(true)
  }

  // TODO: this test is not really testing anything as MFC does boring during
  // LowerAnnotations (which happens right after parsing).  Consider reworking
  // this into a test that uses D/I (or some other mechanism of having a
  // pre-deduplicated circuit).  This is likely better handled as a test in
  // CIRCT than in Chisel.
  it should "still work even with dedup off" in {
    runTester(new TopTesterFail, annotations = Seq(TesterDriver.VerilatorBackend))
  }

  class InternalBore extends RawModule {
    val in = IO(Input(Bool()))
    val out = IO(Output(Bool()))
    out := false.B
    BoringUtils.bore(in, Seq(out))
  }

  class InternalBoreTester extends ShouldntAssertTester {
    val dut = Module(new InternalBore)
    dut.in := true.B
    chisel3.assert(dut.out === true.B)
  }

  it should "work for an internal (same module) BoringUtils.bore" in {
    runTester(new InternalBoreTester, annotations = TesterDriver.verilatorOnly) should be(true)
  }

  it should "work using new API" in {
    class Baz extends RawModule {
      val a_wire = WireInit(UInt(1.W), DontCare)
      dontTouch(a_wire)
    }
    class Bar extends RawModule {
      val b_wire = WireInit(UInt(2.W), DontCare)
      dontTouch(b_wire)

      val baz = Module(new Baz)
    }
    class Foo extends RawModule {
      val a = IO(Output(UInt()))
      val b = IO(Output(UInt()))
      val c = IO(Output(UInt()))

      val c_wire = WireInit(UInt(3.W), DontCare)
      dontTouch(c_wire)

      val bar = Module(new Bar)

      a := BoringUtils.bore(bar.baz.a_wire)
      b := BoringUtils.bore(bar.b_wire)
      c := BoringUtils.bore(c_wire)
    }
    matchesAndOmits(circt.stage.ChiselStage.emitCHIRRTL(new Foo))(
      "module Baz :",
      "output a_bore : UInt<1>",
      "connect a_bore, a_wire",
      "module Bar :",
      "output b_bore : UInt<2>",
      "connect a_bore, baz.a_bore",
      "connect b_bore, b_wire",
      "module Foo :",
      "connect a, a_bore",
      "connect b, b_bore",
      "connect c, c_wire",
      "connect a_bore, bar.a_bore",
      "connect b_bore, bar.b_bore"
    )()
  }

  it should "bore up and down through the lowest common ancestor" in {
    class Bar extends RawModule {
      val a = Wire(Bool())
    }

    class Baz(_a: Bool) extends RawModule {
      val b = WireInit(Bool(), BoringUtils.bore(_a))
    }

    class Foo extends RawModule {
      val bar = Module(new Bar)
      val baz = Module(new Baz(bar.a))
    }

    matchesAndOmits(circt.stage.ChiselStage.emitCHIRRTL(new Foo))(
      "module Bar :",
      "output b_bore : UInt<1>",
      "connect b_bore, a",
      "module Baz :",
      "input b_bore : UInt<1>",
      "wire b_bore_1 : UInt<1>",
      "connect b_bore_1, b_bore",
      "connect b, b_bore_1",
      "module Foo",
      "connect baz.b_bore, bar.b_bore"
    )()
  }

  it should "not work over a Definition/Instance boundary" in {
    import chisel3.experimental.hierarchy._
    @instantiable
    class Bar extends RawModule {
      @public val a_wire = WireInit(UInt(1.W), DontCare)
    }
    class Foo extends RawModule {
      val bar = Instance(Definition((new Bar)))
      BoringUtils.bore(bar.a_wire)
    }
    val e = intercept[Exception] {
      circt.stage.ChiselStage.emitCHIRRTL(new Foo, args)
    }
    e.getMessage should include("Cannot bore across a Definition/Instance boundary")
  }

  it should "work boring upwards" in {
    import chisel3.experimental.hierarchy._
    class Bar(parentData: Data) extends RawModule {
      val q = Wire(UInt(1.W))
      q := BoringUtils.bore(parentData)
    }
    class Foo extends RawModule {
      val a = IO(Input(UInt(1.W)))
      val bar = Module(new Bar(a))
    }
    matchesAndOmits(circt.stage.ChiselStage.emitCHIRRTL(new Foo))(
      "module Bar :",
      "input q_bore : UInt<1>",
      "connect q, q_bore_1", // Do normal connection before secret ones
      "connect q_bore_1, q_bore",
      "module Foo :",
      "input a : UInt<1>",
      "connect bar.q_bore, a"
    )()
  }

  it should "be included in DataMirror.modulePorts" in {
    import chisel3.reflect.DataMirror
    class Bar extends RawModule {
      val a_wire = WireInit(UInt(1.W), DontCare)
      dontTouch(a_wire)
    }
    class Foo extends RawModule {
      val a = IO(Output(UInt()))
      val bar = Module(new Bar)
      //val preBore = DataMirror.modulePorts(bar)
      a := BoringUtils.bore(bar.a_wire)
      val postBore = DataMirror.modulePorts(bar)
      postBore.size should be(1)
    }
    circt.stage.ChiselStage.emitCHIRRTL(new Foo, args)
  }
  it should "fail if bore after calling DataMirror.modulePorts" in {
    import chisel3.reflect.DataMirror
    class Bar extends RawModule {
      val a_wire = WireInit(UInt(1.W), DontCare)
      dontTouch(a_wire)
    }
    class Foo extends RawModule {
      val a = IO(Output(UInt()))
      val bar = Module(new Bar)
      val preBore = DataMirror.modulePorts(bar)
      a := BoringUtils.bore(bar.a_wire)
    }
    val (log, res) = grabLog(
      intercept[Exception] {
        circt.stage.ChiselStage.emitCHIRRTL(new Foo)
      }
    )
    log should include("Reflecting on all io's fully closes Bar, but it is later bored through!")
    log should include("Can only bore into modules that are not fully closed")
  }
  it should "be ok if with .toDefinition, if it is after boring" in {
    import chisel3.reflect.DataMirror
    class Bar extends RawModule {
      val a_wire = WireInit(UInt(1.W), DontCare)
      dontTouch(a_wire)
    }
    class Foo extends RawModule {
      val a = IO(Output(UInt()))
      val bar = Module(new Bar)
      a := BoringUtils.bore(bar.a_wire)
      bar.toDefinition
    }

  }
  it should "error if ever bored after calling .toDefinition" in {
    import chisel3.reflect.DataMirror
    class Bar extends RawModule {
      val a_wire = WireInit(UInt(1.W), DontCare)
    }
    class Foo extends RawModule {
      val bar = Module(new Bar)
      bar.toDefinition
      BoringUtils.bore(bar.a_wire)
    }

    val (log, res) = grabLog(
      intercept[Exception] {
        circt.stage.ChiselStage.emitCHIRRTL(new Foo)
      }
    )
    log should include("Calling .toDefinition fully closes Bar, but it is later bored through!")
    log should include("Can only bore into modules that are not fully closed")
  }

  it should "not create a new port when source is a port" in {
    class Baz extends RawModule {
      val a = IO(Output(Bool()))
    }

    class Bar extends RawModule {
      val baz = Module(new Baz)
    }

    class Foo extends RawModule {
      val a = IO(Output(Bool()))

      val bar = Module(new Bar)

      a := BoringUtils.bore(bar.baz.a)
    }

    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Foo)
    matchesAndOmits(chirrtl)()(
      "connect a_bore, a"
    )
  }

  it should "bore from a Probe" in {
    class Baz extends RawModule {
      val a = IO(Probe(Bool()))
      define(a, ProbeValue(false.B))
    }

    class Bar extends RawModule {
      val baz = Module(new Baz)
    }

    class Foo extends RawModule {
      val a = IO(Output(Bool()))

      val bar = Module(new Bar)

      a := read(BoringUtils.bore(bar.baz.a))
    }

    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Foo)
    matchesAndOmits(chirrtl)()(
      "connect a_bore, a"
    )
  }
}
