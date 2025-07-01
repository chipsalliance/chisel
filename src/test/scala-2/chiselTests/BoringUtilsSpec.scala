// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.util.Counter
import chisel3.experimental.{BaseModule, OpaqueType}
import chisel3.probe._
import chisel3.properties.Property
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.testing.scalatest.FileCheck
import chisel3.util.experimental.BoringUtils
import circt.stage.ChiselStage
import firrtl.annotations.Annotation
import firrtl.transforms.DontTouchAnnotation
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

abstract class ShouldntAssertTester(cyclesToWait: BigInt = 4) extends Module {
  val dut: BaseModule
  val (_, done) = Counter(true.B, 2)
  when(done) { stop() }
}

class BoringUtilsSpec extends AnyFlatSpec with Matchers with LogUtils with FileCheck with ChiselSim {
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

  behavior.of("BoringUtils.addSink and BoringUtils.addSource")

  it should "connect two wires within a module" in {
    simulate(new ShouldntAssertTester { val dut = Module(new BoringInverter) })(RunUntilFinished(3))
  }

  trait WireX { this: BaseModule =>
    val x = dontTouch(Wire(UInt(4.W)))
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
    simulate(new TopTester)(RunUntilFinished(3))
  }

  // TODO: this test is not really testing anything as MFC does boring during
  // LowerAnnotations (which happens right after parsing).  Consider reworking
  // this into a test that uses D/I (or some other mechanism of having a
  // pre-deduplicated circuit).  This is likely better handled as a test in
  // CIRCT than in Chisel.
  it should "still work even with dedup off" in {
    simulate(new TopTesterFail)(RunUntilFinished(3))
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

  it should "work for an internal, same module, BoringUtils.bore" in {
    simulate(new InternalBoreTester)(RunUntilFinished(3))
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
    ChiselStage
      .emitCHIRRTL(new Foo)
      .fileCheck()(
        """|CHECK-LABEL: module Baz :
           |CHECK:         output a_bore : UInt<1>
           |CHECK:         connect a_bore, a_wire
           |CHECK-LABEL: module Bar :
           |CHECK:         output b_bore : UInt<2>
           |CHECK:         connect a_bore, baz.a_bore
           |CHECK:         connect b_bore, b_wire
           |CHECK-LABEL: module Foo :
           |CHECK:         connect a, a_bore
           |CHECK:         connect b, b_bore
           |CHECK:         connect c, c_wire
           |CHECK:         connect a_bore, bar.a_bore
           |CHECK:         connect b_bore, bar.b_bore
           |""".stripMargin
      )
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

    ChiselStage
      .emitCHIRRTL(new Foo)
      .fileCheck()(
        """|CHECK-LABEL: module Bar :
           |CHECK:         output b_bore : UInt<1>
           |CHECK:         connect b_bore, a
           |CHECK-LABEL: module Baz :
           |CHECK:         input b_bore : UInt<1>
           |CHECK:         wire b_bore_1 : UInt<1>
           |CHECK:         connect b, b_bore_1
           |CHECK:         connect b_bore_1, b_bore
           |CHECK-LABEL: module Foo
           |CHECK:         connect baz.b_bore, bar.b_bore
           |""".stripMargin
      )
  }

  it should "not create input probes" in {

    object A extends layer.Layer(layer.LayerConfig.Extract())

    class Bar extends RawModule {

      val a_probe = IO(Output(Probe(Bool(), A)))

      layer.block(A) {
        val a = dontTouch(WireInit(false.B))
        define(a_probe, ProbeValue(a))
      }

    }

    class Foo extends RawModule {

      val bar = Module(new Bar)

      layer.block(A) {
        class Baz extends RawModule {
          val b = dontTouch(WireInit(BoringUtils.tapAndRead(bar.a_probe)))
        }
        val baz = Module(new Baz)
      }

    }

    val firrtl = circt.stage.ChiselStage.emitCHIRRTL(new Foo)
    firrtl should include("input b_bore : UInt<1>")
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

  it should "work if boring from an Instance's output port" in {
    import chisel3.experimental.hierarchy._
    @instantiable
    class Bar extends RawModule {
      @public val out = IO(Output(UInt(1.W)))
      out := DontCare
    }
    class Foo extends RawModule {
      val bar = Instance(Definition((new Bar)))
      val sink = BoringUtils.bore(bar.out)
    }
    ChiselStage
      .emitCHIRRTL(new Foo)
      .fileCheck()(
        """|CHECK-LABEL: module Bar :
           |CHECK:         output out : UInt<1>
           |CHECK-LABEL: module Foo :
           |CHECK:         connect sink, bar.out
           |""".stripMargin
      )
  }

  it should "work if driving an Instance's input port" in {
    import chisel3.experimental.hierarchy._
    @instantiable
    class Bar extends RawModule {
      @public val in = IO(Input(UInt(1.W)))
    }
    class Foo extends RawModule {
      val bar = Instance(Definition((new Bar)))
      val source = BoringUtils.drive(bar.in)
    }
    ChiselStage
      .emitCHIRRTL(new Foo)
      .fileCheck()(
        """|CHECK-LABEL: module Bar :
           |CHECK:         input in : UInt<1>
           |CHECK-LABEL: module Foo :
           |CHECK:         connect bar.in, source
           |""".stripMargin
      )
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
    ChiselStage
      .emitCHIRRTL(new Foo)
      .fileCheck()(
        """|CHECK-LABEL: module Bar :
           |CHECK:         input q_bore : UInt<1>
           |CHECK:         connect q, q_bore_1
           |CHECK:         connect q_bore_1, q_bore
           |CHECK-LABEL: module Foo :
           |CHECK:         input a : UInt<1>
           |CHECK:         connect bar.q_bore, a
           |""".stripMargin
      )
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
      // val preBore = DataMirror.modulePorts(bar)
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

  it should "support boring on a Module even after .toInstance (and accessing a port)" in {
    import chisel3.experimental.hierarchy._
    @instantiable
    class Bar extends RawModule {
      @public val port = IO(Output(UInt(8.W)))
      val a_wire = WireInit(UInt(1.W), DontCare)
    }
    class Foo extends RawModule {
      val bar = Module(new Bar)
      val bi = bar.toInstance
      val x = BoringUtils.bore(bar.a_wire)
      val p = bi.port // Previously, the lookup here would close the module due to reflecting on the IOs of bar
      val y = BoringUtils.bore(bar.a_wire)
    }

    ChiselStage
      .emitCHIRRTL(new Foo)
      .fileCheck()(
        """|CHECK-LABEL: module Bar :
           |CHECK:         output port : UInt<8>
           |CHECK:         output x_bore : UInt<1>
           |CHECK:         output y_bore : UInt<1>
           |CHECK:         connect x_bore, a_wire
           |CHECK:         connect y_bore, a_wire
           |CHECK-LABEL: module Foo :
           |CHECK:         connect x, bar.x_bore
           |CHECK:         connect y, bar.y_bore
           |""".stripMargin
      )
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

    (circt.stage.ChiselStage.emitCHIRRTL(new Foo) should not).include("connect a_bore, a")
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

    (circt.stage.ChiselStage.emitCHIRRTL(new Foo) should not).include("connect a_bore, a")
  }

  it should "bore from a Property" in {
    class Baz extends RawModule {
      val a = IO(Output(Property[Int]()))
    }

    class Bar extends RawModule {
      val baz = Module(new Baz)
    }

    class Foo extends RawModule {
      val a = IO(Output(Property[Int]()))

      val bar = Module(new Bar)

      a := BoringUtils.bore(bar.baz.a)
    }

    ChiselStage
      .emitCHIRRTL(new Foo)
      .fileCheck()(
        """|CHECK-LABEL: module Bar :
           |CHECK:         output a_bore : Integer
           |CHECK:         propassign a_bore, baz.a
           |CHECK-LABEL: public module Foo :
           |CHECK:         propassign a, bar.a_bore
           |""".stripMargin
      )
  }

  it should "bore from an opaque type that wraps a Property" in {
    class MyOpaqueProperty extends Record with OpaqueType {
      private val underlying = Property[Int]()
      val elements = scala.collection.immutable.SeqMap("" -> underlying)
      override protected def errorOnAsUInt = true
    }

    class Baz extends RawModule {
      val a = IO(Output(new MyOpaqueProperty))
    }

    class Bar extends RawModule {
      val baz = Module(new Baz)
    }

    class Foo extends RawModule {
      val a = IO(Output(new MyOpaqueProperty))

      val bar = Module(new Bar)

      a := BoringUtils.bore(bar.baz.a)
    }

    ChiselStage
      .emitCHIRRTL(new Foo)
      .fileCheck()(
        """|CHECK-LABEL: module Bar :
           |CHECK:         output a_bore : Integer
           |CHECK:         propassign a_bore, baz.a
           |CHECK-LABEL: public module Foo :
           |CHECK:         propassign a_bore, bar.a_bore
           |""".stripMargin
      )

  }

  it should "bore from nested opaque types that wrap a Property" in {
    class MyOpaqueProperty extends Record with OpaqueType {
      private val underlying = Property[Int]()
      val elements = scala.collection.immutable.SeqMap("" -> underlying)
      override protected def errorOnAsUInt = true
    }

    class MyOuterOpaque extends Record with OpaqueType {
      private val underlying = new MyOpaqueProperty
      val elements = scala.collection.immutable.SeqMap("" -> underlying)
      override protected def errorOnAsUInt = true
    }

    class Baz extends RawModule {
      val a = IO(Output(new MyOpaqueProperty))
    }

    class Bar extends RawModule {
      val baz = Module(new Baz)
    }

    class Foo extends RawModule {
      val a = IO(Output(new MyOpaqueProperty))

      val bar = Module(new Bar)

      a := BoringUtils.bore(bar.baz.a)
    }

    ChiselStage
      .emitCHIRRTL(new Foo)
      .fileCheck()(
        """|CHECK-LABEL: module Bar :
           |CHECK:         output a_bore : Integer
           |CHECK:         propassign a_bore, baz.a
           |CHECK-LABEL: public module Foo :
           |CHECK:         propassign a_bore, bar.a_bore
           |""".stripMargin
      )
  }

  behavior.of("BoringUtils.drive")

  it should "fail on probes" in {
    class Foo extends RawModule {
      val a = Wire(Bool())
      val p = ProbeValue(a)
    }

    class Bar extends RawModule {
      val foo = Module(new Foo)

      BoringUtils.drive(foo.p) := 1.B
    }

    val e = the[Exception] thrownBy circt.stage.ChiselStage.emitCHIRRTL(new Bar)

    e.getMessage should include("requirement failed: cannot drive a probe from BoringUtils.drive")
  }

  it should "bore ports for driving hardware" in {
    class Foo extends RawModule {
      val a = Wire(Bool())
    }

    class Bar extends RawModule {
      val foo = Module(new Foo)

      BoringUtils.drive(foo.a) := 1.B
    }

    ChiselStage
      .emitCHIRRTL(new Bar)
      .fileCheck()(
        """|CHECK-LABEL: module Foo :
           |CHECK:         input bore
           |CHECK:         connect a, bore
           |CHECK-LABEL: module Bar :
           |CHECK:         wire bore
           |CHECK:         connect bore, UInt<1>(0h1)
           |CHECK:         connect foo.bore, bore
           |""".stripMargin
      )

  }

  it should "bore ports for driving properties" in {
    class Foo extends RawModule {
      val a = Wire(Property[Int]())
    }

    class Bar extends RawModule {
      val foo = Module(new Foo)

      BoringUtils.drive(foo.a) := Property(1)
    }

    ChiselStage
      .emitCHIRRTL(new Bar)
      .fileCheck()(
        """|CHECK-LABEL: module Foo :
           |CHECK:         input bore :
           |CHECK:         propassign a, bore
           |CHECK-LABEL: public module Bar :
           |CHECK:         propassign foo.bore, Integer(1)
           |""".stripMargin
      )
  }

  it should "bore to the final instance, but not into it, for inputs" in {
    class Foo extends RawModule {
      val a = IO(Input(Property[Int]()))
    }

    class Bar extends RawModule {
      val foo = Module(new Foo)
    }

    class Baz extends RawModule {
      val bar = Module(new Bar)

      BoringUtils.drive(bar.foo.a) := Property(1)
    }

    ChiselStage
      .emitCHIRRTL(new Baz)
      .fileCheck()(
        """|CHECK-LABEL: module Bar :
           |CHECK:         input bore :
           |CHECK:         propassign foo.a, bore
           |CHECK-LABEL: public module Baz :
           |CHECK:         propassign bar.bore, Integer(1)
           |""".stripMargin
      )
  }

  it should "bore into the final instance for outputs" in {
    class Foo extends RawModule {
      val a = IO(Output(Property[Int]()))
    }

    class Bar extends RawModule {
      val foo = Module(new Foo)
    }

    class Baz extends RawModule {
      val bar = Module(new Bar)

      BoringUtils.drive(bar.foo.a) := Property(1)
    }

    ChiselStage
      .emitCHIRRTL(new Baz)
      .fileCheck()(
        """|CHECK-LABEL: module Foo :
           |CHECK:         input bore :
           |CHECK:         propassign a, bore
           |CHECk-LABEL: module Bar :
           |CHECK:         input bore :
           |CHECK:         propassign foo.bore, bore
           |CHECK-LABEL: public module Baz :
           |CHECK:         propassign bar.bore, Integer(1)
           |""".stripMargin
      )
  }
  it should "fail if endIOCreation is set" in {

    class Bar extends RawModule {
      val baz = Module(new Baz)
      BoringUtils.bore(baz.c)
    }

    class Baz extends RawModule {
      val c = Wire(Property[Int]())
      endIOCreation()
    }

    val e = intercept[Exception] {
      circt.stage.ChiselStage.emitCHIRRTL(new Bar, args)
    }
    e.getMessage should include("Cannot bore or tap into Baz (from Bar) if IO creation is not allowed")
  }
}
