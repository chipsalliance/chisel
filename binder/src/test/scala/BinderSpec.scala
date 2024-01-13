// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.random

import java.io.OutputStream
import geny.Writable
import scala.collection.immutable.SeqMap
import chisel3._
import chisel3.probe._
import chisel3.util._
import chisel3.util.experimental._
import chisel3.experimental._
import chisel3.internal.CIRCTConverter
import chisel3.internal.panama.circt._
import chisel3.properties._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class EmptyModule extends RawModule

class Blinky(freq: Int, startOn: Boolean = false) extends Module {
  val io = IO(new Bundle {
    val led0 = Output(Bool())
  })
  // Blink LED every second using Chisel built-in util.Counter
  val led = RegInit(startOn.B)
  val (_, counterWrap) = Counter(true.B, freq / 2)
  when(counterWrap) {
    led := ~led
  }
  io.led0 := led
}

class TruncationTest extends RawModule {
  val dest = IO(Output(UInt(8.W)))
  val src = IO(Input(UInt(16.W)))
  dest := src

  val vDest = IO(Output(Vec(4, UInt(1.W))))
  vDest := VecInit(Seq.fill(4)(1.U(2.W)))
}

// https://github.com/chipsalliance/chisel/issues/3548#issuecomment-1734346659
class BitLengthOfNeg1Test extends RawModule {
  val a = IO(Output(UInt()))
  a := -1.S.asUInt
}

class PropertyTest extends RawModule {
  val i = IO(Input(UInt(8.W)))

  val p = IO(Output(Property[Path]()))
  p := Property(Path(i))

  val a = IO(Output(Property[Seq[Seq[Property[Int]]]]()))
  val b = IO(Output(Property[Seq[Property[Seq[Int]]]]()))
  a := Property(Seq[Seq[Int]](Seq(123)))
  b := Property(Seq[Seq[Int]](Seq(123)))
}

class ProbeSimpleTest extends Module {
  val a = IO(Output(RWProbe(Bool())))

  val w = WireInit(Bool(), false.B)
  val w_probe = RWProbeValue(w)
  define(a, w_probe)

  forceInitial(a, false.B)
  releaseInitial(a)

  force(a, false.B)
  release(a)
}

class BoreBar extends RawModule {
  val a = Wire(Bool())
}
class BoreBaz(_a: Bool) extends RawModule {
  val b = Wire(Bool())
  b := BoringUtils.tapAndRead(_a)
}
class BoreTop extends RawModule {
  val bar = Module(new BoreBar)
  val baz = Module(new BoreBaz(bar.a))
}

class ProbeRead extends RawModule {
  val clockP = IO(RWProbe(Bool()))
  val resetP = IO(RWProbe(Bool()))
  val clock = IO(Output(Clock()))
  val reset = IO(Output(Bool()))

  val genClock = read(clockP)
  clock := genClock.asClock
  reset := read(resetP)
}

class DontCareIOTest extends Module {
  val io = IO(new Bundle {
    val in = Flipped(DecoupledIO(UInt(8.W)))
  })
  io.in := DontCare
}

class ZeroWidthTest extends Module {
  val a = IO(Input(UInt(32.W)))
  val b = IO(Input(SInt(32.W)))
  val c = IO(Output(UInt(33.W)))
  c := Cat(a >> 32, b)
}

class BinderTest extends AnyFlatSpec with Matchers {

  def streamString(module: => RawModule, stream: CIRCTConverter => Writable): String = Seq(
    new chisel3.stage.phases.Elaborate,
    chisel3.internal.panama.Convert
  ).foldLeft(
    firrtl.AnnotationSeq(Seq(chisel3.stage.ChiselGeneratorAnnotation(() => module)))
  ) { case (annos, phase) => phase.transform(annos) }
    .collectFirst {
      case PanamaCIRCTConverterAnnotation(converter) =>
        val string = new java.io.ByteArrayOutputStream
        stream(converter).writeBytesTo(string)
        new String(string.toByteArray)
    }
    .get

  def mlirString(module:    => RawModule): String = streamString(module, _.mlirStream)
  def firrtlString(module:  => RawModule): String = streamString(module, _.firrtlStream)
  def verilogString(module: => RawModule): String = streamString(module, _.verilogStream)

  behavior.of("binder")

  it should "generate RTL with circt binder" in {
    firrtlString(new EmptyModule) should
      (include("FIRRTL version")
        .and(include("circuit EmptyModule :"))
        .and(include("module EmptyModule :")))

    firrtlString(new Blinky(1000)) should
      (include("circuit Blinky")
        .and(include("module Blinky"))
        .and(include("input clock : Clock"))
        .and(include("input reset : UInt<1>"))
        .and(include("output io : { led0 : UInt<1> }"))
        .and(include("when counterWrap"))
        .and(include("connect led, _led_T")))

    verilogString(new Blinky(1000)) should
      (include("module Blinky")
        .and(include("input  clock,"))
        .and(include("       reset,"))
        .and(include("output io_led0"))
        .and(include("if (counterWrap)"))
        .and(include("counterWrap_c_value <=")))

    verilogString(new TruncationTest) should include("assign dest = src[7:0]")
      .and(include("assign vDest_0 = 1'h1"))
      .and(include("assign vDest_1 = 1'h1"))
      .and(include("assign vDest_2 = 1'h1"))
      .and(include("assign vDest_3 = 1'h1"))

    firrtlString(new BitLengthOfNeg1Test) should include("asUInt(SInt<1>(-1))")

    streamString(
      new PropertyTest,
      (converter: CIRCTConverter) => {
        val pm = converter.passManager()
        assert(pm.populatePreprocessTransforms())
        assert(pm.populateCHIRRTLToLowFIRRTL())
        assert(pm.populateLowFIRRTLToHW())
        assert(pm.populateFinalizeIR())
        assert(pm.run())

        val om = converter.om()
        val evaluator = om.evaluator()
        val obj = evaluator.instantiate("PropertyTest_Class", Seq(om.newBasePathEmpty))
        val path = obj.field("p").asInstanceOf[PanamaCIRCTOMEvaluatorValuePath].asString

        assert(path == "OMReferenceTarget:~PropertyTest|PropertyTest>i")

        converter.mlirStream
      }
    ) should include("om.class.field @p, %1 : !om.frozenpath")
      .and(include("om.class.field @a, %3 : !om.list<!om.list<!om.integer>>"))
      .and(include("om.class.field @b, %3 : !om.list<!om.list<!om.integer>>"))

    firrtlString(new ProbeSimpleTest) should include("define a = rwprobe(w)")
      .and(include("force_initial(a, UInt<1>(0))"))
      .and(include("release_initial(a)"))
      .and(include("force(clock, _T, a, UInt<1>(0))"))
      .and(include("release(clock, _T_1, a)"))

    firrtlString(new BoreTop) should include("output b_bore")
      .and(include("define b_bore = probe(a)"))
      .and(include("connect baz.b_bore, read(bar.b_bore)"))
    firrtlString(new ProbeRead) should include("asClock(read(clockP))")

    verilogString(new DontCareIOTest) should include("assign io_in_ready = 1'h0")
    mlirString(new ZeroWidthTest) should include(
      "firrtl.cat %_c_T, %c_lo : (!firrtl.uint<1>, !firrtl.uint<32>) -> !firrtl.uint<33>"
    )
  }
}
