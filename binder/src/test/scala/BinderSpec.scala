// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.random

import geny.Writable
import chisel3._
import chisel3.util._
import chisel3.experimental._
import chisel3.internal.CIRCTConverter
import chisel3.internal.panama.circt.PanamaCIRCTConverterAnnotation
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

class BinderTest extends AnyFlatSpec with Matchers {

  def StreamString(module: => RawModule, stream: CIRCTConverter => Writable): String = Seq(
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

  def firrtlString(module:  => RawModule): String = StreamString(module, _.firrtlStream)
  def verilogString(module: => RawModule): String = StreamString(module, _.verilogStream)

  behavior.of("binder")

  it should "generate RTL with circt binder" in {
    firrtlString(new EmptyModule) should equal(
      """|FIRRTL version 3.1.0
         |circuit EmptyModule :
         |  module EmptyModule :
         |""".stripMargin)

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
  }
}
