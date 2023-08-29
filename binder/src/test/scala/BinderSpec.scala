// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.random

import chisel3._
import chisel3.util._
import chisel3.experimental._
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

  def firrtlString(module: => RawModule): String = Seq(
    new chisel3.stage.phases.Elaborate,
    chisel3.internal.panama.Convert
  ).foldLeft(
    firrtl.AnnotationSeq(Seq(chisel3.stage.ChiselGeneratorAnnotation(() => module)))
  ) { case (annos, phase) => phase.transform(annos) }
    .collectFirst {
      case PanamaCIRCTConverterAnnotation(converter) =>
        val string = new java.io.ByteArrayOutputStream
        converter.firrtlStream.writeBytesTo(string)
        new String(string.toByteArray)
    }
    .get

  behavior.of("binder")

  it should "generate RTL with circt binder" in {
    firrtlString(new EmptyModule) should equal(
      """|circuit EmptyModule :
         |  module EmptyModule :
         |""".stripMargin)

    firrtlString(new Blinky(1000)) should equal(
      """|circuit Blinky :
         |  module Blinky :
         |    input clock : Clock
         |    input reset : UInt<1>
         |    output io : { led0 : UInt<1> }
         |
         |    reg led : UInt<1>, clock with :
         |      reset => (reset, UInt<1>(0)) @[binder/src/test/scala/BinderSpec.scala 19:20]
         |    reg counterWrap_c_value : UInt<9>, clock with :
         |      reset => (reset, UInt<9>(0)) @[src/main/scala/chisel3/util/Counter.scala 61:40]
         |    wire counterWrap : UInt<1> @[src/main/scala/chisel3/util/Counter.scala 117:24]
         |    counterWrap <= UInt<1>(0) @[src/main/scala/chisel3/util/Counter.scala 117:24]
         |    when UInt<1>(1) : @[src/main/scala/chisel3/util/Counter.scala 118:16]
         |      node counterWrap_wrap_wrap = eq(counterWrap_c_value, UInt<9>(499)) @[src/main/scala/chisel3/util/Counter.scala 73:24]
         |      node _counterWrap_wrap_value_T = add(counterWrap_c_value, UInt<1>(1)) @[src/main/scala/chisel3/util/Counter.scala 77:24]
         |      node _counterWrap_wrap_value_T_1 = tail(_counterWrap_wrap_value_T, 1) @[src/main/scala/chisel3/util/Counter.scala 77:24]
         |      counterWrap_c_value <= _counterWrap_wrap_value_T_1 @[src/main/scala/chisel3/util/Counter.scala 77:15]
         |      when counterWrap_wrap_wrap : @[src/main/scala/chisel3/util/Counter.scala 87:20]
         |        counterWrap_c_value <= UInt<1>(0) @[src/main/scala/chisel3/util/Counter.scala 87:28]
         |      else :
         |      counterWrap <= counterWrap_wrap_wrap @[src/main/scala/chisel3/util/Counter.scala 118:23]
         |    else :
         |    when counterWrap : @[binder/src/test/scala/BinderSpec.scala 21:21]
         |      node _led_T = not(led) @[binder/src/test/scala/BinderSpec.scala 22:12]
         |      led <= _led_T @[binder/src/test/scala/BinderSpec.scala 22:9]
         |    else :
         |    io.led0 <= led @[binder/src/test/scala/BinderSpec.scala 24:11]
         |""".stripMargin)
  }
}
