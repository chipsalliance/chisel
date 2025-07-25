// SPDX-License-Identifier: Apache-2.0

package chiselTests.util

import _root_.circt.stage.ChiselStage
import chisel3._
import chisel3.experimental.noPrefix
import chisel3.util.Cat
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

object CatSpec {

  class JackIsATypeSystemGod extends Module {
    val in = IO(Input(Vec(0, UInt(8.W))))
    val out = IO(Output(UInt(8.W)))

    out := Cat(in)
  }

}

class CatSpec extends AnyFlatSpec with Matchers {

  import CatSpec._

  behavior.of("chisel3.util.Cat")

  it should "not fail to elaborate a zero-element Vec" in {

    ChiselStage.emitCHIRRTL(new JackIsATypeSystemGod)

  }

  it should "not override the names of its arguments" in {
    class MyModule extends RawModule {
      val a, b, c, d = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt()))

      out := Cat(a, b, c, d)
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    for (name <- Seq("a", "b", "c", "d")) {
      chirrtl should include(s"input $name : UInt<8>")
    }
  }

  it should "emit a single expression for the entire cat" in {
    class MyModule extends RawModule {
      val in = IO(Input(Vec(8, UInt(8.W))))
      val out = IO(Output(UInt(64.W)))

      out := Cat(in)
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("cat(in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7])")
  }

  it should "have a source locator when passing a seq" in {
    class MyModule extends RawModule {
      val in = IO(Input(Vec(8, UInt(8.W))))
      val out = IO(Output(UInt()))

      out := Cat(in)
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("cat(in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7])")
    (chirrtl should not).include("Cat.scala")
  }

  it should "have a source locator when passing args" in {
    class MyModule extends RawModule {
      val in = IO(Input(Vec(8, UInt(8.W))))
      val out = IO(Output(UInt()))
      out := Cat(in(0), in(1))
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("cat(in[0], in[1])")
    (chirrtl should not).include("Cat.scala")
  }

}
