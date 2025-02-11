// SPDX-License-Identifier: Apache-2.0

package chiselTests.util

import chisel3._
import chisel3.util.Cat
import chisel3.experimental.noPrefix
import chiselTests.ChiselFlatSpec
import _root_.circt.stage.ChiselStage

object CatSpec {

  class JackIsATypeSystemGod extends Module {
    val in = IO(Input(Vec(0, UInt(8.W))))
    val out = IO(Output(UInt(8.W)))

    out := Cat(in)
  }

}

class CatSpec extends ChiselFlatSpec {

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

  it should "have prefixed naming" in {
    class MyModule extends RawModule {
      val in = IO(Input(Vec(8, UInt(8.W))))
      val out = IO(Output(UInt()))

      // noPrefix to avoid `out` as prefix
      out := noPrefix(Cat(in))
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("node lo_lo = cat(in[6], in[7])")
    chirrtl should include("node lo_hi = cat(in[4], in[5])")
    chirrtl should include("node hi_lo = cat(in[2], in[3])")
    chirrtl should include("node hi_hi = cat(in[0], in[1])")
  }

  it should "have a source locator when passing a seq" in {
    class MyModule extends RawModule {
      val in = IO(Input(Vec(8, UInt(8.W))))
      val out = IO(Output(UInt()))

      // noPrefix to avoid `out` as prefix
      out := Cat(in)
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    chirrtl should include("cat(in[0], in[1])")
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
