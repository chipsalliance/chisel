package chiselTests.util

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.util.{RegEnable}

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class RegEnableSpec extends AnyFlatSpec with Matchers {
  behavior.of("util.RegEnable")

  it should "have source locators when passed nextVal, ena" in {
    class MyModule extends Module {
      val in = IO(Input(Bool()))
      val out = IO(Output(Bool()))
      out := RegEnable(in, true.B)
    }
    val chirrtl = ChiselStage.emitChirrtl(new MyModule)
    val reset = """reset.*RegSpec.scala""".r
    (chirrtl should include).regex(reset)
    val update = """out_r.* in .*RegSpec.scala""".r
    (chirrtl should include).regex(update)
    (chirrtl should not).include("Reg.scala")
  }

    it should "have source locators when passed next, init, enable" in {
    class MyModule extends Module {
      val in = IO(Input(Bool()))
      val out = IO(Output(Bool()))
      out := RegEnable(in, true.B, true.B)
    }
    val chirrtl = ChiselStage.emitChirrtl(new MyModule)
    val reset = """reset .*RegSpec.scala""".r
    (chirrtl should include).regex(reset)
    val update = """out_r.* in .*RegSpec.scala""".r
    (chirrtl should include).regex(update)
    (chirrtl should not).include("Reg.scala")
  }
}
