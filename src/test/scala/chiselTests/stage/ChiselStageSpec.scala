// See LICENSE for license details.

package chiselTests.stage

import chisel3._
import chisel3.stage.ChiselStage

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

object ChiselStageSpec {

  class Foo extends MultiIOModule {
    val addr = IO(Input(UInt(4.W)))
    val out = IO(Output(Bool()))
    val bar = SyncReadMem(8, Bool())
    out := bar(addr)
  }

}

class ChiselStageSpec extends AnyFlatSpec with Matchers {

  import ChiselStageSpec._

  private trait ChiselStageFixture {
    val stage = new ChiselStage
  }

  behavior of "ChiselStage.emitChirrtl"

  it should "return a CHIRRTL string" in new ChiselStageFixture {
    stage.emitChirrtl(new Foo) should include ("infer mport")
  }

  behavior of "ChiselStage.emitFirrtl"

  it should "return a High FIRRTL string" in new ChiselStageFixture {
    stage.emitFirrtl(new Foo) should include ("mem bar")
  }

  behavior of "ChiselStage.emitVerilog"

  it should "return a Verilog string" in new ChiselStageFixture {
    stage.emitVerilog(new Foo) should include ("endmodule")
  }

  behavior of "ChiselStage$.elaborate"

  it should "generate a Chisel circuit from a Chisel module" in {
    ChiselStage.elaborate(new Foo)
  }

  behavior of "ChiselStage$.convert"

  it should "generate a CHIRRTL circuit from a Chisel module" in {
    ChiselStage.convert(new Foo)
  }

}
