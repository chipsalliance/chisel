// See LICENSE for license details.

package chiselTests.stage

import chisel3._
import chisel3.stage.ChiselStage

import org.scalatest.{FlatSpec, Matchers}

import firrtl.options.Dependency

object ChiselStageSpec {

  class Foo extends MultiIOModule {
    val addr = IO(Input(UInt(4.W)))
    val out = IO(Output(Bool()))
    val bar = SyncReadMem(8, Bool())
    out := bar(addr)
  }

}

class ChiselStageSpec extends FlatSpec with Matchers {

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

<<<<<<< HEAD
  it should "generate a Chisel circuit from a Chisel module" in {
    ChiselStage.elaborate(new Foo)
=======
  ignore should "generate a Chisel circuit from a Chisel module" in {
    info("no files were written")
    catchWrites { ChiselStage.elaborate(new Foo) } shouldBe a[Right[_, _]]
>>>>>>> 9f1d6cbb... Ignore tests using System.setSecurityManager (#1661)
  }

  behavior of "ChiselStage$.convert"

<<<<<<< HEAD
  it should "generate a CHIRRTL circuit from a Chisel module" in {
    ChiselStage.convert(new Foo)
=======
  ignore should "generate a CHIRRTL circuit from a Chisel module" in {
    info("no files were written")
    catchWrites { ChiselStage.convert(new Foo) } shouldBe a[Right[_, _]]
  }

  behavior of "ChiselStage$.emitChirrtl"

  ignore should "generate a CHIRRTL string from a Chisel module" in {
    val wrapped = catchWrites { ChiselStage.emitChirrtl(new Foo) }

    info("no files were written")
    wrapped shouldBe a[Right[_, _]]

    info("returned string looks like FIRRTL")
    wrapped.right.get should include ("circuit")
  }

  behavior of "ChiselStage$.emitFirrtl"

  ignore should "generate a FIRRTL string from a Chisel module" in {
    val wrapped = catchWrites { ChiselStage.emitFirrtl(new Foo) }

    info("no files were written")
    wrapped shouldBe a[Right[_, _]]

    info("returned string looks like FIRRTL")
    wrapped.right.get should include ("circuit")
  }

  behavior of "ChiselStage$.emitVerilog"

  ignore should "generate a Verilog string from a Chisel module" in {
    val wrapped = catchWrites { ChiselStage.emitVerilog(new Foo) }

    info("no files were written")
    wrapped shouldBe a[Right[_, _]]

    info("returned string looks like Verilog")
    wrapped.right.get should include ("endmodule")
  }

  behavior of "ChiselStage$.emitSystemVerilog"

  ignore should "generate a SystemvVerilog string from a Chisel module" in {
    val wrapped = catchWrites { ChiselStage.emitSystemVerilog(new Foo) }
    info("no files were written")
    wrapped shouldBe a[Right[_, _]]

    info("returned string looks like Verilog")
    wrapped.right.get should include ("endmodule")
>>>>>>> 9f1d6cbb... Ignore tests using System.setSecurityManager (#1661)
  }

  behavior of "ChiselStage phase ordering"

  it should "only run elaboration once" in new ChiselStageFixture {
    info("Phase order is:\n" + stage.phaseManager.prettyPrint("    "))

    val order = stage.phaseManager.flattenedTransformOrder.map(Dependency.fromTransform)

    info("Elaborate only runs once")
    exactly (1, order) should be (Dependency[chisel3.stage.phases.Elaborate])
  }

}
