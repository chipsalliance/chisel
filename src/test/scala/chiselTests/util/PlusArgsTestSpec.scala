package chiselTests.util

import chisel3._
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage}
import chisel3.testers.BasicTester
import chisel3.util.circt.PlusArgsTest

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.io.Source

private class PlusArgsTestTop extends Module {
  val io = IO(new Bundle {
    val w = Output(UInt(1.W))
    val x = Output(UInt(1.W))
    val z = Input(UInt(32.W))
  })
  io.w := PlusArgsTest(UInt(32.W), "FOO")
  io.x := PlusArgsTest(io.z, "BAR")
}

/** A test for intrinsics.  Since chisel is producing intrinsics as tagged
  * extmodules (for now), we explicitly test the chirrtl and annotations rather
  * than the processed firrtl or verilog.  It is worth noting that annotations
  * are implemented (for now) in a way which makes the output valid for all
  * firrtl compilers, hence we write a localized, not end-to-end test
  */
class PlusArgsTestSpec extends AnyFlatSpec with Matchers {
  it should "Should work for types" in {
    val fir = ChiselStage.emitChirrtl(new PlusArgsTestTop)
    val a1 = """extmodule PlusArgsTest_0""".r
    (fir should include).regex(a1)
    val b1 = """defname = PlusArgsTest_0""".r
    (fir should include).regex(b1)
    val a2 = """extmodule PlusArgsTest_1""".r
    (fir should include).regex(a2)
    val b2 = """defname = PlusArgsTest_1""".r
    (fir should include).regex(b2)

    // The second elaboration uses a unique name since the Builder is reused (?)
    val c = """Intrinsic\(~PlusArgsTestTop\|PlusArgsTest.*,circt.plusargs.test\)"""
    ((new chisel3.stage.ChiselStage)
      .execute(
        args = Array("--no-run-firrtl"),
        annotations = Seq(chisel3.stage.ChiselGeneratorAnnotation(() => new PlusArgsTestTop))
      )
      .mkString("\n") should include).regex(c)
  }
}
