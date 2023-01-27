package chiselTests.util

import chisel3._
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage}
import chisel3.testers.BasicTester
import chisel3.util.circt.PlusArgsValue

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.io.Source

private class PlusArgsValueTop extends Module {
  val io = IO(new Bundle {
    val wf = Output(UInt(1.W))
    val wv = Output(UInt(32.W))
    val xf = Output(UInt(1.W))
    val xv = Output(UInt(32.W))
  })
  val tmpw = PlusArgsValue(UInt(32.W), "FOO=%d")
  val tmpx = PlusArgsValue(io.xv, "BAR=%d")
  io.xf := tmpx.found
  io.xv := tmpx.result
}

/** A test for intrinsics.  Since chisel is producing intrinsics as tagged
  * extmodules (for now), we explicitly test the chirrtl and annotations rather
  * than the processed firrtl or verilog.  It is worth noting that annotations
  * are implemented (for now) in a way which makes the output valid for all
  * firrtl compilers, hence we write a localized, not end-to-end test
  */
class PlusArgsValueSpec extends AnyFlatSpec with Matchers {
  it should "Should work for types" in {
    val fir = ChiselStage.emitChirrtl(new PlusArgsValueTop)
    val a1 = """extmodule PlusArgsValue_0""".r
    (fir should include).regex(a1)
    val b1 = """defname = PlusArgsValue_0""".r
    (fir should include).regex(b1)
    val a2 = """extmodule PlusArgsValue_1""".r
    (fir should include).regex(a2)
    val b2 = """defname = PlusArgsValue_1""".r
    (fir should include).regex(b2)

    // The second elaboration uses a unique name since the Builder is reused (?)
    val c = """Intrinsic\(~PlusArgsValueTop\|PlusArgsValue.*,circt.plusargs.value\)"""
    ((new chisel3.stage.ChiselStage)
      .execute(
        args = Array("--no-run-firrtl"),
        annotations = Seq(chisel3.stage.ChiselGeneratorAnnotation(() => new PlusArgsValueTop))
      )
      .mkString("\n") should include).regex(c)
  }
}
