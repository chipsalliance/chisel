// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage
import org.scalatest.matchers.should.Matchers

import java.io.File

class SimpleTest extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(8.W))
    val out = Output(UInt(8.W))
  })
  io.out := io.in
  cover(io.in === 3.U)
  when(io.in === 3.U) {
    assume(io.in =/= 2.U)
    assert(io.out === io.in, p"${FullName(io.in)}:${io.in} is equal to ${FullName(io.out)}:${io.out}")
  }
}

class VerificationSpec extends ChiselPropSpec with Matchers {

  def assertContains(s: Seq[String], x: String): Unit = {
    val containsLine = s.map(_.contains(x)).reduce(_ || _)
    assert(containsLine, s"\n  $x\nwas not found in`\n  ${s.mkString("\n  ")}``")
  }

  property("basic equality check should work") {
    val fir = ChiselStage.emitCHIRRTL(new SimpleTest)
    val lines = fir.split("(\n|@)").map(_.trim).toIndexedSeq

    // reset guard around the verification statement
    assertContains(lines, "when _T_1 :")
    assertContains(lines, "cover(clock, _T, UInt<1>(0h1), \"\")")

    assertContains(lines, "assume(clock, _T_3, UInt<1>(0h1), \"Assumption failed")
    assertContains(lines, "node _T_5 = eq(io.out, io.in)")
    assertContains(lines, "node _T_6 = eq(reset, UInt<1>(0h0))")
    assertContains(
      lines,
      """intrinsic(circt_chisel_ifelsefatal<format = "Assertion failed: io.in:%d is equal to io.out:%d\n", label = "chisel3_builtin">, clock, _T_5, _T_6, io.in, io.out)"""
    )
  }

  property("annotation of verification constructs should work") {

    /** Circuit that contains and annotates verification nodes. */
    class AnnotationTest extends Module {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
      io.out := io.in
      val cov = cover(io.in === 3.U)
      val assm = chisel3.assume(io.in =/= 2.U)
      val asst = chisel3.assert(io.out === io.in)
    }

    // compile circuit
    val testDir = new File("test_run_dir", "VerificationAnnotationTests")
    ChiselStage.emitSystemVerilogFile(gen = new AnnotationTest, args = Array("-td", testDir.getPath))

    // read in FIRRTL file
    val svFile = new File(testDir, "AnnotationTest.sv")
    svFile should exist
    val svLines = scala.io.Source.fromFile(svFile).getLines().toList

    // check that verification appear in verilog output
    exactly(1, svLines) should include("cover__cov: cov")
    exactly(1, svLines) should include("assume__assm:")
    exactly(1, svLines) should include("assume(io_in != 8'h2)")
    exactly(1, svLines) should include("$error(\"Assumption failed")
  }
}
