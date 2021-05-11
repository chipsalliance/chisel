// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.experimental.minimizer

import chisel3._
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage}
import chisel3.util._
import chisel3.util.experimental.decode._
import chisel3.util.pla
import firrtl.backends.experimental.smt.EmittedSMTModelAnnotation
import firrtl.options.TargetDirAnnotation
import firrtl.util.BackendCompilationUtilities
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MinimizerSpec extends AnyFlatSpec with Matchers {
  class DecodeTestModule(minimizer: Minimizer, table: TruthTable, reference: TruthTable) extends Module {
    val i = IO(Input(UInt(table.table.head._1.getWidth.W)))
    val (plaI, plaO) = pla(reference.table.toSeq, BitPat(reference.default.value.U))
    plaI := i
    chisel3.experimental.verification.assert((decoder(minimizer, i, table) === plaO) | (i === reference.default))
  }

  def test(minimizer: Minimizer, testcase: (TruthTable, TruthTable)) = {
    val testDir = os.pwd / "test_run_dir" / "MinimizerSpec" / BackendCompilationUtilities.timeStamp
    os.makeDir.all(testDir)
    val checkFile = testDir / "check.smt"
    os.write(checkFile,
      (new ChiselStage).execute(
        Array("-E", "experimental-smt2"),
        Seq(
          ChiselGeneratorAnnotation(() => new DecodeTestModule(minimizer, table = testcase._1, testcase._2)),
          TargetDirAnnotation(testDir.toString)
        )
      ).collectFirst {
        case EmittedSMTModelAnnotation(_, smt, _) => smt
      }.get +
        """; combinational logic check
          |(declare-fun s0 () DecodeTestModule_s)
          |(assert (not (DecodeTestModule_a s0)))
          |(check-sat)
          |""".stripMargin
    )
    os.proc("z3", checkFile).call().out.toString
  }

  val case0 = (
    TruthTable(
      Map(
        BitPat("b000") -> BitPat("b0"),
        // BitPat("b001") -> BitPat("b?"),  // same as default, can be omitted
        // BitPat("b010") -> BitPat("b?"),  // same as default, can be omitted
        BitPat("b011") -> BitPat("b0"),
        BitPat("b100") -> BitPat("b1"),
        BitPat("b101") -> BitPat("b1"),
        BitPat("b110") -> BitPat("b0"),
        BitPat("b111") -> BitPat("b1")
      ),
      BitPat("b?")
    ),
    TruthTable(
      Map(
        BitPat("b10?") -> BitPat("b1"),
        BitPat("b1?1") -> BitPat("b1"),
      ),
      BitPat("b?")
    )
  )

  val case1 = (
    TruthTable(
      Map(
        BitPat("b000") -> BitPat("b0"),
        BitPat("b001") -> BitPat("b0"),
        // BitPat("b010") -> BitPat("b?"),  // same as default, can be omitted
        (BitPat("b011") -> BitPat("b0")),
        // BitPat("b100") -> BitPat("b?"),  // same as default, can be omitted
        // BitPat("b101") -> BitPat("b?"),  // same as default, can be omitted
        BitPat("b110") -> BitPat("b1"),
        BitPat("b111") -> BitPat("b0")
      ),
      BitPat("b?")
    ),
    TruthTable(
      Map(
        BitPat("b?10") -> BitPat("b1"),
        // BitPat("b1?0") -> BitPat("b1"),  // both are ok
      ),
      BitPat("b?")
    ),
  )
}
