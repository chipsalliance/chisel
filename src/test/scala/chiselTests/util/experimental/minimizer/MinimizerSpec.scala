// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.experimental.minimizer

import chisel3._
import chisel3.util._
import chisel3.util.experimental.decode._
import chisel3.util.pla
import chiselTests.SMTModelCheckingSpec

class DecodeTestModule(minimizer: Minimizer, table: TruthTable) extends Module {
  val i = IO(Input(UInt(table.table.head._1.getWidth.W)))
  val (unminimizedI, unminimizedO) = pla(table.table.toSeq)
  unminimizedI := i
  val minimizedO: UInt = decoder(minimizer, i, table)

  chisel3.experimental.verification.assert(
    // for each instruction, if input matches, output should match, not no matched, fallback to default
    (table.table.map { case (key, value) => (i === key) && (minimizedO === value) } ++
      Seq(table.table.keys.map(i =/= _).reduce(_ && _) && minimizedO === table.default)).reduce(_ || _)
  )
}

trait MinimizerSpec extends SMTModelCheckingSpec {
  def minimizer: Minimizer

  def minimizerTest(testcase: TruthTable, caseName: String) = {
    test(
      () => new DecodeTestModule(minimizer, table = testcase),
      s"${minimizer.getClass.getSimpleName}.$caseName",
      success
    )
  }

  val case0 = TruthTable(
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
    )

  val case1 = TruthTable(
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
    )
}
