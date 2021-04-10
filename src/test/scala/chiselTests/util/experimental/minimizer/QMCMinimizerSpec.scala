// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.experimental.minimizer

import chisel3.util._
import chisel3.util.experimental.minimizer.QMCMinimizer
import org.scalatest.flatspec.AnyFlatSpec


class QMCMinimizerSpec extends AnyFlatSpec {
  private def bitPatEqual(x: BitPat, y: BitPat): Boolean = {
    x.value == y.value && x.mask == y.mask && x.getWidth == y.getWidth
  }

  "A simple truth table" should "be simplified correctly" in {
    val default = BitPat("b?")

    val truthTable = Seq(
      (BitPat("b000"), BitPat("b0")),
      // (BitPat("b001"), BitPat("b?")),  // same as default, can be omitted
      // (BitPat("b010"), BitPat("b?")),  // same as default, can be omitted
      (BitPat("b011"), BitPat("b0")),
      (BitPat("b100"), BitPat("b1")),
      (BitPat("b101"), BitPat("b1")),
      (BitPat("b110"), BitPat("b0")),
      (BitPat("b111"), BitPat("b1")),
    )

    val minimizedTable = QMCMinimizer().minimize(default, truthTable)
    assert(minimizedTable.length == 2)

    val minterms = minimizedTable.filter(a => bitPatEqual(a._2, BitPat("b1"))).map(_._1)
    assert(minterms.length == 2)
    assert(minterms.count(bitPatEqual(_, BitPat("b10?"))) == 1)
    assert(minterms.count(bitPatEqual(_, BitPat("b1?1"))) == 1)
  }
}
