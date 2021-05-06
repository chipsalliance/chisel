// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.experimental

import chisel3.util.BitPat
import chisel3.util.experimental.decode.TruthTable
import org.scalatest.flatspec.AnyFlatSpec

class TruthTableSpec extends AnyFlatSpec {
  val table = TruthTable(
    Map(
      // BitPat("b000") -> BitPat("b0"),
      BitPat("b001") -> BitPat("b?"),
      BitPat("b010") -> BitPat("b?"),
      // BitPat("b011") -> BitPat("b0"),
      BitPat("b100") -> BitPat("b1"),
      BitPat("b101") -> BitPat("b1"),
      // BitPat("b110") -> BitPat("b0"),
      BitPat("b111") -> BitPat("b1")
    ),
    BitPat("b0")
  )
  val str = """001->?
              |010->?
              |100->1
              |101->1
              |111->1
              |0""".stripMargin
  "TruthTable" should "serialize" in {
    assert(table.toString contains "001->?")
    assert(table.toString contains "010->?")
    assert(table.toString contains "100->1")
    assert(table.toString contains "111->1")
    assert(table.toString contains "     0")
  }
  "TruthTable" should "deserialize" in {
    assert(TruthTable(str) == table)
  }
}
