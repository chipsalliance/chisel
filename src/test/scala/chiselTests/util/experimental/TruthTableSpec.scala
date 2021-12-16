// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.experimental

import chisel3._
import chisel3.util.BitPat
import chisel3.util.experimental.decode.{TruthTable, decoder}
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
    assert(TruthTable.fromString(str) == table)
  }
  "TruthTable" should "merge same key" in {
    assert(
      TruthTable.fromString(
        """001100->??1
          |001100->1??
          |???
          |""".stripMargin
      ) == TruthTable.fromString(
        """001100->1?1
          |???
          |""".stripMargin
      )
    )
  }
  "TruthTable" should "crash when merging 0 and 1" in {
    intercept[IllegalArgumentException] {
      TruthTable.fromString(
        """0->0
          |0->1
          |???
          |""".stripMargin
      )
    }
  }
  "TruthTable" should "be reproducible" in {
    class Foo extends Module {

      val io = IO(new Bundle{
        val in = Input(UInt(4.W))
        val out = Output(UInt(16.W))
      })


      val table = TruthTable(
        (0 until 16).map{
          i => BitPat(i.U(4.W)) -> BitPat((1<<i).U(16.W))
        },
        BitPat.dontCare(16)
      )

      io.out := decoder.qmc(io.in, table)
    }
    assert(chisel3.stage.ChiselStage.emitChirrtl(new Foo) == chisel3.stage.ChiselStage.emitChirrtl(new Foo))
  }
}
