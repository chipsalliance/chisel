package chiselTests.util

import chisel3.util.experimental.BitSet
import chisel3.util.BitPat
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class BitSetSpec extends AnyFlatSpec with Matchers {
  behavior of classOf[BitSet].toString

  it should "reject unequal width when constructing a BitSet" in {
    intercept[IllegalArgumentException] {
      BitSet.fromString(
        """b0010
          |b00010
          |""".stripMargin)
    }
  }

  it should "return empty subtraction result correctly" in {
    val aBitPat = BitPat("b10?")
    val bBitPat = BitPat("b1??")

    aBitPat.subtract(bBitPat).isEmpty should be (true)
  }

  it should "return nonempty subtraction result correctly" in {
    val aBitPat = BitPat("b10?")
    val bBitPat = BitPat("b1??")
    val cBitPat = BitPat("b11?")
    val dBitPat = BitPat("b100")

    val diffBitPat = bBitPat.subtract(aBitPat)
    bBitPat.cover(diffBitPat) should be (true)
    diffBitPat.equals(cBitPat) should be (true)

    val largerdiffBitPat = bBitPat.subtract(dBitPat)
    aBitPat.cover(dBitPat) should be (true)
    largerdiffBitPat.cover(diffBitPat) should be (true)
  }

  it should "be able to handle complex subtract between BitSet" in {
    val aBitSet = BitSet.fromString(
      """b?01?0
        |b11111
        |b00000
        |""".stripMargin)
    val bBitSet = BitSet.fromString(
      """b?1111
        |b?0000
        |""".stripMargin
    )
    val expected = BitPat("b?01?0")

    expected.equals(aBitSet.subtract(bBitSet)) should be (true)
  }

  it should "be generated from BitPat union" in {
    val aBitSet = BitSet.fromString(
      """b001?0
        |b000??""".stripMargin)
    val aBitPat = BitPat("b000??")
    val bBitPat = BitPat("b001?0")
    val cBitPat = BitPat("b00000")
    aBitPat.cover(cBitPat) should be (true)
    aBitSet.cover(bBitPat) should be (true)

    aBitSet.equals(aBitPat.union(bBitPat)) should be (true)
  }

  it should "be generated from BitPat subtraction" in {
    val aBitSet = BitSet.fromString(
      """b001?0
        |b000??""".stripMargin)
    val aBitPat = BitPat("b00???")
    val bBitPat = BitPat("b001?1")

    aBitSet.equals(aBitPat.subtract(bBitPat)) should be (true)
  }

  it should "union two BitSet together" in {
    val aBitSet = BitSet.fromString(
      """b001?0
        |b001?1
        |""".stripMargin)
    val bBitSet = BitSet.fromString(
      """b000??
        |b01???
        |""".stripMargin
    )
    val cBitPat = BitPat("b0????")
    cBitPat.equals(aBitSet.union(bBitSet)) should be (true)
  }

  it should "be decoded" in {
    import chisel3._
    import chisel3.util.experimental.decode.decoder
    // [0 - 256] part into: [0 - 31], [32 - 47, 64 - 127], [192 - 255]
    // "0011????" "10??????" is empty to error
    chisel3.stage.ChiselStage.emitSystemVerilog(new Module {
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(4.W)))
      out := decoder.bitset(in, Seq(
        BitSet.fromString(
          "b000?????"
        ),
        BitSet.fromString(
          """b0010????
            |b01??????
            |""".stripMargin
        ),
        BitSet.fromString(
          "b11??????"
        )
      ), true)
    })
  }

}
