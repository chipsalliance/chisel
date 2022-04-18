package chiselTests.util.experimental

import chisel3._
import chisel3.stage.PrintFullStackTraceAnnotation
import chisel3.testers.BasicTester
import chisel3.util.{pla, BitPat}
import chiselTests.ChiselFlatSpec

class PlaSpec extends ChiselFlatSpec {
  "A 1-of-8 decoder (eg. 74xx138 without enables)" should "be generated correctly" in {
    assertTesterPasses(new BasicTester {
      val table = Seq(
        (BitPat("b000"), BitPat("b00000001")),
        (BitPat("b001"), BitPat("b00000010")),
        (BitPat("b010"), BitPat("b00000100")),
        (BitPat("b011"), BitPat("b00001000")),
        (BitPat("b100"), BitPat("b00010000")),
        (BitPat("b101"), BitPat("b00100000")),
        (BitPat("b110"), BitPat("b01000000")),
        (BitPat("b111"), BitPat("b10000000"))
      )
      table.foreach {
        case (i, o) =>
          val (plaIn, plaOut) = pla(table)
          plaIn := WireDefault(i.value.U(3.W))
          chisel3.assert(
            plaOut === o.value.U(8.W),
            "Input " + i.toString + " produced incorrect output BitPat(%b)",
            plaOut
          )
      }
      stop()
    })
  }

  "An active-low 1-of-8 decoder (eg. inverted 74xx138 without enables)" should "be generated correctly" in {
    assertTesterPasses(new BasicTester {
      val table = Seq(
        (BitPat("b000"), BitPat("b00000001")),
        (BitPat("b001"), BitPat("b00000010")),
        (BitPat("b010"), BitPat("b00000100")),
        (BitPat("b011"), BitPat("b00001000")),
        (BitPat("b100"), BitPat("b00010000")),
        (BitPat("b101"), BitPat("b00100000")),
        (BitPat("b110"), BitPat("b01000000")),
        (BitPat("b111"), BitPat("b10000000"))
      )
      table.foreach {
        case (i, o) =>
          val (plaIn, plaOut) = pla(table, BitPat("b11111111"))
          plaIn := WireDefault(i.value.U(3.W))
          chisel3.assert(
            plaOut === ~o.value.U(8.W),
            "Input " + i.toString + " produced incorrect output BitPat(%b)",
            plaOut
          )
      }
      stop()
    })
  }

  "#2112" should "be generated correctly" in {
    assertTesterPasses(new BasicTester {
      val table = Seq(
        (BitPat("b000"), BitPat("b?01")),
        (BitPat("b111"), BitPat("b?01"))
      )
      table.foreach {
        case (i, o) =>
          val (plaIn, plaOut) = pla(table)
          plaIn := WireDefault(i.value.U(3.W))
          chisel3.assert(o === plaOut, "Input " + i.toString + " produced incorrect output BitPat(%b)", plaOut)
      }
      stop()
    })
  }

  "A simple PLA" should "be generated correctly" in {
    assertTesterPasses(new BasicTester {
      val table = Seq(
        (BitPat("b0000"), BitPat("b1")),
        (BitPat("b0001"), BitPat("b1")),
        (BitPat("b0010"), BitPat("b0")),
        (BitPat("b0011"), BitPat("b1")),
        (BitPat("b0100"), BitPat("b1")),
        (BitPat("b0101"), BitPat("b0")),
        (BitPat("b0110"), BitPat("b0")),
        (BitPat("b0111"), BitPat("b0")),
        (BitPat("b1000"), BitPat("b0")),
        (BitPat("b1001"), BitPat("b0")),
        (BitPat("b1010"), BitPat("b1")),
        (BitPat("b1011"), BitPat("b0")),
        (BitPat("b1100"), BitPat("b0")),
        (BitPat("b1101"), BitPat("b1")),
        (BitPat("b1110"), BitPat("b1")),
        (BitPat("b1111"), BitPat("b1"))
      )
      table.foreach {
        case (i, o) =>
          val (plaIn, plaOut) = pla(table)
          plaIn := WireDefault(i.value.U(4.W))
          chisel3.assert(
            plaOut === o.value.U(1.W),
            "Input " + i.toString + " produced incorrect output BitPat(%b)",
            plaOut
          )
      }
      stop()
    })
  }
}
