// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.experimental.minimizer

import chisel3._
import chisel3.stage.PrintFullStackTraceAnnotation
import chisel3.util._
import chisel3.testers.BasicTester
import chisel3.util.experimental.minimizer.QMCMinimizer
import chiselTests.ChiselFlatSpec


class QMCDecoderSpec extends ChiselFlatSpec {
  "Function with more than 32 outputs" should "be simplified correctly" in {
    assertTesterPasses(new BasicTester {
      //      val out = WireDefault(
      //        QMCDecoder().decode(
      //          addr = WireDefault(1.U(1.W)),
      //          default = BitPat("b" + "1" + "0" * 32),
      //          mapping = Seq(BitPat("b0") -> BitPat("b" + "?" * 33))
      //        )
      //      )
      //      chisel3.assert(out(32) === 1.U(1.W))
      stop()
    }, Seq(), Seq(PrintFullStackTraceAnnotation))
  }

  "A" should "be simplified correctly" in {
    assertTesterPasses(new BasicTester {
      println(QMCMinimizer().minimize(
        BitPat("b0"),
        Seq(
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
          (BitPat("b1111"), BitPat("b1")),
        )
      ))
      stop()
    }, Seq(), Seq(PrintFullStackTraceAnnotation))
  }
}
