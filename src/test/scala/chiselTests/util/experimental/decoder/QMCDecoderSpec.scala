// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.experimental.decoder

import chisel3._
import chisel3.util._

import chisel3.testers.BasicTester
import chisel3.util.experimental.decoder.QMCDecoder
import chiselTests.ChiselFlatSpec


class QMCDecoderSpec extends ChiselFlatSpec {
  "Function with more than 32 outputs" should "be simplified correctly" in {
    assertTesterPasses(new BasicTester {
      val out = WireDefault(
        QMCDecoder().decode(
          addr = WireDefault(1.U(1.W)),
          default = BitPat("b" + "1" + "0" * 32),
          mapping = Seq(BitPat("b0") -> BitPat("b" + "?" * 33))
        )
      )
      chisel3.assert(out(32) === 1.U(1.W))
      stop()
    })
  }
}
