// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.testing.scalatest.FileCheck
import circt.stage.ChiselStage
import org.scalatest.propspec.AnyPropSpec

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

class VerificationSpec extends AnyPropSpec with FileCheck {

  property("basic equality check should work") {
    ChiselStage.emitCHIRRTL(new SimpleTest).fileCheck() {
      """|CHECK:      node _T = eq(io.in, UInt<2>(0h3))
         |CHECK:      when _T_1 :
         |CHECK-NEXT:     cover(clock, _T, UInt<1>(0h1), "")
         |CHECK:      node _T_3 = neq(io.in, UInt<2>(0h2))
         |CHECK:      assume(clock, _T_3, UInt<1>(0h1), "Assumption failed at
         |CHECK:      node _T_5 = eq(io.out, io.in)
         |CHECK:      node _T_6 = eq(reset, UInt<1>(0h0))
         |CHECK:      intrinsic(circt_chisel_ifelsefatal<format = "Assertion failed: io.in:%d is equal to io.out:%d\n", label = "chisel3_builtin">, clock, _T_5, _T_6, io.in, io.out)
         |""".stripMargin
    }
  }

  property("labeling of verification constructs should work") {

    /** Circuit that contains and labels verification nodes. */
    class LabelTest extends Module {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
      io.out := io.in
      layer.elideBlocks {
        val cov = cover(io.in === 3.U)
        val assm = chisel3.assume(io.in =/= 2.U)
        val asst = chisel3.assert(io.out === io.in)
      }
    }
    // check that verification appear in verilog output
    ChiselStage.emitSystemVerilog(new LabelTest).fileCheck() {
      """|CHECK:      cover__cov: cover(io_in == 8'h3);
         |CHECK:      assume__assm:
         |CHECK-NEXT:   assume(io_in != 8'h2)
         |CHECK-NEXT:   $error("Assumption failed
         |""".stripMargin
    }
  }
}
