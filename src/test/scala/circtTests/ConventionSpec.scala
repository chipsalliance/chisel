// SPDX-License-Identifier: Apache-2.0

package circtTests

import chisel3._
import circt.convention
import circt.stage.ChiselStage
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

class ConventionSpec extends AnyFunSpec with Matchers {
  describe("convention.scalarized") {
    it("should scalarize the ports of a module") {
      class PassThrough extends RawModule {
        val i = IO(Input(new Bundle {
          val x = UInt(16.W)
          val y = UInt(16.W)
        }))
        val o = IO(Output(new Bundle {
          val x = UInt(16.W)
          val y = UInt(16.W)
        }))
        o := i
      }

      def makePassThrough() = {
        val p = new PassThrough
        convention.scalarized(p)
        p
      }

      val expected = Array(
        "input  [15:0] i_x,",
        "input  [15:0] i_y,",
        "output [15:0] o_x,",
        "output [15:0] o_y",
        "assign o_x = i_x;",
        "assign o_y = i_y;"
      )

      val options = Array(
        "--preserve-aggregate=all",
        "--strip-debug-info",
        "--scalarize-top-module=false",
        "--lowering-options=disallowPortDeclSharing"
      )

      val actual = ChiselStage.emitSystemVerilog(makePassThrough(), Array(), options).split("\n").map(_.trim)
      expected.foreach { e => actual should contain(e) }
    }
  }
}
