// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class BoolSpec extends AnyFlatSpec with ChiselSim with Matchers {

  "implication" should "work in RTL" in {

    val truthTable = Seq(
      ((0, 0), 1),
      ((0, 1), 1),
      ((1, 0), 0),
      ((1, 1), 1)
    )

    simulateRaw(
      new RawModule {
        val a, b = IO(Input(Bool()))
        val c = IO(Output(Bool()))
        c :<= a.implies(b)
      }
    ) { dut =>
      truthTable.foreach { case ((a, b), c) =>
        info(s"$a -> $b == $c")
        dut.a.poke(a)
        dut.b.poke(b)
        dut.c.expect(c)
      }
    }
  }

  "padding a Bool to a wider width" should "give a descriptive error message" in {
    val e = the[IllegalArgumentException] thrownBy {
      ChiselStage.emitCHIRRTL(new RawModule {
        val in = IO(Input(Bool()))
        val out = IO(Output(UInt(2.W)))
        out := in.pad(2)
      })
    }
    e.getMessage should include("a Bool is always 1 bit wide")
  }
}
