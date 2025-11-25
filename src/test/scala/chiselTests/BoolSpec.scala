// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import org.scalatest.flatspec.AnyFlatSpec
import scala.reflect.Selectable.reflectiveSelectable

class BoolSpec extends AnyFlatSpec with ChiselSim {

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
}
