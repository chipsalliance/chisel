// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.util.Enum
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class EnumSpec extends AnyFlatSpec with ChiselSim {

  "1-entry Enums" should "work" in {
    simulate(new Module {
      val onlyState :: Nil = Enum(1)
      val wire = WireDefault(onlyState)
      chisel3.assert(wire === onlyState)
      stop()
    })(RunUntilFinished(3))
  }
}
