// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.util.Enum
import chisel3.testers.BasicTester

class EnumSpec extends ChiselFlatSpec {

  "1-entry Enums" should "work" in {
    assertTesterPasses(new BasicTester {
      val onlyState :: Nil = Enum(1)
      val wire = WireDefault(onlyState)
      chisel3.assert(wire === onlyState)
      stop()
    })
  }
}
