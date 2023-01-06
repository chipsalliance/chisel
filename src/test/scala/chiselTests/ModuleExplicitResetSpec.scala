// SPDX-License-Identifier: Apache-2.0

package chiselTests

import circt.stage.ChiselStage

import scala.annotation.nowarn

@nowarn("msg=Chisel compatibility mode is deprecated")
class ModuleExplicitResetSpec extends ChiselFlatSpec {

  "A Module with an explicit reset in compatibility mode" should "elaborate" in {
    import Chisel._
    val myReset = true.B
    class ModuleExplicitReset(reset: Bool) extends Module(_reset = reset) {
      val io = new Bundle {
        val done = Bool(OUTPUT)
      }

      io.done := false.B
    }

    ChiselStage.elaborate {
      new ModuleExplicitReset(myReset)
    }
  }
}
