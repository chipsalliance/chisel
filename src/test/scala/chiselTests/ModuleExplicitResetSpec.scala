// See LICENSE for license details.

package chiselTests

import chisel3.stage.ChiselStage

class ModuleExplicitResetSpec extends ChiselFlatSpec  {

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
