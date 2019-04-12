// See LICENSE for license details.

package chiselTests

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

    elaborate {
      new ModuleExplicitReset(myReset)
    }
  }
}
