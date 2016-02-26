// See LICENSE for license details.

package chiselTests

import Chisel._
import Chisel.testers.BasicTester

/** This llustrates a problem, where the string second
  * argument to Chisel.assert causes a firrtl error
  */
class QuoteProblemCircuit extends Module {
  val w = 32
  val magic_number = 42
  val io = new Bundle {
    val out = UInt(OUTPUT, w)
  }
  io.out := UInt(magic_number)
}

class QuoteProblemTests extends BasicTester {
  val device_under_test = Module( new QuoteProblemCircuit  )
  val c = device_under_test

  when(Bool(true)) {
    /*
    The assert statement here creates a complex printf with escaped quotes
    which scala firrtl's parser does not seem to be able to handle.
     */
    assert(Bool(false), "This better be the answer")
  }
}
class QuoteProblemTester extends ChiselFlatSpec {
  "QuoteProblem" should "compile and run without incident" in {
    assertTesterPasses { new QuoteProblemTests }
  }
}

