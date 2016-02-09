// See LICENSE for license details.

package chiselTests

import Chisel._
import Chisel.testers.BasicTester

/**
  * Created by chick on 2/8/16.
  */
class FinishTester extends BasicTester {
  var finish_was_run = false

  override def finish(): Unit = {
    finish_was_run = true
  }
}

class FinishTesterSpec extends ChiselFlatSpec {
  class DummyCircuit extends Module {
    val io = new Bundle {
      val in = Bool(INPUT)
      val out = Bool(OUTPUT)
    }
  }

  runTester {
    new FinishTester {
      val dut = new DummyCircuit

      "Extending BasicTester" should "allow developer to have finish method run automatically" in {
        assert(finish_was_run)
      }
    }
  }
}
