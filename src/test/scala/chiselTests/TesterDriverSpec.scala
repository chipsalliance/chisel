// See LICENSE for license details.

package chiselTests

import Chisel._
import Chisel.testers.BasicTester

/** Extends basic tester with a finish method.
  *
  */
class FinishTester extends BasicTester {
  var finish_was_run = false

  override def finish(): Unit = {
     finish_was_run = true
  }
}

class TesterDriverSpec extends ChiselFlatSpec {
  class DummyCircuit extends Module {
    val io = new Bundle {
      val in = UInt(INPUT, width = 1)
      val out = UInt(OUTPUT, width = 1)
    }

    io.out := io.in
  }

  class DummyTester extends FinishTester {
    val dut = new DummyCircuit

    dut.io.in := UInt(1)
    Chisel.assert(dut.io.out === UInt(1))

    "Extending BasicTester" should "allow developer to have finish method run automatically" in {
      assert(finish_was_run)
    }
  }

  runTester {
    new DummyTester
  }
}

