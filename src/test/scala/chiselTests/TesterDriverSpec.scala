// See LICENSE for license details.

package chiselTests

import Chisel._
import Chisel.testers.BasicTester

/** Extend basic tester with a finish method.  TesterDriver will call the
  * finish method after the Tester's constructor has completed
  * -
  * In this example we use last connect semantics to alter the circuit after
  * the constructor has completed
  */
class FinishTester extends BasicTester {
  val test_wire_width = 2
  val test_wire_override_value = 3

  val test_wire = Wire(UInt(1, width = test_wire_width))

  test_wire := UInt(1, width = test_wire_width)
  // though we just test_wire to 1, the assert below will be true because
  // the finish will override it
  assert(test_wire === UInt(test_wire_override_value))

  override def finish(): Unit = {
    test_wire := UInt(test_wire_override_value, width = test_wire_width)
  }
}

class DummyCircuit extends Module {
  val io = new Bundle {
    val in = UInt(INPUT, width = 1)
    val out = UInt(OUTPUT, width = 1)
  }

  io.out := io.in
}

class DummyTester extends FinishTester {
  val dut = Module(new DummyCircuit)

  dut.io.in := UInt(1)
  Chisel.assert(dut.io.out === UInt(1))

  stop()
}

class TesterDriverSpec extends ChiselFlatSpec {
  "TesterDriver calls a BasicTester subclass's finish method which" should
    "allow modifications of test circuit after tester constructor is done" in {
    assertTesterPasses {
      new DummyTester
    }
  }
}

