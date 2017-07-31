// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._
//import chisel3.core.ExplicitCompileOptions.Strict

/** Extend BasicTester with a simple circuit and finish method.  TesterDriver will call the
  * finish method after the FinishTester's constructor has completed, which will alter the
  * circuit after the constructor has finished.
  */
class FinishTester extends BasicTester {
  val test_wire_width = 2
  val test_wire_override_value = 3

  val counter = Counter(1)

  when(counter.inc()) {
    stop()
  }

  val test_wire = WireInit(1.U(test_wire_width.W))

  // though we just set test_wire to 1, the assert below will pass because
  // the finish will change its value
  assert(test_wire === test_wire_override_value.asUInt)

  /** In finish we use last connect semantics to alter the test_wire in the circuit
    * with a new value
    */
  override def finish(): Unit = {
    test_wire := test_wire_override_value.asUInt
  }
}

class TesterDriverSpec extends ChiselFlatSpec {
  "TesterDriver calls BasicTester's finish method which" should
    "allow modifications of test circuit after the tester's constructor is done" in {
    assertTesterPasses {
      new FinishTester
    }
  }
}

