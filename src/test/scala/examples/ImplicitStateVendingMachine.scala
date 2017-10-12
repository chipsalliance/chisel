// See LICENSE for license details.

package examples

import chiselTests.ChiselFlatSpec
import chisel3._

// Vending machine implemented with an implicit state machine
class ImplicitStateVendingMachine extends SimpleVendingMachine {
  // We let the value of nickel be 1 and dime be 2 for efficiency reasons
  val value = RegInit(0.asUInt(3.W))
  val incValue = WireInit(0.asUInt(3.W))
  val doDispense = value >= 4.U // 4 * nickel as 1 == $0.20

  when (doDispense) {
    value := 0.U // No change given
  } .otherwise {
    value := value + incValue
  }

  when (io.nickel) { incValue := 1.U }
  when (io.dime) { incValue := 2.U }

  io.dispense := doDispense
}

class ImplicitStateVendingMachineSpec extends ChiselFlatSpec {
  "An vending machine implemented with implicit state" should "work" in {
    assertTesterPasses { new SimpleVendingMachineTester(new ImplicitStateVendingMachine) }
  }
}
