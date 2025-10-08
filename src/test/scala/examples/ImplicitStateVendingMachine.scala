// SPDX-License-Identifier: Apache-2.0

package examples

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

// Vending machine implemented with an implicit state machine
class ImplicitStateVendingMachine extends SimpleVendingMachine {
  // We let the value of nickel be 1 and dime be 2 for efficiency reasons
  val value = RegInit(0.asUInt(3.W))
  val incValue = WireDefault(0.asUInt(3.W))
  val doDispense = value >= 4.U // 4 * nickel as 1 == $0.20

  when(doDispense) {
    value := 0.U // No change given
  }.otherwise {
    value := value + incValue
  }

  when(io.nickel) { incValue := 1.U }
  when(io.dime) { incValue := 2.U }

  io.dispense := doDispense
}

class ImplicitStateVendingMachineSpec extends AnyFlatSpec with Matchers with ChiselSim {
  "An vending machine implemented with implicit state" should "work" in {
    simulate { new SimpleVendingMachineTester(new ImplicitStateVendingMachine) }(RunUntilFinished(11))
  }
}
