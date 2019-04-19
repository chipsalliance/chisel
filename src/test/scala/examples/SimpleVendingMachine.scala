// See LICENSE for license details.

package examples

import chiselTests.ChiselFlatSpec
import chisel3.testers.BasicTester
import chisel3._
import chisel3.util._

class SimpleVendingMachineIO extends Bundle {
  val nickel = Input(Bool())
  val dime   = Input(Bool())
  val dispense  = Output(Bool())
}

// Superclass for vending machines with very simple IO
abstract class SimpleVendingMachine extends Module {
  val io = IO(new SimpleVendingMachineIO)
  assert(!(io.nickel && io.dime), "Only one of nickel or dime can be input at a time!")
}

// Vending machine implemented with a Finite State Machine
class FSMVendingMachine extends SimpleVendingMachine {
  val sIdle :: s5 :: s10 :: s15 :: sOk :: Nil = Enum(5)
  val state = RegInit(sIdle)

  switch (state) {
    is (sIdle) {
      when (io.nickel) { state := s5 }
      when (io.dime)   { state := s10 }
    }
    is (s5) {
      when (io.nickel) { state := s10 }
      when (io.dime)   { state := s15 }
    }
    is (s10) {
      when (io.nickel) { state := s15 }
      when (io.dime)   { state := sOk }
    }
    is (s15) {
      when (io.nickel) { state := sOk }
      when (io.dime)   { state := sOk }
    }
    is (sOk) {
      state := sIdle
    }
  }
  io.dispense := (state === sOk)
}

class VerilogVendingMachine extends BlackBox {
  // Because this is a blackbox, we must explicity add clock and reset
  val io = IO(new SimpleVendingMachineIO {
    val clock = Input(Clock())
    val reset = Input(Bool())
  })
}

// Shim because Blackbox io is slightly different than normal Chisel Modules
class VerilogVendingMachineWrapper extends SimpleVendingMachine {
  val impl = Module(new VerilogVendingMachine)
  impl.io.clock := clock
  impl.io.reset := reset
  impl.io.nickel := io.nickel
  impl.io.dime := io.dime
  io.dispense := impl.io.dispense
}

// Accept a reference to a SimpleVendingMachine so it can be constructed inside
// the tester (in a call to Module.apply as required by Chisel
class SimpleVendingMachineTester(mod: => SimpleVendingMachine) extends BasicTester {

  val dut = Module(mod)

  val (cycle, done) = Counter(true.B, 10)
  when (done) { stop(); stop() } // Stop twice because of Verilator

  val nickelInputs = VecInit(true.B, true.B, true.B, true.B, true.B, false.B, false.B, false.B, true.B, false.B)
  val dimeInputs   = VecInit(false.B, false.B, false.B, false.B, false.B, true.B, true.B, false.B, false.B, true.B)
  val expected     = VecInit(false.B, false.B, false.B, false.B, true.B , false.B, false.B, true.B, false.B, false.B)

  dut.io.nickel := nickelInputs(cycle)
  dut.io.dime := dimeInputs(cycle)
  assert(dut.io.dispense === expected(cycle))
}

class SimpleVendingMachineSpec extends ChiselFlatSpec {
  "An FSM implementation of a vending machine" should "work" in {
    assertTesterPasses { new SimpleVendingMachineTester(new FSMVendingMachine) }
  }
  "An Verilog implementation of a vending machine" should "work" in {
    assertTesterPasses(new SimpleVendingMachineTester(new VerilogVendingMachineWrapper),
                       List("/chisel3/VerilogVendingMachine.v"))
  }
}
