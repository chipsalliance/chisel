// SPDX-License-Identifier: Apache-2.0

package examples

import chisel3._
import chisel3.testers.{BasicTester, TesterDriver}
import chisel3.util._
import chiselTests.ChiselFlatSpec

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
  // TODO: we really need a better State Machine API for both PPA and read ability
  // General Synthesis Flow won't re-encode this.
  // User should use Mux1H to write their own high-performance FSM
  //  val sIdle = "b00001".U
  //  val s5 = "b00010".U
  //  val s10 = "b00100".U
  //  val s15 = "b01000".U
  //  val sOk = "b10000".U
  //  val decode = decoder(state ## io.nickel ## io.dime, TruthTable(
  //    """????100->000010
  //      |????110->000100
  //      |????101->001000
  //      |????111->001000
  //      |???1?00->000100
  //      |???1?10->001000
  //      |???1?01->010000
  //      |???1?11->010000
  //      |??1??00->010000
  //      |??1??10->010000
  //      |??1??01->100000
  //      |??1??11->100000
  //      |?1???00->010000
  //      |?1???10->100000
  //      |?1???01->100000
  //      |?1???11->100000
  //      |1??????->000011
  //      |?????
  //      |""".stripMargin
  //  ))
  //  state := decode(5,1)
  //  io.dispense := decode(0)

  val sIdle :: s5 :: s10 :: s15 :: sOk :: Nil = Enum(5)
  val state = RegInit(sIdle)

  when(state === sIdle) {
    when(io.nickel) { state := s5 }
    when(io.dime) { state := s10 }
  }.elsewhen(state === s5) {
    when(io.nickel) { state := s10 }
    when(io.dime) { state := s15 }
  }.elsewhen(state === s10) {
    when(io.nickel) { state := s15 }
    when(io.dime) { state := sOk }
  }.elsewhen(state === s15) {
    when(io.nickel) { state := sOk }
    when(io.dime) { state := sOk }
  }.elsewhen(state === sOk) {
    state := sIdle
  }
  io.dispense := (state === sOk)
}

class VerilogVendingMachine extends BlackBox {
  // Because this is a blackbox, we must explicitly add clock and reset
  val io = IO(new SimpleVendingMachineIO {
    val clock = Input(Clock())
    val reset = Input(Reset())
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
                       List("/chisel3/VerilogVendingMachine.v"), annotations = TesterDriver.verilatorOnly)
  }
}
