package chiselTest

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class ResetTypeTests extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "resets"

  it should "behave correctly with a synchronous reset" in {
    test(new SimpleResetTestCircuit(false) with RequireSyncReset) { dut =>
      // the test bench automatically resets the circuit
      dut.out.expect(123.U)

      dut.in.poke(213.U)
      dut.clock.step()
      dut.out.expect(213.U)

      // reset becomes active only after a clock step
      dut.reset.poke(true.B)
      dut.out.expect(213.U)
      dut.clock.step()
      dut.out.expect(123.U)
    }
  }

  // TODO: add test for RequiresAsyncRest once chiseltest can deal with asynchronous resets

  it should "fail the assertion with a synchronous reset" in {
    assertThrows[ChiselAssertionError] {
      test(new SimpleResetTestCircuit(true) with RequireSyncReset) { _ =>
        // the implicit reset in the simulator will trigger the assertion!
      }
    }
  }

  it should "behave correctly with a reset as initial value" in {
    test(new SimpleResetTestCircuit(true) with RequireResetIsInitialValue) { _ => }
  }

  it should "not trigger the assertion when using withResetIsInitialValue" in {
    test(new SimpleWithInitialValueResetTestCircuit) { _ => }
  }
}

class SimpleResetTestCircuit(enableAssertion: Boolean) extends Module {
  val reg = RegInit(123.U(8.W))
  val in = IO(Input(UInt(8.W)))
  reg := in
  val out = IO(Output(UInt(8.W)))
  out := reg

  if(enableAssertion) {
    withReset(false.B) {
      assert(reg === 123.U, "reg is %d not 123", reg)
    }
  }
}


class SimpleWithInitialValueResetTestCircuit extends Module {
  val reg = withResetIsInitialValue { RegInit(123.U(8.W)) }
  val in = IO(Input(UInt(8.W)))
  reg := in
  val out = IO(Output(UInt(8.W)))
  out := reg

  withReset(false.B) {
    assert(reg === 123.U, "reg is %d not 123", reg)
  }
}

