// See LICENSE for license details.

package chisel3.iotesters

import chisel3._

/**
  * Runs the ClassicTester and returns a Boolean indicating test success or failure
  * @@backendType determines whether the ClassicTester uses verilator or the firrtl interpreter to simulate the circuit
  * Will do intermediate compliation steps to setup the backend specified, including cpp compilation for the verilator backend and firrtl IR compilation for the firrlt backend
  */
object runPeekPokeTester {
  def apply[T <: Module](dutGen: () => T, backendType: String = "firrtl")(
      testerGen: (T, Option[Backend]) => PeekPokeTester[T]): Boolean = {
    val (dut, backend) = backendType match {
      case "firrtl" => setupFirrtlTerpBackend(dutGen)
      case "verilator" => setupVerilatorBackend(dutGen)
      case "vcs" => setupVCSBackend(dutGen)
      case _ => throw new Exception("Unrecongnized backend type $backendType")
    }
    try {
      testerGen(dut, Some(backend)).finish
    } catch { case e: Throwable =>
      TesterProcess.killall
      throw e
    }
  }
}

/**
  * Runs the ClassicTester using the verilator backend without doing Verilator compilation and returns a Boolean indicating success or failure
  * Requires the caller to supply path the already compile Verilator binary
  */
object runPeekPokeTesterWithBinary {
  def apply[T <: Module] (dutGen: () => T, cmd: Seq[String])
                         (testerGen: (T, Option[Backend]) => PeekPokeTester[T]): Boolean = {
    CircuitGraph.clear
    val circuit = Driver.elaborate(dutGen)
    val dut = (CircuitGraph construct circuit).asInstanceOf[T]
    try {
      testerGen(dut, Some(new VerilatorBackend(dut, cmd))).finish
    } catch { case e: Throwable =>
      TesterProcess.killall
      throw e
    }
  }
  def apply[T <: Module] (dutGen: () => T, binary: String)
                         (testerGen: (T, Option[Backend]) => PeekPokeTester[T]): Boolean =
    apply(dutGen, Seq(binary))(testerGen)
}

