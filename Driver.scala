// See LICENSE for license details.

package chisel3.iotesters

import chisel3.Module
import scala.util.DynamicVariable

object Driver {
  private val backendVar = new DynamicVariable[Option[Backend]](None)
  private[iotesters] def backend = backendVar.value

  /**
    * Runs the ClassicTester and returns a Boolean indicating test success or failure
    * @@backendType determines whether the ClassicTester uses verilator or the firrtl interpreter to simulate the circuit
    * Will do intermediate compliation steps to setup the backend specified, including cpp compilation for the verilator backend and firrtl IR compilation for the firrlt backend
    */
  def apply[T <: Module](dutGen: () => T, backendType: String = "firrtl")(
      testerGen: T => PeekPokeTester[T]): Boolean = {
    val (dut, backend) = backendType match {
      case "firrtl" => setupFirrtlTerpBackend(dutGen)
      case "verilator" => setupVerilatorBackend(dutGen)
      case "vcs" => setupVCSBackend(dutGen)
      case _ => throw new Exception("Unrecongnized backend type $backendType")
    }
    backendVar.withValue(Some(backend)) {
      try {
        testerGen(dut).finish
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
  def run[T <: Module] (dutGen: () => T, cmd: Seq[String])
                       (testerGen: T => PeekPokeTester[T]): Boolean = {
    val circuit = chisel3.Driver.elaborate(dutGen)
    val dut = getTopModule(circuit).asInstanceOf[T]
    backendVar.withValue(Some(new VerilatorBackend(dut, cmd))) {
      try {
        testerGen(dut).finish
      } catch { case e: Throwable =>
        TesterProcess.killall
        throw e
      }
    }
  }

  def run[T <: Module] (dutGen: () => T, binary: String)
                       (testerGen: T => PeekPokeTester[T]): Boolean =
    run(dutGen, Seq(binary))(testerGen)

  def run[T <: Module](dutGen: () => T)(testerGen: T => PeekPokeTester[T]): Boolean = {
    val circuit = chisel3.Driver.elaborate(dutGen)
    val dut = getTopModule(circuit).asInstanceOf[T]
    try {
      testerGen(dut).finish
    } catch { case e: Throwable =>
      TesterProcess.killall
      throw e
    }
  }
}
