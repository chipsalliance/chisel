// See LICENSE for license details.

package chisel3.testers2

import chisel3._

class FirrterpreterBackend(inst: FirrterpreterInstance) extends TesterBackend {
  def poke(signal: Data, value: BigInt): Unit = {

  }

  def peek(signal: Data): BigInt = {

  }
  def stalePeek(signal: Data): BigInt = {

  }

  def check(signal: Data, value: BigInt): BigInt = {

  }
  def staleCheck(signal: Data, value: BigInt): BigInt = {

  }

  def step(signal: Clock, cycles: Int): Unit = {

  }
}

class FirrterpreterInstance(dut: Module) {

}

case class TestErrorEntry(
  val msg: String,
  val loc: Seq[StackTraceElement]
)

object Firrterpreter {
  import chisel3.internal.firrtl.Circuit
  import chisel3.experimental.BaseModule

  import firrtl._

  import firrtl_interpreter._

  def getTopModule(circuit: Circuit): BaseModule = {
    (circuit.components find (_.name == circuit.name)).get.id
  }

  def runTest[T <: Module](dutGen: () => T)(test: T => Unit) {
    val optionsManager = new ExecutionOptionsManager("chisel3")
        with HasChiselExecutionOptions with HasFirrtlOptions with HasInterpreterSuite {
      firrtlOptions = FirrtlExecutionOptions(
        compilerName = "low"
      )
    }

    chisel3.Driver.execute(optionsManager, dutGen) match {
      case ChiselExecutionSuccess(Some(circuit), _, Some(firrtlExecutionResult)) =>
        firrtlExecutionResult match {
          case FirrtlExecutionSuccess(_, compiledFirrtl) =>
            val dut = getTopModule(circuit).asInstanceOf[T]
            val interpretiveTester = new InterpretiveTester(compiledFirrtl, optionsManager)

          case FirrtlExecutionFailure(message) =>
            throw new Exception(s"FirrtlBackend: failed firrtl compile message: $message")
        }
      case _ =>
        throw new Exception("Problem with compilation")
    }
  }
}
