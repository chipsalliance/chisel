// See LICENSE for license details.

package chisel3.testers2

import chisel3._

class FirrterpreterBackend(inst: FirrterpreterInstance) extends Backend {
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
  def runTest[T <: Module](dutGen: () => T, test: T => Seq[TestErrorEntry]) {

  }
}
