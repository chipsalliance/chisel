package chisel3.simulator

import chisel3._
import chisel3.experimental.hierarchy._
import chisel3.testing.HasTestingDirectory

abstract class ChiselSimMain[T <: Module](gen: => T) extends ControlAPI with PeekPokeAPI with SimulatorAPI { self: Singleton =>

  def mainClass: String = self.getClass.getName
  
  def testdir: HasTestingDirectory = HasTestingDirectory.default

  // User must implement
  def test(dut: T): Unit

  final def main(args: Array[String]) = {
    implicit val testingDirectory = testdir
    if (args.length == 0) {
      // Except we don't want to simulate, we want to generate .fir and the ninja file
      simulate(gen)(test)
      // TODO: This should be something like:
      // exportSimulation(gen)
    } else {
      // This should be what is invoked by the generated ninja, it should run the test function
      // against the pre-compiled simulation.
      ???
      // TODO: This should be something like:
      // runCompiledSimulation(gen)(test)
    }
  }
}
