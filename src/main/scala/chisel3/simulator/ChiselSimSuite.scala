package chisel3.simulator

import chisel3._
import chisel3.experimental.hierarchy._
import chisel3.testing.HasTestingDirectory

/** Base class for ChiselSim main functions that export simulation artifacts.
  *
  * This generates a .fir file and ninja build file. The ninja file can then be used
  * to compile and run the simulation via ChiselSimRunner.
  *
  * Example usage:
  * {{{
  * object MySimMain extends ChiselSimMain(new MyModule) {
  *   override def testdir = new HasTestingDirectory {
  *     override def getDirectory = Paths.get("my-testdir")
  *   }
  *
  *   def test(dut: MyModule): Unit = {
  *     dut.io.in.poke(42)
  *     dut.clock.step()
  *     dut.io.out.expect(42)
  *   }
  * }
  * }}}
  *
  * Then run:
  *   ./mill chisel[2.13].runMain MySimMain          # Export phase
  *   ninja -C my-testdir simulate                   # Compile and run
  */
abstract class ChiselSimSuite[T <: Module](gen: => T) extends ControlAPI with PeekPokeAPI with SimulatorAPI {
  self: Singleton =>

  /** The main class name, used in the generated ninja file to invoke the run phase */
  def mainClass: String = {
    val name = self.getClass.getName
    // Remove trailing $ from Scala object names
    if (name.endsWith("$")) name.dropRight(1) else name
  }

  /** Override to customize the test directory */
  def testdir: HasTestingDirectory = HasTestingDirectory.default

  /** User must implement the test stimulus */
  def test(dut: T): Unit

  /** Run the simulation with pre-compiled artifacts.
    * Called by ChiselSimRunner when invoked from ninja.
    */
  def runSimulation(): Unit = {
    // Use current directory since ninja runs from within the workspace
    implicit val testingDirectory: HasTestingDirectory = new HasTestingDirectory {
      override def getDirectory: java.nio.file.Path = java.nio.file.Paths.get(".")
    }
    runCompiledSimulation(gen)(test)
  }

  final def main(args: Array[String]): Unit = {
    // Export phase: Generate .fir file and ninja build file
    implicit val testingDirectory: HasTestingDirectory = testdir
    val exported = exportSimulation(gen, mainClass)
    println(s"Exported simulation to: ${exported.workspacePath}")
    println(s"  FIRRTL file: ${exported.firFilePath}")
    println(s"  Ninja file:  ${exported.ninjaFilePath}")
    println()
    println("To generate Verilog, run:")
    println(s"  ninja -C ${exported.workspacePath} verilog")
    println()
    println("To compile and run the simulation, run:")
    println(s"  ninja -C ${exported.workspacePath} simulate")
  }
}
