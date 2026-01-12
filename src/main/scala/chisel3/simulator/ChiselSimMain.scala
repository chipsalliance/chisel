package chisel3.simulator

import chisel3._
import chisel3.experimental.hierarchy._
import chisel3.testing.HasTestingDirectory

/** Base class for ChiselSim main functions that support two-phase simulation:
  * 1. Export phase (no args): Generates .fir file and ninja build file
  * 2. Run phase (with --run arg): Runs the simulation against compiled artifacts
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
abstract class ChiselSimMain[T <: Module](gen: => T) extends ControlAPI with PeekPokeAPI with SimulatorAPI {
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

  final def main(args: Array[String]): Unit = {
    implicit val testingDirectory: HasTestingDirectory = testdir
    if (args.isEmpty || !args.contains("--run")) {
      // Export phase: Generate .fir file and ninja build file
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
    } else {
      // Run phase: Run the simulation with pre-compiled artifacts
      runCompiledSimulation(gen)(test)
    }
  }
}
