package chisel3.simulator

import chisel3._
import chisel3.experimental.hierarchy._
import chisel3.testing.HasTestingDirectory
import java.nio.file.Path
import scala.collection.mutable

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

  private val _tests = mutable.ArrayBuffer.empty[(String, T => Unit)]

  /** Register a test with a description */
  def test(desc: String)(f: T => Unit): Unit = {
    _tests += ((desc, f))
  }

  /** Get the list of registered tests (description, function pairs) */
  def tests: Seq[(String, T => Unit)] = _tests.toSeq

  /** Run a specific test by index with pre-compiled artifacts using named pipes for IPC.
    * Called by ChiselSimRunner when invoked from ninja.
    *
    * The simulation binary should already be running and listening on the pipes.
    * This method connects to the pipes, runs the test, and sends shutdown command.
    *
    * @param testIndex the 0-based index of the test to run
    * @param commandPipe path to the command pipe (for sending commands to simulation)
    * @param messagePipe path to the message pipe (for receiving messages from simulation)
    * @param workdir the working directory containing the simulation
    */
  def runSimulationWithPipes(testIndex: Int, commandPipe: Path, messagePipe: Path, workdir: Path): Unit = {
    if (testIndex < 0 || testIndex >= _tests.size) {
      throw new IllegalArgumentException(s"Test index $testIndex out of range (0..${_tests.size - 1})")
    }
    val (testName, testFn) = _tests(testIndex)
    println(s"Running test $testIndex: $testName")

    // Get the parent directory, handling relative paths that may not have a parent
    val parentDir = Option(workdir.toAbsolutePath.getParent).getOrElse(workdir.toAbsolutePath)
    implicit val testingDirectory: HasTestingDirectory = new HasTestingDirectory {
      override def getDirectory: java.nio.file.Path = parentDir
    }
    runCompiledSimulationWithPipes(gen, commandPipe, messagePipe, workdir)(testFn)
  }

  final def main(args: Array[String]): Unit = {
    // Export phase: Generate .fir file and ninja build file
    implicit val testingDirectory: HasTestingDirectory = testdir
    val testDescriptions = _tests.map(_._1).toSeq
    val exported = exportSimulation(gen, mainClass, testDescriptions)
    println(s"Exported simulation to: ${exported.workspacePath}")
    println(s"  FIRRTL file: ${exported.firFilePath}")
    println(s"  Ninja file:  ${exported.ninjaFilePath}")
    println()
    println("To generate Verilog, run:")
    println(s"  ninja -C ${exported.workspacePath} verilog")
    println()
    println("To compile the simulation, run:")
    println(s"  ninja -C ${exported.workspacePath} verilate")
    println()
    if (testDescriptions.nonEmpty) {
      println("To run individual tests:")
      testDescriptions.zipWithIndex.foreach { case (desc, i) =>
        println(s"  ninja -C ${exported.workspacePath} test${i + 1}   # $desc")
      }
      println()
      println("To run all tests:")
      println(s"  ninja -C ${exported.workspacePath} testAll")
    }
  }
}
