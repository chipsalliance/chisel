package chisel3.simulator

import java.io.{BufferedReader, BufferedWriter, File, InputStreamReader, OutputStreamWriter}
import java.nio.file.{Path, Paths}

/** Runner for ChiselSimSuite simulations.
  *
  * This is invoked by the generated ninja file to run a pre-compiled simulation.
  * It directly launches the simulation binary as a subprocess and manages the IPC
  * through stdin/stdout (not named pipes), making it cross-platform compatible.
  *
  * Usage:
  *   java -cp <classpath> chisel3.simulator.SimulationRunner <workdir> <mainClass> <testName>
  *
  * Where:
  *   - workdir: The working directory containing the simulation binary and artifacts
  *   - mainClass: The fully qualified name of a ChiselSimSuite object
  *   - testName: The name/description of the test to run
  */
object SimulationRunner {
  def main(args: Array[String]): Unit = {
    if (args.length < 3) {
      System.err.println("Usage: SimulationRunner <workdir> <mainClass> <testName>")
      System.err.println("  workdir: The working directory containing the simulation binary")
      System.err.println("  mainClass: The fully qualified name of a ChiselSimSuite object")
      System.err.println("  testName: The name/description of the test to run")
      System.exit(1)
    }

    val workdir = Paths.get(args(0))
    val mainClassName = args(1)
    val testName = args(2)

    try {
      // Load the ChiselSimSuite class and get the test function
      val clazz = Class.forName(mainClassName + "$")
      val moduleField = clazz.getField("MODULE$")
      val instance = moduleField.get(null)

      instance match {
        case simSuite: ChiselSimSuite[_] =>
          runTest(simSuite, testName, workdir)
        case _ =>
          System.err.println(s"Error: $mainClassName is not a ChiselSimSuite")
          System.exit(1)
      }
    } catch {
      case e: ClassNotFoundException =>
        System.err.println(s"Error: Could not find class $mainClassName")
        System.err.println(s"  ${e.getMessage}")
        System.exit(1)
      case e: NoSuchFieldException =>
        System.err.println(s"Error: $mainClassName does not appear to be a Scala object")
        System.err.println(s"  ${e.getMessage}")
        System.exit(1)
      case e: IllegalArgumentException =>
        System.err.println(s"Error: ${e.getMessage}")
        System.exit(1)
      case e: Exception =>
        System.err.println(s"Error running simulation: ${e.getMessage}")
        e.printStackTrace()
        System.exit(1)
    }
  }

  private def runTest[T <: chisel3.Module](
    simSuite: ChiselSimSuite[T],
    testName: String,
    workdir:  Path
  ): Unit = {
    println(s"Running test: $testName")

    // Get the test function from the suite
    val testEntry = simSuite.tests.find(_._1 == testName)
    if (testEntry.isEmpty) {
      val available = simSuite.tests.map(_._1).mkString("'", "', '", "'")
      throw new IllegalArgumentException(s"Test '$testName' not found. Available tests: $available")
    }
    val (_, testFn) = testEntry.get

    // Run the simulation using the suite's infrastructure
    simSuite.runSimulationDirectly(testName, workdir)
  }
}

