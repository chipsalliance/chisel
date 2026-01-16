// SPDX-License-Identifier: Apache-2.0

package svsim

import java.io.File
import java.nio.file.{Path, Paths}

/** Runner for pre-compiled simulations.
  *
  * This is invoked by the generated ninja file to run a pre-compiled simulation.
  * It launches the simulation binary as a subprocess and uses reflection to call
  * into the test suite class.
  *
  * Usage:
  *   java -cp <classpath> svsim.SimulationRunner <simulationBinary> <mainClass> <testName>
  *
  * Where:
  *   - simulationBinary: Full path to the simulation binary
  *   - mainClass: The fully qualified name of a ChiselSimSuite object
  *   - testName: The name/description of the test to run
  *
  * The workdir is derived from the parent directory of the simulation binary.
  * The module-info.json file is expected to be in the same directory as the binary.
  */
object SimulationRunner {
  def main(args: Array[String]): Unit = {
    if (args.length < 3) {
      System.err.println("Usage: SimulationRunner <simulationBinary> <mainClass> <testName>")
      System.err.println("  simulationBinary: Full path to the simulation binary")
      System.err.println("  mainClass: The fully qualified name of a ChiselSimSuite object")
      System.err.println("  testName: The name/description of the test to run")
      System.exit(1)
    }

    val simulationBinary = Paths.get(args(0))
    val mainClassName = args(1)
    val testName = args(2)

    // Derive workdir from the simulation binary's parent directory
    val workdir = simulationBinary.getParent
    if (workdir == null) {
      System.err.println(s"Error: Cannot determine workdir from simulation binary path: $simulationBinary")
      System.exit(1)
    }

    try {
      // Load the test suite class and call runSimulationDirectly via reflection
      val clazz = Class.forName(mainClassName + "$")
      val moduleField = clazz.getField("MODULE$")
      val instance = moduleField.get(null)

      // Use reflection to call runSimulationDirectly(testName, simulationBinary, workdir)
      // This avoids a compile-time dependency on chisel3.simulator
      val method = instance.getClass.getMethod(
        "runSimulationDirectly",
        classOf[String],
        classOf[Path],
        classOf[Path]
      )

      println(s"Running test: $testName")
      method.invoke(instance, testName, simulationBinary, workdir)

    } catch {
      case e: ClassNotFoundException =>
        System.err.println(s"Error: Could not find class $mainClassName")
        System.err.println(s"  ${e.getMessage}")
        System.exit(1)
      case e: NoSuchFieldException =>
        System.err.println(s"Error: $mainClassName does not appear to be a Scala object")
        System.err.println(s"  ${e.getMessage}")
        System.exit(1)
      case e: NoSuchMethodException =>
        System.err.println(s"Error: $mainClassName does not have the expected runSimulationDirectly method")
        System.err.println(s"  ${e.getMessage}")
        System.exit(1)
      case e: java.lang.reflect.InvocationTargetException =>
        // Unwrap the actual exception from the reflection call
        val cause = e.getCause
        if (cause != null) {
          System.err.println(s"Error running simulation: ${cause.getMessage}")
          cause.printStackTrace()
        } else {
          System.err.println(s"Error running simulation: ${e.getMessage}")
          e.printStackTrace()
        }
        System.exit(1)
      case e: Exception =>
        System.err.println(s"Error running simulation: ${e.getMessage}")
        e.printStackTrace()
        System.exit(1)
    }
  }
}

