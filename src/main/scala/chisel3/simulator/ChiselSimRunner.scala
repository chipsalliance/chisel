package chisel3.simulator

import java.nio.file.Paths

/** Runner for ChiselSimSuite simulations.
  *
  * This is invoked by the generated ninja file to run a pre-compiled simulation.
  * It takes the main class name, test name, and workdir as arguments.
  *
  * The simulation binary should already be running and listening on named pipes:
  *   - workdir/cmd.pipe: for sending commands
  *   - workdir/msg.pipe: for receiving messages
  *
  * Usage:
  *   java -cp <classpath> chisel3.simulator.ChiselSimRunner <MainClassName> <testName> <workdir>
  */
object ChiselSimRunner {
  def main(args: Array[String]): Unit = {
    if (args.length < 3) {
      System.err.println("Usage: ChiselSimRunner <MainClassName> <testName> <workdir>")
      System.err.println("  MainClassName: The fully qualified name of a ChiselSimSuite object")
      System.err.println("  testName: The name/description of the test to run")
      System.err.println("  workdir: The working directory containing the simulation and pipes")
      System.exit(1)
    }

    val mainClassName = args(0)
    val testName = args(1)
    val workdir = Paths.get(args(2))
    val commandPipe = workdir.resolve("cmd.pipe")
    val messagePipe = workdir.resolve("msg.pipe")

    try {
      // Load the class and get the MODULE$ field (Scala object instance)
      val clazz = Class.forName(mainClassName + "$")
      val moduleField = clazz.getField("MODULE$")
      val instance = moduleField.get(null)

      // Check that it's a ChiselSimSuite and call runSimulation with pipes
      instance match {
        case simSuite: ChiselSimSuite[_] =>
          simSuite.runSimulationWithPipes(testName, commandPipe, messagePipe, workdir)
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
}
