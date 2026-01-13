package chisel3.simulator

/** Runner for ChiselSimMain simulations.
  *
  * This is invoked by the generated ninja file to run a pre-compiled simulation.
  * It takes the main class name as an argument and calls its runSimulation method.
  *
  * Usage:
  *   java -cp <classpath> chisel3.simulator.ChiselSimRunner <MainClassName>
  */
object ChiselSimRunner {
  def main(args: Array[String]): Unit = {
    if (args.isEmpty) {
      System.err.println("Usage: ChiselSimRunner <MainClassName>")
      System.err.println("  MainClassName: The fully qualified name of a ChiselSimMain object")
      System.exit(1)
    }

    val mainClassName = args(0)

    try {
      // Load the class and get the MODULE$ field (Scala object instance)
      val clazz = Class.forName(mainClassName + "$")
      val moduleField = clazz.getField("MODULE$")
      val instance = moduleField.get(null)

      // Check that it's a ChiselSimSuite and call runSimulation
      instance match {
        case simMain: ChiselSimSuite[_] =>
          simMain.runSimulation()
        case _ =>
          System.err.println(s"Error: $mainClassName is not a ChiselSimMain")
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
      case e: Exception =>
        System.err.println(s"Error running simulation: ${e.getMessage}")
        e.printStackTrace()
        System.exit(1)
    }
  }
}
