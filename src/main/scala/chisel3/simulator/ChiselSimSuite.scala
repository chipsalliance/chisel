package chisel3.simulator

import chisel3._
import chisel3.experimental.hierarchy._
import chisel3.simulator.stimulus.ResetProcedure
import chisel3.testing.HasTestingDirectory
import firrtl.annoSeqToSeq
import java.nio.file.Path
import java.io.File
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

  /** Run a specific test by name with pre-compiled artifacts.
    * Called by svsim.SimulationRunner when invoked from ninja.
    *
    * This method launches the simulation binary as a subprocess and manages the IPC
    * directly, making it cross-platform compatible (no named pipes).
    *
    * @param testName the name/description of the test to run
    * @param simulationBinary full path to the simulation binary
    * @param workdir the working directory containing the simulation artifacts (module-info.json, etc.)
    */
  def runSimulationDirectly(testName: String, simulationBinary: Path, workdir: Path): Unit = {
    val testEntry = _tests.find(_._1 == testName)
    if (testEntry.isEmpty) {
      val available = _tests.map(_._1).mkString("'", "', '", "'")
      throw new IllegalArgumentException(s"Test '$testName' not found. Available tests: $available")
    }
    val (_, testFn) = testEntry.get

    // Get the parent directory (workspace root), handling relative paths
    val parentDir = Option(workdir.toAbsolutePath.getParent).getOrElse(workdir.toAbsolutePath)
    implicit val testingDirectory: HasTestingDirectory = new HasTestingDirectory {
      override def getDirectory: java.nio.file.Path = parentDir
    }

    // Load module info from the JSON file
    val moduleInfoFile = workdir.resolve("module-info.json").toFile
    val moduleInfoJson = scala.io.Source.fromFile(moduleInfoFile).mkString
    val moduleInfo = parseModuleInfo(moduleInfoJson)

    // Create a workspace to get port mappings
    val workspace = new svsim.Workspace(
      path = parentDir.toString,
      workingDirectoryPrefix = "workdir"
    )

    // Create a Simulation directly with the binary path
    val simulation = svsim.Simulation.fromBinary(
      binaryPath = simulationBinary,
      workingDirectoryPath = workdir.toAbsolutePath.toString,
      moduleInfo = moduleInfo
    )

    // Elaborate the module to get port information (without generating Verilog)
    val outputAnnotations = chisel3.stage.ChiselGeneratorAnnotation(() => gen).elaborate

    // Extract the DUT from DesignAnnotation
    val designAnnotation = outputAnnotations.collectFirst {
      case da: chisel3.stage.DesignAnnotation[_] => da.asInstanceOf[chisel3.stage.DesignAnnotation[T]]
    }.getOrElse(throw new Exception("DesignAnnotation not found after elaboration"))

    val dut = designAnnotation.design
    val layers = designAnnotation.layers
    val ports = workspace.getModuleInfoPorts(dut)

    // Create an ElaboratedModule for the SimulatedModule
    val elaboratedModule = new ElaboratedModule(dut, ports, layers)

    // Run the simulation
    simulation.run() { controller =>
      val simModule = new SimulatedModule(elaboratedModule, controller)

      AnySimulatedModule.withValue(simModule) {
        // Apply reset procedure
        ResetProcedure.module[T](0)(simModule.wrapped)

        // Run the test stimulus
        testFn(simModule.wrapped)

        // Complete the simulation
        simModule.completeSimulation()
      }
    }
  }

  /** Parse a module info JSON string into a ModuleInfo object */
  private def parseModuleInfo(json: String): svsim.ModuleInfo = {
    // Simple JSON parser for the format: {"name":"...", "ports":[...]}
    val namePattern = """"name"\s*:\s*"([^"]+)"""".r
    val portsPattern = """"ports"\s*:\s*\[([^\]]*)\]""".r
    val portPattern = """\{"name"\s*:\s*"([^"]+)"\s*,\s*"isSettable"\s*:\s*(true|false)\s*,\s*"isGettable"\s*:\s*(true|false)\}""".r

    val name = namePattern.findFirstMatchIn(json).map(_.group(1)).getOrElse("")
    val portsSection = portsPattern.findFirstMatchIn(json).map(_.group(1)).getOrElse("")
    val ports = portPattern.findAllMatchIn(portsSection).map { m =>
      svsim.ModuleInfo.Port(
        name = m.group(1),
        isSettable = m.group(2) == "true",
        isGettable = m.group(3) == "true"
      )
    }.toSeq

    svsim.ModuleInfo(name, ports)
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
