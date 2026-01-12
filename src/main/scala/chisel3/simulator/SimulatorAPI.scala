// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator

import chisel3.{Module, RawModule}
import chisel3.experimental.inlinetest.{HasTests, SimulatedTest, TestChoice, TestHarness}
import chisel3.simulator.stimulus.{InlineTestStimulus, ResetProcedure}
import chisel3.testing.HasTestingDirectory
import chisel3.util.simpleClassName
import java.io.{BufferedWriter, File, FileWriter}
import java.nio.file.Files

trait SimulatorAPI {

  /** Simulate a [[RawModule]] without any initialization procedure.
    *
    * Use of this method is not advised when [[simulate]] can be used instead.
    * This method may cause problems with certain simulators as it is up to the
    * user to ensure that race conditions will not exist during the time zero
    * and reset procedure.
    *
    * @param module the Chisel module to generate
    * @param chiselOpts command line options to pass to Chisel
    * @param firtoolOpts command line options to pass to firtool
    * @param settings ChiselSim-related settings used for simulation
    * @param subdirectory an optional subdirectory for the test.  This will be a
    * subdirectory under what is provided by `testingDirectory`.
    * @param stimulus directed stimulus to use
    * @param testingDirectory a type class implementation that can be used to
    * change the behavior of where files will be created
    *
    * @note Take care when passing `chiselOpts`.  The following options are set
    * by default and if you set incompatible options, the simulation will fail.
    */
  def simulateRaw[T <: RawModule](
    module:       => T,
    chiselOpts:   Array[String] = Array.empty,
    firtoolOpts:  Array[String] = Array.empty,
    settings:     Settings[T] = Settings.defaultRaw[T],
    subdirectory: Option[String] = None
  )(stimulus: (T) => Unit)(
    implicit hasSimulator:        HasSimulator,
    testingDirectory:             HasTestingDirectory,
    chiselOptsModifications:      ChiselOptionsModifications,
    firtoolOptsModifications:     FirtoolOptionsModifications,
    commonSettingsModifications:  svsim.CommonSettingsModifications,
    backendSettingsModifications: svsim.BackendSettingsModifications
  ): Unit = {

    val modifiedTestingDirectory = subdirectory match {
      case Some(subdir) => testingDirectory.withSubdirectory(subdir)
      case None         => testingDirectory
    }

    hasSimulator
      .getSimulator(modifiedTestingDirectory)
      .simulate(module = module, chiselOpts = chiselOpts, firtoolOpts = firtoolOpts, settings = settings) { module =>
        stimulus(module.wrapped)
      }
      .result
  }

  /** Simulate a [[Module]] using a standard initialization procedure.
    *
    * For details of the initialization procedure see [[ResetProcedure]].
    *
    * @param module the Chisel module to generate
    * @param chiselOpts command line options to pass to Chisel
    * @param firtoolOpts command line options to pass to firtool
    * @param settings ChiselSim-related settings used for simulation
    * @param additionalResetCycles a number of _additional_ cycles to assert
    * reset for
    * @param subdirectory an optional subdirectory for the test.  This will be a
    * subdirectory under what is provided by `testingDirectory`.
    * @param stimulus directed stimulus to use
    * @param testingDirectory a type class implementation that can be used to
    * change the behavior of where files will be created
    *
    * @note Take care when passing `chiselOpts`.  The following options are set
    * by default and if you set incompatible options, the simulation will fail.
    */
  def simulate[T <: Module](
    module:                => T,
    chiselOpts:            Array[String] = Array.empty,
    firtoolOpts:           Array[String] = Array.empty,
    settings:              Settings[T] = Settings.default[T],
    additionalResetCycles: Int = 0,
    subdirectory:          Option[String] = None
  )(stimulus: (T) => Unit)(
    implicit hasSimulator:        HasSimulator,
    testingDirectory:             HasTestingDirectory,
    chiselOptsModifications:      ChiselOptionsModifications,
    firtoolOptsModifications:     FirtoolOptionsModifications,
    commonSettingsModifications:  svsim.CommonSettingsModifications,
    backendSettingsModifications: svsim.BackendSettingsModifications
  ): Unit = simulateRaw(
    module = module,
    chiselOpts = chiselOpts,
    firtoolOpts = firtoolOpts,
    settings = settings,
    subdirectory = subdirectory
  ) { dut =>
    ResetProcedure.module(additionalResetCycles)(dut)
    stimulus(dut)
  }

  /** Simulate the tests of a [[HasTests]] module.
    *
    * @param module the Chisel module to generate
    * @param test the choice of which test(s) to run
    * @param timeout number of cycles after which the test fails if unfinished
    * @param chiselOpts command line options to pass to Chisel
    * @param firtoolOpts command line options to pass to firtool
    * @param settings ChiselSim-related settings used for simulation
    * @param additionalResetCycles a number of _additional_ cycles to assert
    * reset for
    * @param subdirectory an optional subdirectory for the test.  This will be a
    * subdirectory under what is provided by `testingDirectory`.
    * @param stimulus directed stimulus to use
    * @param testingDirectory a type class implementation that can be used to
    * change the behavior of where files will be created
    *
    * @note Take care when passing `chiselOpts`.  The following options are set
    * by default and if you set incompatible options, the simulation will fail.
    */
  def simulateTests[T <: RawModule with HasTests](
    module:                => T,
    tests:                 TestChoice.Type,
    timeout:               Int,
    chiselOpts:            Array[String] = Array.empty,
    firtoolOpts:           Array[String] = Array.empty,
    settings:              Settings[TestHarness[T]] = Settings.defaultRaw[TestHarness[T]],
    additionalResetCycles: Int = 0,
    subdirectory:          Option[String] = None
  )(
    implicit hasSimulator:        HasSimulator,
    testingDirectory:             HasTestingDirectory,
    chiselOptsModifications:      ChiselOptionsModifications,
    firtoolOptsModifications:     FirtoolOptionsModifications,
    commonSettingsModifications:  svsim.CommonSettingsModifications,
    backendSettingsModifications: svsim.BackendSettingsModifications
  ) = {
    val modifiedTestingDirectory = subdirectory match {
      case Some(subdir) => testingDirectory.withSubdirectory(subdir)
      case None         => testingDirectory
    }

    hasSimulator
      .getSimulator(modifiedTestingDirectory)
      .simulateTests(
        module = module,
        includeTestGlobs = tests.globs,
        chiselOpts = chiselOpts,
        firtoolOpts = firtoolOpts,
        settings = settings
      ) { dut => InlineTestStimulus(timeout, additionalResetCycles, period = 10)(dut.wrapped) }
  }

  /** Export a simulation without compiling or running it.
    *
    * This generates:
    *   - CHIRRTL (.fir) file from the Chisel module
    *   - A ninja build file with rules to:
    *     - Convert .fir to SystemVerilog using firtool
    *     - Run the simulation by invoking the main class with an argument
    *
    * @param module the Chisel module to generate
    * @param mainClass the main class name that will be invoked to run the simulation
    * @param chiselOpts command line options to pass to Chisel
    * @param firtoolOpts command line options to pass to firtool
    * @param testingDirectory where files will be created
    */
  def exportSimulation[T <: Module](
    module:      => T,
    mainClass:   String,
    chiselOpts:  Array[String] = Array.empty,
    firtoolOpts: Array[String] = Array.empty
  )(
    implicit testingDirectory: HasTestingDirectory
  ): ExportedSimulation = {
    val workspacePath = testingDirectory.getDirectory.toString

    // Create workspace directories
    val workspaceDir = new File(workspacePath)
    workspaceDir.mkdirs()
    val supportArtifactsPath = s"$workspacePath/support-artifacts"
    val primarySourcesPath = s"$workspacePath/primary-sources"
    new File(supportArtifactsPath).mkdirs()
    new File(primarySourcesPath).mkdirs()

    // Elaborate the module and generate CHIRRTL
    val elaboratedCircuit = circt.stage.ChiselStage.elaborate(module, chiselOpts)
    val circuitName = elaboratedCircuit.name

    // Serialize the CHIRRTL to a file
    val firFile = new File(supportArtifactsPath, s"$circuitName.fir")
    val firWriter = new BufferedWriter(new FileWriter(firFile))
    try {
      elaboratedCircuit.lazilySerialize.foreach(firWriter.write)
    } finally {
      firWriter.close()
    }

    // Resolve firtool binary path
    val firtoolBinary = {
      val version = chisel3.BuildInfo.firtoolVersion.get
      // Create a simple logger shim for firtoolresolver
      val loggerShim = new firtoolresolver.Logger {
        def error(msg: String): Unit = System.err.println(s"[error] $msg")
        def warn(msg:  String): Unit = System.err.println(s"[warn] $msg")
        def info(msg:  String): Unit = System.out.println(s"[info] $msg")
        def debug(msg: String): Unit = () // Suppress debug
        def trace(msg: String): Unit = () // Suppress trace
      }
      val resolved = firtoolresolver.Resolve(loggerShim, version)
      resolved match {
        case Left(msg)  => throw new Exception(s"Failed to resolve firtool: $msg")
        case Right(bin) => bin.path.toString
      }
    }

    // Build firtool command arguments (use absolute paths)
    val absolutePrimarySourcesPath = new File(primarySourcesPath).getAbsolutePath
    val firtoolArgs = Seq(
      firFile.getAbsolutePath,
      "-warn-on-unprocessed-annotations",
      "-disable-annotation-unknown",
      "--split-verilog",
      s"-o=$absolutePrimarySourcesPath"
    ) ++ firtoolOpts

    // Get the classpath from the current JVM - this will be used for the java invocation
    val classpath = System.getProperty("java.class.path")

    // Generate the ninja build file
    val ninjaFile = new File(workspacePath, "build.ninja")
    val ninjaWriter = new BufferedWriter(new FileWriter(ninjaFile))
    try {
      ninjaWriter.write("# Ninja build file for ChiselSim exported simulation\n")
      ninjaWriter.write("# Run `ninja verilog` to generate SystemVerilog from FIRRTL\n")
      ninjaWriter.write("# Run `ninja simulate` to compile and run the simulation\n")
      ninjaWriter.write("\n")

      // Variables
      ninjaWriter.write(s"firtoolPath = $firtoolBinary\n")
      ninjaWriter.write(s"firtoolArgs = ${firtoolArgs.map(a => s"'$a'").mkString(" ")}\n")
      ninjaWriter.write(s"classpath = $classpath\n")
      ninjaWriter.write(s"mainClass = $mainClass\n")
      ninjaWriter.write("\n")

      // Rule to generate Verilog from FIRRTL
      // In Ninja, $varName references a variable defined above
      val firtoolPathVar = "$" + "firtoolPath"
      val firtoolArgsVar = "$" + "firtoolArgs"
      ninjaWriter.write("rule firtool\n")
      ninjaWriter.write(s"  command = $firtoolPathVar $firtoolArgsVar\n")
      ninjaWriter.write("  description = Generating SystemVerilog from FIRRTL\n")
      ninjaWriter.write("\n")

      // Rule to run the simulation using raw java invocation
      val classpathVar = "$" + "classpath"
      val mainClassVar = "$" + "mainClass"
      ninjaWriter.write("rule run_simulation\n")
      ninjaWriter.write(s"  command = java -cp '$classpathVar' $mainClassVar --run\n")
      ninjaWriter.write("  description = Running simulation\n")
      ninjaWriter.write("\n")

      // Build targets
      // The verilog target generates all .sv files from the .fir file
      ninjaWriter.write(s"build verilog: firtool\n")
      ninjaWriter.write("\n")

      // The simulate target depends on verilog and runs the simulation
      ninjaWriter.write(s"build simulate: run_simulation | verilog\n")
      ninjaWriter.write("\n")

      // Default target
      ninjaWriter.write("default verilog\n")
      ninjaWriter.write("\n")
    } finally {
      ninjaWriter.close()
    }

    ExportedSimulation(
      workspacePath = workspacePath,
      firFilePath = firFile.getAbsolutePath,
      ninjaFilePath = ninjaFile.getAbsolutePath,
      circuitName = circuitName
    )
  }

  /** Run a simulation against a pre-compiled simulation binary.
    *
    * This is typically called when the main class is invoked with an argument,
    * indicating that the simulation should be run against already-compiled artifacts.
    * This method does NOT reset the workspace, preserving pre-generated Verilog files.
    *
    * @param module the Chisel module (used to get port information)
    * @param settings ChiselSim-related settings used for simulation
    * @param additionalResetCycles a number of _additional_ cycles to assert reset for
    * @param stimulus directed stimulus to use
    * @param testingDirectory where the pre-compiled simulation artifacts are located
    */
  def runCompiledSimulation[T <: Module](
    module:                => T,
    settings:              Settings[T] = Settings.default[T],
    additionalResetCycles: Int = 0
  )(stimulus: (T) => Unit)(
    implicit hasSimulator:        HasSimulator,
    testingDirectory:             HasTestingDirectory,
    chiselOptsModifications:      ChiselOptionsModifications,
    firtoolOptsModifications:     FirtoolOptionsModifications,
    commonSettingsModifications:  svsim.CommonSettingsModifications,
    backendSettingsModifications: svsim.BackendSettingsModifications
  ): Unit = {
    // Use simulatePrecompiled which does NOT reset the workspace
    hasSimulator
      .getSimulator(testingDirectory)
      .simulatePrecompiled(module = module, settings = settings) { dut =>
        ResetProcedure.module[T](additionalResetCycles)(dut.wrapped)
        stimulus(dut.wrapped)
      }
      .result
  }
}

/** Result of exporting a simulation */
case class ExportedSimulation(
  workspacePath: String,
  firFilePath:   String,
  ninjaFilePath: String,
  circuitName:   String
)
