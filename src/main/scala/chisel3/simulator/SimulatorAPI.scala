// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator

import chisel3.{Module, RawModule}
import chisel3.experimental.inlinetest.{HasTests, SimulatedTest, TestChoice, TestHarness}
import chisel3.simulator.stimulus.{InlineTestStimulus, ResetProcedure}
import chisel3.testing.HasTestingDirectory
import chisel3.util.simpleClassName
import java.io.{BufferedWriter, File, FileWriter}
import java.nio.file.{Files, Paths}
import java.nio.file.attribute.BasicFileAttributes
import java.nio.file.{FileVisitResult, FileVisitor}
import svsim.{CommonCompilationSettings, ModuleInfo, Workspace}
import firrtl.annoSeqToSeq

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
    *   - Testbench and simulation driver files (testbench.sv, simulation-driver.cpp, c-dpi-bridge.cpp)
    *   - A ninja build file with rules to:
    *     - Convert .fir to SystemVerilog using firtool
    *     - Compile with Verilator
    *     - Run the simulation
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
    val absoluteWorkspacePath = new File(workspacePath).getAbsolutePath

    // Create an svsim Workspace - this handles directory creation
    val workspace = new Workspace(path = workspacePath)
    workspace.reset()

    // Elaborate the module to get port information
    // Use ChiselGeneratorAnnotation.elaborate which returns the DesignAnnotation with the DUT
    val outputAnnotations = chisel3.stage.ChiselGeneratorAnnotation(() => module).elaborate

    // Extract the DUT from DesignAnnotation
    val designAnnotation = outputAnnotations.collectFirst {
      case da: chisel3.stage.DesignAnnotation[_] => da.asInstanceOf[chisel3.stage.DesignAnnotation[T]]
    }.getOrElse(throw new Exception("DesignAnnotation not found after elaboration"))

    val dut = designAnnotation.design
    val circuitName = dut.name

    // Get port information from the elaborated module
    val ports = workspace.getModuleInfoPorts(dut)
    workspace.initializeModuleInfo(dut, ports.map(_._2))

    // Get the elaborated circuit for serialization
    val elaboratedCircuit = outputAnnotations.collectFirst {
      case cca: chisel3.stage.ChiselCircuitAnnotation => cca.elaboratedCircuit
    }.getOrElse(throw new Exception("ChiselCircuitAnnotation not found after elaboration"))

    // Serialize the CHIRRTL to a file
    val supportArtifactsPath = workspace.supportArtifactsPath
    val primarySourcesPath = workspace.primarySourcesPath
    val generatedSourcesPath = workspace.generatedSourcesPath
    val firFile = new File(supportArtifactsPath, s"$circuitName.fir")
    val firWriter = new BufferedWriter(new FileWriter(firFile))
    try {
      elaboratedCircuit.lazilySerialize.foreach(firWriter.write)
    } finally {
      firWriter.close()
    }

    // Generate testbench.sv, c-dpi-bridge.cpp, simulation-driver.cpp
    workspace.generateAdditionalSources(timescale = Some(CommonCompilationSettings.Timescale.default))

    // Resolve firtool binary path
    val firtoolBinary = {
      val version = chisel3.BuildInfo.firtoolVersion.get
      val loggerShim = new firtoolresolver.Logger {
        def error(msg: String): Unit = System.err.println(s"[error] $msg")
        def warn(msg:  String): Unit = System.err.println(s"[warn] $msg")
        def info(msg:  String): Unit = System.out.println(s"[info] $msg")
        def debug(msg: String): Unit = ()
        def trace(msg: String): Unit = ()
      }
      val resolved = firtoolresolver.Resolve(loggerShim, version)
      resolved match {
        case Left(msg)  => throw new Exception(s"Failed to resolve firtool: $msg")
        case Right(bin) => bin.path.toString
      }
    }

    // Resolve Verilator binary path
    val verilatorBackend = svsim.verilator.Backend.initializeFromProcessEnvironment()
    val verilatorPath = verilatorBackend.getExecutablePath

    // Build firtool command arguments
    val absolutePrimarySourcesPath = new File(primarySourcesPath).getAbsolutePath
    val firtoolArgs = Seq(
      firFile.getAbsolutePath,
      "-warn-on-unprocessed-annotations",
      "-disable-annotation-unknown",
      "--split-verilog",
      s"-o=$absolutePrimarySourcesPath"
    ) ++ firtoolOpts

    // Get the classpath from the current JVM
    val classpath = System.getProperty("java.class.path")

    // Build Verilator command arguments
    val workdirTag = "verilator"
    val workingDirectoryPath = s"$absoluteWorkspacePath/workdir-$workdirTag"

    // Create common compilation settings with include dirs
    // Use relative paths for portability (relative to workdir-verilator)
    val commonSettings = CommonCompilationSettings(
      includeDirs = Some(Seq("../primary-sources")),
      libraryPaths = Some(Seq("../primary-sources"))
      // Use default parallelism
    )

    // Get Verilator parameters
    val verilatorSettings = svsim.verilator.Backend.CompilationSettings.default
    val parameters = verilatorBackend.generateParameters(
      outputBinaryName = "simulation",
      topModuleName = Workspace.testbenchModuleName,
      // The header path needs to be relative to verilated-sources where make runs
      // ".." goes from verilated-sources to workdir-verilator
      additionalHeaderPaths = Seq(".."),
      commonSettings = commonSettings,
      backendSpecificSettings = verilatorSettings
    )

    // Save module info as JSON for later use during simulation
    // Write to both support-artifacts and workdir for flexibility
    val moduleInfoJson = {
      val portsJson = ports.map(_._2).map { p =>
        s"""{"name":"${p.name}","isSettable":${p.isSettable},"isGettable":${p.isGettable}}"""
      }.mkString("[", ",", "]")
      s"""{"name":"${circuitName}","ports":$portsJson}"""
    }

    val supportModuleInfoFile = new File(supportArtifactsPath, "module-info.json")
    val supportModuleInfoWriter = new BufferedWriter(new FileWriter(supportModuleInfoFile))
    try {
      supportModuleInfoWriter.write(moduleInfoJson)
    } finally {
      supportModuleInfoWriter.close()
    }

    // Create the working directory and also write module info there
    new File(workingDirectoryPath).mkdirs()
    val workdirModuleInfoFile = new File(workingDirectoryPath, "module-info.json")
    val workdirModuleInfoWriter = new BufferedWriter(new FileWriter(workdirModuleInfoFile))
    try {
      workdirModuleInfoWriter.write(moduleInfoJson)
    } finally {
      workdirModuleInfoWriter.close()
    }

    // Generate the ninja build file
    val ninjaFile = new File(workspacePath, "build.ninja")
    val ninjaWriter = new BufferedWriter(new FileWriter(ninjaFile))
    try {
      def l(s: String): Unit = { ninjaWriter.write(s); ninjaWriter.write("\n") }
      def quoteForNinja(s: String): String = s"'${s.replace("$", "$$")}'"

      l("# Ninja build file for ChiselSim exported simulation")
      l("# Run `ninja verilog` to generate SystemVerilog from FIRRTL")
      l("# Run `ninja verilate` to compile with Verilator")
      l("# Run `ninja simulate` to run the simulation")
      l("")

      // Variables
      l(s"firtoolPath = $firtoolBinary")
      l(s"firtoolArgs = ${firtoolArgs.map(quoteForNinja).mkString(" ")}")
      l(s"verilatorPath = $verilatorPath")
      val verilatorArgs = parameters.getCompilerArguments.map(quoteForNinja).mkString(" ")
      l(s"verilatorArgs = $verilatorArgs")
      l(s"classpath = $classpath")
      l(s"mainClass = $mainClass")
      l(s"workdir = workdir-$workdirTag")
      l("")

      // Rule to generate Verilog from FIRRTL
      l("rule firtool")
      l("  command = $firtoolPath $firtoolArgs && touch $out")
      l("  description = Generating SystemVerilog from FIRRTL")
      l("")

      // Rule to compile with Verilator
      // Note: touch uses ../$out because we cd into $workdir first
      l("rule verilator")
      l(s"  command = cd $$workdir && $$verilatorPath $$verilatorArgs -F sourceFiles.F && touch .verilator.stamp")
      l("  description = Compiling with Verilator")
      l("")

      // Rule to run the simulation with IPC via named pipes
      // This creates FIFOs, launches the simulation binary in background,
      // then runs the Scala test which connects to the pipes
      // Note: In ninja, $$ escapes to $, so $$! becomes $! (shell's background PID)
      // Run simulation in background with full path, redirect all output to prevent blocking
      l("rule run_simulation")
      l("  command = " +
        "rm -f $workdir/cmd.pipe $workdir/msg.pipe && " +
        "mkfifo $workdir/cmd.pipe $workdir/msg.pipe && " +
        "SVSIM_COMMAND_PIPE=$workdir/cmd.pipe SVSIM_MESSAGE_PIPE=$workdir/msg.pipe $workdir/simulation > $workdir/simulation.log 2>&1 & " +
        "SIM_PID=$$!; " +
        "java -cp '$classpath' chisel3.simulator.ChiselSimRunner $mainClass $workdir; " +
        "RESULT=$$?; kill $$SIM_PID 2>/dev/null; rm -f $workdir/cmd.pipe $workdir/msg.pipe; exit $$RESULT")
      l("  description = Running simulation")
      l("  pool = console")
      l("")

      // Build targets
      val firtoolStamp = "primary-sources/.firtool.stamp"
      val verilatorStamp = s"workdir-$workdirTag/.verilator.stamp"
      val relativeFirFile = s"support-artifacts/$circuitName.fir"

      l(s"build $firtoolStamp: firtool $relativeFirFile")
      l("")

      l(s"build verilog: phony $firtoolStamp")
      l("")

      l(s"build $verilatorStamp: verilator | verilog")
      l("")

      l(s"build verilate: phony $verilatorStamp")
      l("")

      l(s"build simulate: run_simulation | verilate")
      l("")

      l("default verilog")
      l("")
    } finally {
      ninjaWriter.close()
    }

    // Write sourceFiles.F for Verilator
    val sourceFilesF = new File(workingDirectoryPath, "sourceFiles.F")
    val sourceFilesWriter = new BufferedWriter(new FileWriter(sourceFilesF))
    try {
      // Add relative paths to source files (relative to workdir-verilator)
      // The generated sources are in ../generated-sources relative to workdir-verilator
      sourceFilesWriter.write("# Source files for Verilator compilation\n")
      sourceFilesWriter.write("../generated-sources/testbench.sv\n")
      sourceFilesWriter.write("../generated-sources/simulation-driver.cpp\n")
      sourceFilesWriter.write("../generated-sources/c-dpi-bridge.cpp\n")
      // Primary sources (Verilog from firtool) will be added via -y flag in verilator args
    } finally {
      sourceFilesWriter.close()
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

  /** Run a simulation using named pipes for IPC with a pre-running simulation binary.
    *
    * This method connects to named pipes where the simulation binary is already
    * listening, runs the test stimulus, and sends a shutdown command when done.
    *
    * @param module the Chisel module (used to get port information via elaboration)
    * @param commandPipe path to the command pipe (write to send commands)
    * @param messagePipe path to the message pipe (read to receive messages)
    * @param workdir path to the working directory containing simulation artifacts
    * @param additionalResetCycles a number of _additional_ cycles to assert reset for
    * @param stimulus directed stimulus to use
    * @param testingDirectory where the pre-compiled simulation artifacts are located
    */
  def runCompiledSimulationWithPipes[T <: Module](
    module:                => T,
    commandPipe:           java.nio.file.Path,
    messagePipe:           java.nio.file.Path,
    workdir:               java.nio.file.Path,
    additionalResetCycles: Int = 0
  )(stimulus: (T) => Unit)(
    implicit testingDirectory: HasTestingDirectory
  ): Unit = {
    // Load module info from the JSON file
    val moduleInfoFile = workdir.resolve("module-info.json").toFile
    val moduleInfoJson = scala.io.Source.fromFile(moduleInfoFile).mkString
    val moduleInfo = parseModuleInfo(moduleInfoJson)

    // Create a workspace to get port mappings
    val workspace = new svsim.Workspace(
      path = testingDirectory.getDirectory.toString,
      workingDirectoryPrefix = "workdir"
    )

    // Elaborate the module to get port information (without generating Verilog)
    val outputAnnotations = chisel3.stage.ChiselGeneratorAnnotation(() => module).elaborate

    // Extract the DUT from DesignAnnotation
    val designAnnotation = outputAnnotations.collectFirst {
      case da: chisel3.stage.DesignAnnotation[_] => da.asInstanceOf[chisel3.stage.DesignAnnotation[T]]
    }.getOrElse(throw new Exception("DesignAnnotation not found after elaboration"))

    val dut = designAnnotation.design
    val layers = designAnnotation.layers
    val ports = workspace.getModuleInfoPorts(dut)

    // Create an ElaboratedModule for the SimulatedModule
    val elaboratedModule = new ElaboratedModule(dut, ports, layers)

    // Run the simulation with pipes using the static method
    svsim.Simulation.runWithPipes(commandPipe, messagePipe, moduleInfo) { controller =>
      val simModule = new SimulatedModule(elaboratedModule, controller)

      AnySimulatedModule.withValue(simModule) {
        // Apply reset procedure
        ResetProcedure.module[T](additionalResetCycles)(simModule.wrapped)

        // Run the test stimulus
        stimulus(simModule.wrapped)

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

    val name = namePattern.findFirstMatchIn(json).map(_.group(1)).getOrElse("unknown")
    val portsJson = portsPattern.findFirstMatchIn(json).map(_.group(1)).getOrElse("")

    val ports = portPattern.findAllMatchIn(portsJson).map { m =>
      svsim.ModuleInfo.Port(
        name = m.group(1),
        isSettable = m.group(2) == "true",
        isGettable = m.group(3) == "true"
      )
    }.toSeq

    svsim.ModuleInfo(name = name, ports = ports)
  }
}

/** Result of exporting a simulation */
case class ExportedSimulation(
  workspacePath: String,
  firFilePath:   String,
  ninjaFilePath: String,
  circuitName:   String
)
