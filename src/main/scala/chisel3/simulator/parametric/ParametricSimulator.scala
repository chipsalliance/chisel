package chisel3.simulator.parametric

import svsim._
import chisel3.RawModule
import chisel3.simulator.{PeekPokeAPI, SingleBackendSimulator}

import java.nio.file.Files
import java.time.{format, LocalDateTime}

/**
  * The interface object to the simulation.
  *
  * It uses [[svsim]] and [[PeekPokeAPI]] to run a simulation.
  */
object ParametricSimulator extends PeekPokeAPI {

  // private var simulator = new ParametricSimulator

  /** If true, the simulator will be reset before running each simulation */
  private var _resetSimulationBeforeRun = false

  /** Use this method to run a simulations */
  def simulate[T <: RawModule](
    module:   => T,
    settings: Seq[SimulatorSettings] = Seq(),
    simName:  String = "defaultSimulation"
  )(body:     T => Unit
  ): Unit = {
    //    if (_resetSimulationBeforeRun)
    //      reset()
    //    simulator.simulate(module, settings, simName)(body)

    // TODO: check if the simulation should be reset for every run
    (new ParametricSimulator).simulate(module, settings, simName)(body)
  }

  /**
    * Use this method to manually reset the simulator and run multiple
    * independent simulations
    */
  //  def reset(): Unit =
  //    simulator = new ParametricSimulator

  def resetBeforeEachRun(): Unit =
    _resetSimulationBeforeRun = true
}

/**
  * A simulator that can be customized by passing some parameters through a list
  * of [[SimulatorSettings]]. It offers the possibility to:
  *   - output a trace vcd file
  *   - store the files generated during the simulation
  *
  *  More advanced simulators can be implemented starting from this.
  */
class ParametricSimulator {

  // Settings variable of the simulator
  private var _backendCompileSettings = verilator.Backend.CompilationSettings()
  private val _testRunDir = "test_run_dir"
  private var _moduleDutName = ""
  private val _simulatorName = getClass.getSimpleName.stripSuffix("$")
  private var _simName = "defaultSimulationName"

  // Working directory and workspace
  private var _workdirFinalFile: Option[String] = None
  private var _workdir:          Option[String] = None // Initialized when makeSimulator is called
  protected def wantedWorkspacePath: String =
    if (_workdirFinalFile.isDefined)
      Seq(_testRunDir, _moduleDutName, _simulatorName, _simName, _workdirFinalFile.get).mkString("/")
    else
      Seq(_testRunDir, _moduleDutName, _simulatorName, _simName).mkString("/")

  // Traces
  private var _traceExtension:  Option[String] = None
  private var _traceWantedName: String = "trace"
  private var _finalTracePath:  Option[String] = None

  def finalTracePath: Option[String] = _finalTracePath
  private def _emittedTraceRelPath: Option[String] =
    if (_workdir.isDefined && _traceExtension.isDefined)
      Some(Seq(_workdir.get, s"trace.${_traceExtension.get}").mkString("/"))
    else None

  private var _firtoolArgs: Seq[String] = Seq()

  /** Launch and execute a simulation given a list of [[SimulatorSettings]]. */
  def simulate[T <: RawModule](
    module:   => T,
    settings: Seq[SimulatorSettings] = Seq(),
    simName:  String
  )(body:     T => Unit
  ): Unit = {

    // Set the initial settings before the simulation: i.e. backend compile settings
    setInitialSettings(settings)

    // Create a new simulator
    val simulator = makeSimulator
    simulator
      .simulate(module) { simulatedModule =>
        // Set the controller settings: i.e. enable trace
        setControllerSettings(simulatedModule.controller, settings)

        // Update the wanted workspace path
        _moduleDutName = simulatedModule.wrapped.name
        _simName = simName

        // Launch the actual simulation and return the result
        body(simulatedModule.wrapped)
      }
      .result

    // Cleanup the simulation after the execution
    simulator.cleanup()
  }

  /**
    * Set settings before the simulation starts. It sets the initial settings
    * such as the backend compile settings, trace style, etc.
    */
  private def setInitialSettings(settings: Seq[SimulatorSettings]): Unit = {
    settings.foreach {
      case t: TraceSetting =>
        _backendCompileSettings = verilator.Backend.CompilationSettings(
          traceStyle = Some(t.traceStyle),
          outputSplit = None,
          outputSplitCFuncs = None,
          disabledWarnings = Seq(),
          disableFatalExitOnWarnings = false
        )
        // Set also the extension
        _traceExtension = Some(t.extension)

      case TraceName(name) =>
        if (
          !settings.exists {
            case _: TraceSetting => true
            case _ => false
          }
        ) throw new Exception("TraceName must be used with TraceSetting")
        if (settings.count(_.isInstanceOf[TraceName]) > 1)
          throw new Exception("TraceName must be used only once")
        _traceWantedName = name

      case SaveWorkspace(name) =>
        if (settings.count(_.isInstanceOf[SaveWorkspace]) > 1)
          throw new Exception("SaveWorkspace must be used only once")
        if (name == "") // If no name is specified -> Current date time as default (gg-mm-ddThh:mm) no millis
          _workdirFinalFile =
            Some(LocalDateTime.now().format(format.DateTimeFormatter.ofPattern("dd_MM_yyyy:HH-mm-ss")))
        else
          _workdirFinalFile = Some(name)

      case FirtoolArgs(args) => _firtoolArgs = args
      case _                 =>
    }

    // If it contains underscore, add it to the trace name
    if (settings.contains(simulatorSettings.VcdTraceWithUnderscore))
      _traceWantedName = s"${_traceWantedName}_underscore"

  }

  /**
    * Set the controller settings. It sets settings such as the trace output
    * enable.
    */
  private def setControllerSettings(controller: Simulation.Controller, settings: Seq[SimulatorSettings]): Unit =
    settings.foreach {
      case _: TraceSetting => controller.setTraceEnabled(true)
      case _ => // Ignore other settings
    }

  /** Default ParametricSimulator */
  protected class DefaultParametricSimulator(val workspacePath: String, val tag: String)
      extends SingleBackendSimulator[verilator.Backend] {

    val backend:              verilator.Backend = verilator.Backend.initializeFromProcessEnvironment()
    override val firtoolArgs: Seq[String] = _firtoolArgs

    val backendSpecificCompilationSettings: verilator.Backend.CompilationSettings = _backendCompileSettings
    val commonCompilationSettings: CommonCompilationSettings =
      CommonCompilationSettings(
        optimizationStyle = CommonCompilationSettings.OptimizationStyle.OptimizeForCompilationSpeed,
        availableParallelism = CommonCompilationSettings.AvailableParallelism.UpTo(4),
        defaultTimescale = Some(CommonCompilationSettings.Timescale.FromString("1ms/1ms"))
      )

    /**
      * Cleanup the simulation and move the simulation workspace to the wanted
      * workspace path.
      */
    def cleanup(): Unit = {

      val workDirOld = os.Path(workspacePath)
      val workDir = os.pwd / os.RelPath(wantedWorkspacePath)

      // Check if the workspace must be saved or not
      if (_workdirFinalFile.isDefined)
        os.copy(workDirOld, workDir, replaceExisting = true, createFolders = true, mergeFolders = true)

      // Rename the wanted trace
      if (_traceExtension.isDefined) {
        val tracePath = workDirOld / os.RelPath(_emittedTraceRelPath.get)
        val wantedTracePath =
          os.pwd / os.RelPath(Seq(wantedWorkspacePath, s"${_traceWantedName}.${_traceExtension.get}").mkString("/"))
        os.copy(tracePath, wantedTracePath, replaceExisting = true, createFolders = true)
        _finalTracePath = Some(wantedTracePath.toString)
      }

    }
  }

  /** Create a new simulator */
  protected def makeSimulator: DefaultParametricSimulator = { // def allows to create a new simulator each tim
    val className = _simulatorName
    val id = java.lang.management.ManagementFactory.getRuntimeMXBean.getName

    val tmpWorkspacePath = Files.createTempDirectory(s"${className}_${id}_").toString
    val tag = "default"
    val defaultSimulator = new DefaultParametricSimulator(tmpWorkspacePath, tag = tag)
    _workdir = Some(s"${defaultSimulator.workingDirectoryPrefix}-$tag")
    defaultSimulator
  }
}
