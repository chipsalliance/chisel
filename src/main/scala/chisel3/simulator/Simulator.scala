package chisel3.simulator

import chisel3.{Data, RawModule, SimulationTestHarnessInterface}
import chisel3.experimental.inlinetest.{HasTests, SimulatedTest, TestParameters, TestResult}
import firrtl.options.StageUtils.dramaticMessage
import java.nio.file.attribute.BasicFileAttributes
import java.nio.file.{FileSystems, FileVisitResult, FileVisitor, Files, Path, PathMatcher, Paths}
import scala.collection.mutable
import scala.util.{Failure, Success, Try}
import scala.util.control.NoStackTrace
import svsim._

object SimulationOutcome {

  /** A test result, either success or failure. */
  sealed trait Type

  /** Test passed. */
  case object Success extends Type

  /** Test failed somehow. */
  sealed trait Failure extends Type

  /** Test timed out. */
  case class Timeout(n: BigInt) extends Failure

  /** Test failed with an assertion. */
  case class Assertion(simulatorOutput: String) extends Failure

  /** Test signaled failure. */
  case object SignaledFailure extends Failure
}

object Exceptions {

  class AssertionFailed private[simulator] (message: String, val simulatorOutput: String)
      extends RuntimeException(
        dramaticMessage(
          header = Some("One or more assertions failed during Chiselsim simulation"),
          body = message
        )
      )
      with NoStackTrace

  class Timeout private[simulator] (private[simulator] val timesteps: BigInt, message: String)
      extends RuntimeException(
        dramaticMessage(
          header = Some(s"A timeout occurred after $timesteps timesteps"),
          body = message
        )
      )
      with NoStackTrace

  class TestFailed private[simulator]
      extends RuntimeException(
        dramaticMessage(
          header = Some(s"The test finished and signaled failure"),
          body = ""
        )
      )
      with NoStackTrace

  class TestsFailed private[simulator] (
    message:     String,
    val results: Seq[chisel3.experimental.inlinetest.SimulatedTest]
  ) extends RuntimeException(
        dramaticMessage(
          header = Some("One or more tests failed during simulation"),
          body = message
        )
      )
      with NoStackTrace
}

final object Simulator {

  final case class BackendInvocationDigest[T](
    compilationStartTime: Long,
    compilationEndTime:   Long,
    outcome:              BackendInvocationOutcome[T]
  ) {
    def result = outcome match {
      case SimulationDigest(_, _, outcome) => outcome.get
      case CompilationFailed(error)        => throw error
    }
  }
  sealed trait BackendInvocationOutcome[T]
  final case class CompilationFailed[T](error: Throwable) extends BackendInvocationOutcome[T]

  final case class SimulationDigest[T](simulationStartTime: Long, simulationEndTime: Long, outcome: Try[T])
      extends BackendInvocationOutcome[T]
}

trait Simulator[T <: Backend] {

  val backend: T
  def tag:           String
  def workspacePath: String
  def workingDirectoryPrefix = "workdir"
  def customSimulationWorkingDirectory:   Option[String] = None
  def verbose:                            Boolean = false
  def commonCompilationSettings:          CommonCompilationSettings
  def backendSpecificCompilationSettings: backend.CompilationSettings

  /** Post process a simulation log to see if there are any backend-specific failures.
    *
    * @return None if no failures found or Some if they are
    */
  private def postProcessLog(workspace: Workspace): Option[Throwable] = {
    val log =
      Paths.get(workspace.absolutePath, s"${workspace.workingDirectoryPrefix}-${tag}", "simulation-log.txt").toFile
    val lines = scala.io.Source
      .fromFile(log)
      .getLines()
      .zipWithIndex
      .filter { case (line, _) => backend.assertionFailed.matches(line) }
      .toSeq

    Option.when(lines.nonEmpty)(
      new Exceptions.AssertionFailed(
        simulatorOutput = lines.map(_._1).mkString("\n"),
        message = s"""|The following assertion failures were extracted from the log file:
                      |
                      |  lineNo  line
                      |  ${"-" * 76}
                      |${lines.map { case (line, lineNo) => f"$lineNo%8d  $line" }.mkString("\n")}
                      |
                      |For more information, see the complete log file:
                      |
                      |  ${log}""".stripMargin
      )
    )
  }

  /** Simulate a Chisel module with some stimulus
    *
    * @param module a Chisel module to simulate
    * @param chiselOpts command line options to pass to Chisel
    * @param firtoolOpts command line options to pass to firtool
    * @param settings ChiselSim-related settings used for simulation
    * @param body stimulus to apply to the module
    * @param commonSettingsModifications modifications to common compilation
    * settings
    * @param backendSettingsModifications modifications to backend (e.g.,
    * Verilator or VCS) compilation settings
    *
    * @note Take care when passing `chiselOpts`.  The following options are set
    * by default and if you set incompatible options, the simulation will fail.
    */
  final def simulate[T <: RawModule, U](
    module:      => T,
    chiselOpts:  Array[String] = Array.empty,
    firtoolOpts: Array[String] = Array.empty,
    settings:    Settings[T] = Settings.defaultRaw[T]
  )(body: (SimulatedModule[T]) => U)(
    implicit chiselOptsModifications: ChiselOptionsModifications,
    firtoolOptsModifications:         FirtoolOptionsModifications,
    commonSettingsModifications:      svsim.CommonSettingsModifications,
    backendSettingsModifications:     svsim.BackendSettingsModifications
  ): Simulator.BackendInvocationDigest[U] = {
    val workspace = new Workspace(path = workspacePath, workingDirectoryPrefix = workingDirectoryPrefix)
    workspace.reset()
    val elaboratedModule =
      workspace
        .elaborateGeneratedModule(
          () => module,
          args = chiselOptsModifications(chiselOpts).toSeq,
          firtoolArgs = firtoolOptsModifications(firtoolOpts).toSeq
        )
    _simulate(workspace, elaboratedModule, settings)(body)
  }

  final def simulateTests[T <: RawModule with HasTests, U](
    module:           => T,
    includeTestGlobs: Array[String],
    chiselOpts:       Array[String] = Array.empty,
    firtoolOpts:      Array[String] = Array.empty,
    settings: Settings[RawModule with SimulationTestHarnessInterface] =
      Settings.defaultRaw[RawModule with SimulationTestHarnessInterface]
  )(body: (SimulatedModule[RawModule with SimulationTestHarnessInterface]) => U)(
    implicit chiselOptsModifications: ChiselOptionsModifications,
    firtoolOptsModifications:         FirtoolOptionsModifications,
    commonSettingsModifications:      svsim.CommonSettingsModifications,
    backendSettingsModifications:     svsim.BackendSettingsModifications
  ) = {
    val workspace = new Workspace(path = workspacePath, workingDirectoryPrefix = workingDirectoryPrefix)
    workspace.reset()
    val filesystem = FileSystems.getDefault()
    val results = workspace
      .elaborateAndMakeTestHarnessWorkspaces(
        () => module,
        includeTestGlobs = includeTestGlobs.toSeq,
        args = chiselOptsModifications(chiselOpts).toSeq,
        firtoolArgs = firtoolOptsModifications(firtoolOpts).toSeq
      )
      .map { case (testWorkspace, elaboratedTest, elaboratedModule) =>
        val digest = _simulate(testWorkspace, elaboratedModule, settings)(body)
        // Try to unpack the result, otherwise figure out what went wrong.
        // TODO: push this down, i.e. all ChiselSim invocations return a SimulationOutcome
        val outcome: SimulationOutcome.Type =
          try {
            digest.result
            SimulationOutcome.Success
          } catch {
            // Simulation ended due to an aserrtion
            case assertion: Exceptions.AssertionFailed => SimulationOutcome.Assertion(assertion.simulatorOutput)
            // Simulation ended because the testharness signaled success=0
            case _: Exceptions.TestFailed => SimulationOutcome.SignaledFailure
            // Simulation timed out
            case to: Exceptions.Timeout => SimulationOutcome.Timeout(to.timesteps)
            // Simulation did not run correctly
            case e: Throwable => throw e
          }
        SimulatedTest(elaboratedTest, outcome)
      }

    val failures = results.filter(!_.success)
    if (failures.nonEmpty) {
      val moduleName = results.head.dutName
      val failedTests = failures.size
      val passedTests = results.size - failedTests
      val failureMessages = failures.map { test =>
        s"  - ${test.testName}: ${test.result.asInstanceOf[TestResult.Failure].message}"
      }
      val aggregatedMessage =
        s"${moduleName} tests: ${passedTests} passed, ${failedTests} failed\nfailures: \n${failureMessages.mkString("\n")}"
      throw new Exceptions.TestsFailed(aggregatedMessage, results)
    }
  }

  private def _simulate[T <: RawModule, U](
    workspace:        Workspace,
    elaboratedModule: ElaboratedModule[T],
    settings:         Settings[T] = Settings.defaultRaw[T]
  )(body: (SimulatedModule[T]) => U)(
    implicit chiselOptsModifications: ChiselOptionsModifications,
    firtoolOptsModifications:         FirtoolOptionsModifications,
    commonSettingsModifications:      svsim.CommonSettingsModifications,
    backendSettingsModifications:     svsim.BackendSettingsModifications
  ): Simulator.BackendInvocationDigest[U] = {
    // Find all the directories that exist under another directory.
    val primarySourcesDirectories = mutable.LinkedHashSet.empty[String]
    class DirectoryFinder extends FileVisitor[Path] {

      override def visitFile(file: Path, attrs: BasicFileAttributes): FileVisitResult = {
        FileVisitResult.CONTINUE
      }

      override def preVisitDirectory(dir: Path, attrs: BasicFileAttributes): FileVisitResult = {
        FileVisitResult.CONTINUE
      }

      override def postVisitDirectory(dir: Path, ioe: java.io.IOException): FileVisitResult = {
        primarySourcesDirectories += dir.toString
        FileVisitResult.CONTINUE
      }

      override def visitFileFailed(file: Path, ioe: java.io.IOException): FileVisitResult = {
        throw ioe
      }

    }
    Files.walkFileTree(Paths.get(workspace.primarySourcesPath), new DirectoryFinder)

    val commonCompilationSettingsUpdated = commonSettingsModifications(
      commonCompilationSettings.copy(
        // Append to the include directorires based on what the
        // workspace indicates is the path for primary sources.  This
        // ensures that `` `include `` directives can be resolved.
        includeDirs = Some(commonCompilationSettings.includeDirs.getOrElse(Seq.empty) ++ primarySourcesDirectories),
        verilogPreprocessorDefines =
          commonCompilationSettings.verilogPreprocessorDefines ++ settings.preprocessorDefines(elaboratedModule),
        fileFilter =
          commonCompilationSettings.fileFilter.orElse(settings.verilogLayers.shouldIncludeFile(elaboratedModule)),
        directoryFilter = commonCompilationSettings.directoryFilter.orElse(
          settings.verilogLayers.shouldIncludeDirectory(elaboratedModule, workspace.primarySourcesPath)
        ),
        simulationSettings = commonCompilationSettings.simulationSettings.copy(
          plusArgs = commonCompilationSettings.simulationSettings.plusArgs ++ settings.plusArgs,
          enableWavesAtTimeZero =
            commonCompilationSettings.simulationSettings.enableWavesAtTimeZero || settings.enableWavesAtTimeZero
        )
      )
    )

    workspace.generateAdditionalSources(timescale = commonCompilationSettingsUpdated.defaultTimescale)

    // Compile the design.  Early exit if the compilation fails for any reason.
    val compilationStartTime = System.nanoTime()
    val simulation =
      try {
        workspace
          .compile(backend)(
            tag,
            commonCompilationSettingsUpdated,
            backendSettingsModifications(backendSpecificCompilationSettings).asInstanceOf[backend.CompilationSettings],
            customSimulationWorkingDirectory,
            verbose
          )
      } catch {
        case error: Throwable =>
          return Simulator.BackendInvocationDigest(
            compilationStartTime = compilationStartTime,
            compilationEndTime = System.nanoTime(),
            outcome = Simulator.CompilationFailed(error)
          )
      }
    val compilationEndTime = System.nanoTime()

    // Simulate the compiled design.  After the simulation completes,
    // post-process the log to figure out what happened.  This post-processing
    // occurs in _both_ the success and failure modes for multiple reasons:
    //
    // 1. svsim returns a vague `UnexpectedEndOfMessage` on failure.
    // 2. svsim assumes that simulators will either exit on an assertion failure
    //    or can be compile-time configured to do so.  This is _not_ the case
    //    for VCS as VCS requires runtime configuration to do and svsim
    //    substitutes its own executable which ignores command line arguments.
    //
    // Note: this would be much better to handle with extensions to the FIRRTL
    // ABI which would abstract away these differences.
    val simulationOutcome = Try {
      simulation.runElaboratedModule(
        elaboratedModule = elaboratedModule,
        traceEnabled = commonCompilationSettingsUpdated.simulationSettings.enableWavesAtTimeZero
      ) { (module: SimulatedModule[T]) =>
        val outcome = body(module)
        module.completeSimulation()
        outcome

      }
    }.transform(
      s /*success*/ = { case success =>
        postProcessLog(workspace) match {
          case None        => Success(success)
          case Some(error) => Failure(error)
        }
      },
      f /*failure*/ = { case originalError =>
        postProcessLog(workspace) match {
          case None           => Failure(originalError)
          case Some(newError) => Failure(newError)
        }
      }
    )

    val simulationEndTime = System.nanoTime()

    // Return the simulation result.
    Simulator.BackendInvocationDigest(
      compilationStartTime = compilationStartTime,
      compilationEndTime = compilationEndTime,
      outcome = Simulator.SimulationDigest(
        simulationStartTime = compilationEndTime,
        simulationEndTime = simulationEndTime,
        outcome = simulationOutcome
      )
    )

  }

}
