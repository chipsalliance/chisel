package chisel3.simulator

import chisel3.{Data, RawModule}
import firrtl.options.StageUtils.dramaticMessage
import java.nio.file.Paths
import scala.util.{Failure, Success, Try}
import scala.util.control.NoStackTrace
import svsim._

object Exceptions {

  class AssertionFailed private[simulator] (message: String)
      extends RuntimeException(
        dramaticMessage(
          header = Some("One or more assertions failed during Chiselsim simulation"),
          body = message
        )
      )
      with NoStackTrace

  class Timeout private[simulator] (timesteps: BigInt, message: String)
      extends RuntimeException(
        dramaticMessage(
          header = Some(s"A timeout occurred after $timesteps timesteps"),
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
  def firtoolArgs:                        Seq[String] = Seq()
  def commonCompilationSettings:          CommonCompilationSettings
  def backendSpecificCompilationSettings: backend.CompilationSettings

  /** Post process a simulation log to see if there are any backend-specific failures.
    *
    * @return None if no failures found or Some if they are
    */
  private def postProcessLog: Option[Throwable] = {
    val log = Paths.get(workspacePath, s"workdir-${tag}", "simulation-log.txt").toFile
    val lines = scala.io.Source
      .fromFile(log)
      .getLines()
      .zipWithIndex
      .filter { case (line, _) => backend.assertionFailed.matches(line) }
      .toSeq

    Option.when(lines.nonEmpty)(
      new Exceptions.AssertionFailed(
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

  final def simulate[T <: RawModule, U](
    module:         => T,
    chiselSettings: ChiselSettings[T] = ChiselSettings.defaultRaw[T]
  )(body: (SimulatedModule[T]) => U)(
    implicit commonSettingsModifications: svsim.CommonSettingsModifications,
    backendSettingsModifications:         svsim.BackendSettingsModifications
  ): Simulator.BackendInvocationDigest[U] = {
    val workspace = new Workspace(path = workspacePath, workingDirectoryPrefix = workingDirectoryPrefix)
    workspace.reset()
    val elaboratedModule = workspace.elaborateGeneratedModule({ () => module }, firtoolArgs)
    workspace.generateAdditionalSources()

    val commonCompilationSettingsUpdated = commonSettingsModifications(commonCompilationSettings).copy(
      // Append to the include directorires based on what the
      // workspace indicates is the path for primary sources.  This
      // ensures that `` `include `` directives can be resolved.
      includeDirs = Some(commonCompilationSettings.includeDirs.getOrElse(Seq.empty) :+ workspace.primarySourcesPath),
      verilogPreprocessorDefines =
        commonCompilationSettings.verilogPreprocessorDefines ++ chiselSettings.preprocessorDefines(elaboratedModule),
      fileFilter =
        commonCompilationSettings.fileFilter.orElse(chiselSettings.verilogLayers.shouldIncludeFile(elaboratedModule)),
      directoryFilter = commonCompilationSettings.directoryFilter.orElse(
        chiselSettings.verilogLayers.shouldIncludeDirectory(elaboratedModule, workspace.primarySourcesPath)
      )
    )

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
      simulation.runElaboratedModule(elaboratedModule = elaboratedModule) { (module: SimulatedModule[T]) =>
        val outcome = body(module)
        module.completeSimulation()
        outcome

      }
    }.transform(
      s /*success*/ = { case success =>
        postProcessLog match {
          case None        => Success(success)
          case Some(error) => Failure(error)
        }
      },
      f /*failure*/ = { case originalError =>
        postProcessLog match {
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
