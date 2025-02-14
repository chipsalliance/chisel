package chisel3.simulator

import chisel3.{Data, RawModule}
import firrtl.options.StageUtils.dramaticMessage
import java.nio.file.Paths
import scala.util.{Failure, Success, Try}
import scala.util.control.NoStackTrace
import svsim._

private[this] object Exceptions {

  case object CompilationFailed extends Exception

  class AssertionFailed(message: String)
      extends RuntimeException(
        dramaticMessage(
          header = Some("One or more assertions failed during Chiselsim simulation"),
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

  final def simulate[T <: RawModule, U](
    module:       => T,
    layerControl: LayerControl.Type = LayerControl.EnableAll
  )(body: (SimulatedModule[T]) => U): Simulator.BackendInvocationDigest[U] = {
    val workspace = new Workspace(path = workspacePath, workingDirectoryPrefix = workingDirectoryPrefix)
    workspace.reset()
    val elaboratedModule = workspace.elaborateGeneratedModule({ () => module }, firtoolArgs)
    workspace.generateAdditionalSources()

    val commonCompilationSettingsUpdated = commonCompilationSettings.copy(
      // Append to the include directorires based on what the
      // workspace indicates is the path for primary sources.  This
      // ensures that `` `include `` directives can be resolved.
      includeDirs = Some(commonCompilationSettings.includeDirs.getOrElse(Seq.empty) :+ workspace.primarySourcesPath),
      verilogPreprocessorDefines =
        commonCompilationSettings.verilogPreprocessorDefines ++ layerControl.preprocessorDefines(
          elaboratedModule
        ),
      fileFilter = commonCompilationSettings.fileFilter.orElse(layerControl.shouldIncludeFile(elaboratedModule))
    )

    // Compile the design.  Early exit if the compilation fails for any reason.
    val compilationStartTime = System.nanoTime()
    val simulation =
      try {
        workspace
          .compile(backend)(
            tag,
            commonCompilationSettingsUpdated,
            backendSpecificCompilationSettings,
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

    // Simulate the compiled design.  Because svsim returns extremely vague
    // exceptions on failure (UnexpectedEndOfMessage), when we do see a failure,
    // try to figure out what happened and return a better, more specific error.
    // If we can't determine a more specific error, then keep the original one.
    val simulationOutcome = Try {
      simulation.runElaboratedModule(elaboratedModule = elaboratedModule) { (module: SimulatedModule[T]) =>
        val outcome = body(module)
        module.completeSimulation()
        outcome

      }
    }.recoverWith { error =>
      val asserts =
        backend.assertionFailed(Paths.get(workspacePath, s"workdir-${tag}", "simulation-log.txt"))
      asserts match {
        case assertionLines if asserts.nonEmpty =>
          Failure(
            new Exceptions.AssertionFailed(message =
              s"""|The following assertion failures were extracted from the log file:
                  |${asserts.mkString("  - ", "\n  - ", "")} """.stripMargin
            )
          )
        case _ => Failure(error)
      }
    }
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
