package chisel3

import svsim._
import chisel3.reflect.DataMirror
import scala.collection.mutable
import java.nio.file.{Files, Path, Paths}

package object simulator {

  /**
    * An opaque class that can be passed to `Simulation.run` to get access to a `SimulatedModule` in the simulation body.
    */
  final class ElaboratedModule[T] private[simulator] (
    private[simulator] val wrapped: T,
    private[simulator] val ports:   Seq[(Data, ModuleInfo.Port)])

  /**
    * A class that enables using a Chisel module to control an `svsim.Simulation`.
    */
  final class SimulatedModule[T] private[simulator] (
    private[simulator] val elaboratedModule: ElaboratedModule[T],
    controller:                              Simulation.Controller)
      extends AnySimulatedModule(elaboratedModule.ports, controller) {
    def wrapped: T = elaboratedModule.wrapped
  }
  sealed class AnySimulatedModule protected (
    ports:          Seq[(Data, ModuleInfo.Port)],
    val controller: Simulation.Controller) {

    // -- Port Mapping

    private val simulationPorts = ports.map {
      case (data, port) => data -> controller.port(port.name)
    }.toMap
    def port(data: Data): Simulation.Port = {
      simulationPorts(data)
    }

    // -- Peek/Poke API Support

    // When using the low-level API, the user must explicitly call `controller.completeInFlightCommands()` to ensure that all commands are executed. When using a higher-level API like peek/poke, we handle this automatically.
    private var shouldCompleteInFlightCommands: Boolean = false
    private[simulator] def completeSimulation() = {
      if (shouldCompleteInFlightCommands) {
        shouldCompleteInFlightCommands = false
        controller.completeInFlightCommands()
      }
    }

    // The peek/poke API implicitly evaluates on the first peek after one or more pokes. This is _only_ for peek/poke and using `controller` directly will not provide this behavior.
    private var evaluateBeforeNextPeek: Boolean = false
    private[simulator] def willEvaluate() = {
      evaluateBeforeNextPeek = false
    }
    private[simulator] def willPoke() = {
      shouldCompleteInFlightCommands = true
      evaluateBeforeNextPeek = true
    }
    private[simulator] def willPeek() = {
      shouldCompleteInFlightCommands = true
      if (evaluateBeforeNextPeek) {
        willEvaluate()
        controller.run(0)
      }
    }
  }
  private[simulator] object AnySimulatedModule {
    private val dynamicVariable = new scala.util.DynamicVariable[Option[AnySimulatedModule]](None)
    def withValue[T](module: AnySimulatedModule)(body: => T): T = {
      require(dynamicVariable.value.isEmpty, "Nested simulations are not supported.")
      dynamicVariable.withValue(Some(module))(body)
    }
    def current: AnySimulatedModule = dynamicVariable.value.get
  }

  implicit class ChiselSimulation(simulation: Simulation) {
    def runElaboratedModule[T, U](
      elaboratedModule:              ElaboratedModule[T],
      conservativeCommandResolution: Boolean = false,
      verbose:                       Boolean = false,
      executionScriptLimit:          Option[Int] = None
    )(body:                          SimulatedModule[T] => U
    ): U = {
      simulation.run(conservativeCommandResolution, verbose, executionScriptLimit) { controller =>
        val module = new SimulatedModule(elaboratedModule, controller)
        AnySimulatedModule.withValue(module) {
          body(module)
        }
      }
    }
  }

  implicit class ChiselWorkspace(workspace: Workspace) {
    def elaborateGeneratedModule[T <: RawModule](
      generateModule: () => T
    ): ElaboratedModule[T] = {
      // Use CIRCT to generate SystemVerilog sources, and potentially additional artifacts
      var someDut: Option[T] = None
      val outputAnnotations = (new circt.stage.ChiselStage).execute(
        Array("--target", "systemverilog", "--split-verilog"),
        Seq(
          chisel3.stage.ChiselGeneratorAnnotation { () =>
            val dut = generateModule()
            someDut = Some(dut)
            dut
          },
          circt.stage.FirtoolOption("-disable-annotation-unknown"),
          firrtl.options.TargetDirAnnotation(workspace.supportArtifactsPath)
        )
      )

      // Move the files indicated by a filelist.  No-op if the file has already
      // been moved.
      val movedFiles = mutable.HashSet.empty[Path]
      val supportArtifactsPath = Paths.get(workspace.supportArtifactsPath)
      def moveFiles(filelist: Path) =
        // Extract all lines (files) from the filelist.
        Files
          .lines(filelist)
          .map(Paths.get(_))
          // Convert the files to an absolute version and a relative version.
          .map {
            case file if file.startsWith(supportArtifactsPath) =>
              (file, file.subpath(supportArtifactsPath.getNameCount(), -1))
            case file => (supportArtifactsPath.resolve(file), file)
          }
          // Normalize the absolute path so it can be checked if it has already
          // been moved.
          .map { case (abs, rel) => (abs.normalize(), rel) }
          // Move the file into primarySourcesPath if it has not already been moved.
          .forEach {
            case (abs, _) if movedFiles.contains(abs) =>
            case (abs, rel) =>
              Files.move(
                abs,
                Paths.get(workspace.primarySourcesPath).resolve(rel)
              )
              movedFiles += abs
          }

      // Move a file in a filelist which may not exist.  Do nothing if the
      // filelist does not exist.
      def maybeMoveFiles(filelist: Path): Unit = filelist match {
        case _ if Files.exists(filelist) => moveFiles(filelist)
        case _                           =>
      }

      // Move files indicated by 'filelist.f' (which must exist).  Move files
      // indicated by a black box filelist (which may exist).
      moveFiles(supportArtifactsPath.resolve("filelist.f"))
      maybeMoveFiles(supportArtifactsPath.resolve("firrtl_black_box_resource_files.f"))

      // Initialize Module Info
      val dut = someDut.get
      val ports = {

        /**
          * We infer the names of various ports since we don't currently have a good alternative when using MFC. We hope to replace this once we get better support from CIRCT.
          */
        def leafPorts(node: Data, name: String): Seq[(Data, ModuleInfo.Port)] = {
          node match {
            case record: Record => {
              record.elements.toSeq.flatMap {
                case (fieldName, field) =>
                  leafPorts(field, s"${name}_${fieldName}")
              }
            }
            case vec: Vec[_] => {
              vec.zipWithIndex.flatMap {
                case (element, index) =>
                  leafPorts(element, s"${name}_${index}")
              }
            }
            case element: Element =>
              // Return the port only if the width is positive (firtool will optimized it out from the *.sv primary source)
              if (element.widthKnown && element.getWidth > 0) {
                DataMirror.directionOf(element) match {
                  case ActualDirection.Input =>
                    Seq((element, ModuleInfo.Port(name, isGettable = true, isSettable = true)))
                  case ActualDirection.Output => Seq((element, ModuleInfo.Port(name, isGettable = true)))
                  case _                      => Seq()
                }
              } else {
                Seq()
              }
          }
        }
        // Chisel ports can be Data or Property, but there is no ABI for Property ports, so we only return Data.
        DataMirror.modulePorts(dut).flatMap {
          case (name, data: Data) => leafPorts(data, name)
          case _ => Nil
        }
      }
      workspace.elaborate(
        ModuleInfo(
          name = dut.name,
          ports = ports.map(_._2)
        )
      )
      new ElaboratedModule(dut, ports)
    }
  }
}
