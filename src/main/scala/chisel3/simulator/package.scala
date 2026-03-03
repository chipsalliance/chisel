package chisel3

import svsim._
import chisel3.reflect.DataMirror
import chisel3.experimental.dataview.reifyIdentityView
import chisel3.experimental.inlinetest.{ElaboratedTest, HasTests, TestParameters}
import chisel3.stage.DesignAnnotation
import scala.collection.mutable
import java.nio.file.{Files, Path, Paths}
import firrtl.{annoSeqToSeq, seqToAnnoSeq}
import firrtl.AnnotationSeq

package object simulator {

  /** A trait that provides the minimal set of ChiselSim APIs.
    *
    * Example usage:
    * {{{
    * import chisel3.simulator.ChiselSim
    *
    * class Foo extends ChiselSim {
    *   /** This has access to all ChiselSim APIs like `simulate`, `peek`, and `poke`. */
    * }
    * }}}
    *
    * @see [[chisel3.simulator.scalatest.ChiselSim]]
    */
  trait ChiselSim extends ControlAPI with PeekPokeAPI with SimulatorAPI

  /**
    * An opaque class that can be passed to `Simulation.run` to get access to a `SimulatedModule` in the simulation body.
    */
  final class ElaboratedModule[T] private[simulator] (
    private[simulator] val wrapped: T,
    private[simulator] val ports:   Seq[(Data, ModuleInfo.Port)],
    private[simulator] val layers:  Seq[chisel3.layer.Layer]
  ) {

    private[chisel3] val portMap = ports.toMap

  }

  /**
    * A class that enables using a Chisel module to control an `svsim.Simulation`.
    */
  final class SimulatedModule[T] private[simulator] (
    private[simulator] val elaboratedModule: ElaboratedModule[T],
    controller:                              Simulation.Controller
  ) extends AnySimulatedModule(elaboratedModule.ports, controller) {
    def wrapped: T = elaboratedModule.wrapped
  }
  sealed class AnySimulatedModule protected (
    ports:          Seq[(Data, ModuleInfo.Port)],
    val controller: Simulation.Controller
  ) {

    // -- Port Mapping

    private val simulationPorts = ports.map { case (data, port) =>
      data -> controller.port(port.name)
    }.toMap
    def port(data: Data): Simulation.Port = {
      // TODO, we can support non identity views, but it will require changing this API to return a Seq[Port]
      // and packing/unpacking the BigInt literal representation.
      // TODO implement support for read-only.
      val (reified, _) = reifyIdentityView(data).getOrElse {
        val url = "https://github.com/chipsalliance/chisel/issues/new/choose"
        throw new Exception(
          s"Cannot poke $data as is a view that does not map to a single Data. " +
            s"Please file an issue at $url requesting support for this use case."
        )
      }
      simulationPorts(reified)
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
      traceEnabled:                  Boolean = false,
      executionScriptLimit:          Option[Int] = None
    )(body: SimulatedModule[T] => U): U = {
      simulation.run(conservativeCommandResolution, verbose, traceEnabled, executionScriptLimit) { controller =>
        val module = new SimulatedModule(elaboratedModule, controller)
        AnySimulatedModule.withValue(module) {
          body(module)
        }
      }
    }
  }

  implicit class ChiselWorkspace(workspace: Workspace) {
    def elaborateGeneratedModule[T <: RawModule](
      generateModule: () => T,
      args:           Seq[String] = Seq.empty,
      firtoolArgs:    Seq[String] = Seq.empty
    ): ElaboratedModule[T] = {
      val generated = generateWorkspaceSources(generateModule, args, firtoolArgs)
      val ports = getModuleInfoPorts(generated.dut)
      val moduleInfo = initializeModuleInfo(generated.dut, ports.map(_._2))
      val layers = generated.outputAnnotations.collectFirst { case DesignAnnotation(_, layers) => layers }.get
      new ElaboratedModule(generated.dut, ports, layers)
    }

    def elaborateAndMakeTestHarnessWorkspaces[T <: RawModule with HasTests](
      generateModule:   () => T,
      includeTestGlobs: Seq[String],
      args:             Seq[String] = Seq.empty,
      firtoolArgs:      Seq[String] = Seq.empty
    ): Seq[(Workspace, ElaboratedTest[T], ElaboratedModule[RawModule with SimulationTestHarnessInterface])] = {
      val updatedArgs = args ++ includeTestGlobs.map("--include-tests-name=" + _)
      val generated = generateWorkspaceSources(generateModule, updatedArgs, firtoolArgs)
      generated.testHarnesses.map { case elaboratedTest =>
        val testWorkspace = workspace.shallowCopy(workspace.absolutePath + "/tests/" + elaboratedTest.testName)
        val ports = getModuleInfoPorts(elaboratedTest.testHarness)
        val moduleInfo = testWorkspace.initializeModuleInfo(elaboratedTest.testHarness, ports.map(_._2))
        val layers = generated.outputAnnotations.collectFirst { case DesignAnnotation(_, layers) => layers }.get
        (
          testWorkspace,
          elaboratedTest,
          new ElaboratedModule(
            elaboratedTest.testHarness.asInstanceOf[RawModule with SimulationTestHarnessInterface],
            ports,
            layers
          )
        )
      }
    }

    case class GeneratedWorkspaceInfo[T <: RawModule](
      dut:               T,
      testHarnesses:     Seq[ElaboratedTest[T]],
      outputAnnotations: AnnotationSeq
    )

    /** Use CIRCT to generate SystemVerilog sources, and potentially additional artifacts */
    def generateWorkspaceSources[T <: RawModule](
      generateModule: () => T,
      args:           Seq[String],
      firtoolArgs:    Seq[String]
    ): GeneratedWorkspaceInfo[T] = {
      var someDut:           Option[() => T] = None
      var someTestHarnesses: Option[() => Seq[ElaboratedTest[T]]] = None
      val chiselArgs = Array("--target", "systemverilog", "--split-verilog") ++ args
      val outputAnnotations = (new circt.stage.ChiselStage).execute(
        chiselArgs,
        Seq(
          chisel3.stage.ChiselGeneratorAnnotation { () =>
            val dut = generateModule()
            someDut = Some(() => dut)
            someTestHarnesses = Some(() =>
              dut match {
                case dut: HasTests => dut.getElaboratedTests.map(_.asInstanceOf[ElaboratedTest[T]])
                case _ => Nil
              }
            )
            dut
          },
          circt.stage.FirtoolOption("-disable-annotation-unknown"),
          circt.stage.FirtoolOption("-advanced-layer-sink"),
          firrtl.options.TargetDirAnnotation(workspace.supportArtifactsPath)
        ) ++ firtoolArgs.map(circt.stage.FirtoolOption(_))
      )

      // Move the files indicated by a filelist.  No-op if the file has already
      // been moved.
      val movedFiles = mutable.HashSet.empty[Path]
      val supportArtifactsPath = Paths.get(workspace.supportArtifactsPath)
      def moveFile(file: Path): Unit = {
        // Convert the files to an absolute version and a relative version.
        val (_abs, rel) = file match {
          case file if file.startsWith(supportArtifactsPath) =>
            (file, file.subpath(supportArtifactsPath.getNameCount(), file.getNameCount()))
          case file => (supportArtifactsPath.resolve(file), file)
        }
        // Normalize the absolute path so it can be checked if it has already
        // been moved.
        val abs = _abs.normalize()
        // Move the file into primarySourcesPath if it has not already been moved.
        (abs, rel) match {
          case (abs, _) if movedFiles.contains(abs) =>
          case (abs, rel) =>
            val dest = Paths.get(workspace.primarySourcesPath).resolve(rel)
            dest.getParent.toFile.mkdirs
            Files.move(abs, dest)
            movedFiles += abs
        }
      }

      // Move all files from the build flow that have file extensions that
      // likley should be included:
      //   - Verilog files: '*.v', '*.sv', or '*.vh'.
      //   - C++ files: '.cc', '.cpp', '.h'
      val include_re = "^.*\\.(s?v|vh?|cc|cpp|h)$".r
      Files
        .walk(supportArtifactsPath)
        .filter(_.toFile.isFile)
        .filter(f => include_re.matches(f.getFileName.toString))
        .forEach(moveFile)

      GeneratedWorkspaceInfo(
        someDut.get(),
        someTestHarnesses.get(),
        outputAnnotations
      )
    }

    def getModuleInfoPorts(dut: RawModule): Seq[(Data, ModuleInfo.Port)] = {

      /**
          * We infer the names of various ports since we don't currently have a good alternative when using MFC. We hope to replace this once we get better support from CIRCT.
          */
      def leafPorts(node: Data, name: String): Seq[(Data, ModuleInfo.Port)] = {
        node match {
          case record: Record => {
            record.elements.toSeq.flatMap { case (fieldName, field) =>
              leafPorts(field, s"${name}_${fieldName}")
            }
          }
          case vec: Vec[_] => {
            vec.zipWithIndex.flatMap { case (element, index) =>
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
        case _                  => Nil
      }
    }

    def initializeModuleInfo(dut: RawModule, ports: Seq[ModuleInfo.Port]): Unit = {
      val info = ModuleInfo(
        name = dut.name,
        ports = ports
      )
      workspace.elaborate(info)
    }
  }
}
