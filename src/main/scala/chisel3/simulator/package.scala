package chisel3

import svsim._
import chisel3.reflect.DataMirror
import scala.collection.mutable
import java.nio.file.{Files, Path, Paths}

package object simulator {
  implicit class SimulationController(controller: Simulation.Controller) {
    def port(data: Data): Simulation.Port = {
      val context = Simulator.dynamicSimulationContext.value.get
      assert(context.controller == controller)
      context.simulationPorts(data)
    }
  }

  implicit class ChiselWorkspace(workspace: Workspace) {
    def elaborateGeneratedModule[T <: RawModule](
      generateModule: () => T
    ): T = {
      elaborateGeneratedModuleInternal(generateModule)._1
    }
    private[simulator] def elaborateGeneratedModuleInternal[T <: RawModule](
      generateModule: () => T
    ): (T, Seq[(Data, ModuleInfo.Port)]) = {
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
              DataMirror.directionOf(element) match {
                case ActualDirection.Input =>
                  Seq((element, ModuleInfo.Port(name, isGettable = true, isSettable = true)))
                case ActualDirection.Output => Seq((element, ModuleInfo.Port(name, isGettable = true)))
                case _                      => Seq()
              }
          }
        }
        DataMirror.modulePorts(dut).flatMap {
          case (name, data) => leafPorts(data, name)
        }
      }
      workspace.elaborate(
        ModuleInfo(
          name = dut.name,
          ports = ports.map(_._2)
        )
      )
      (dut, ports)
    }
  }
}
