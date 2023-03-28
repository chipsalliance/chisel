package chisel3

import svsim._
import chisel3.reflect.DataMirror

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

      // Move the relevant files over to primary-sources
      val filelist =
        new java.io.BufferedReader(new java.io.FileReader(s"${workspace.supportArtifactsPath}/filelist.f"))
      try {
        filelist.lines().forEach { immutableFilename =>
          var filename = immutableFilename
          /// Some files are provided as absolute paths
          if (filename.startsWith(workspace.supportArtifactsPath)) {
            filename = filename.substring(workspace.supportArtifactsPath.length + 1)
          }
          java.nio.file.Files.move(
            java.nio.file.Paths.get(s"${workspace.supportArtifactsPath}/$filename"),
            java.nio.file.Paths.get(s"${workspace.primarySourcesPath}/$filename")
          )
        }
      } finally {
        filelist.close()
      }

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
                case ActualDirection.Input  => Seq((element, ModuleInfo.Port(name, isSettable = true)))
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
