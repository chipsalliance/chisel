// SPDX-License-Identifier: Apache-2.0

package chisel3.testers

import chisel3._
import chisel3.stage.phases.{Convert, Elaborate, Emitter, MaybeInjectingPhase}
import chisel3.stage.{ChiselCircuitAnnotation, ChiselGeneratorAnnotation}
import firrtl.AnnotationSeq
import firrtl.annotations.NoTargetAnnotation
import firrtl.options.{Dependency, Phase, PhaseManager, TargetDirAnnotation, Unserializable}
import firrtl.stage.FirrtlCircuitAnnotation
import firrtl.transforms.BlackBoxSourceHelper.writeResourceToDirectory

import java.io._
import scala.annotation.nowarn
import scala.sys.process.ProcessLogger

@nowarn("msg=trait BackendCompilationUtilities in package chisel3 is deprecated")
object TesterDriver extends BackendCompilationUtilities {
  //TODO: need to remove BackendCompilationUtilities here but it will break external API
  //      unless all methods of it are implemented

  private[chisel3] trait Backend extends NoTargetAnnotation with Unserializable {
    def execute(
      t:                    () => BasicTester,
      additionalVResources: Seq[String] = Seq(),
      annotations:          AnnotationSeq = Seq(),
      nameHint:             Option[String] = None,
      processLogger:        ProcessLogger = loggingProcessLogger
    ): Boolean
  }
  case object VerilatorBackend extends Backend {
    def ensureExistingAbsolutePath(name: String): os.Path = {
      val otherPath: os.Path =
        try {
          os.pwd / os.RelPath(name)
        } catch {
          case t: Throwable =>
            os.Path(name)
        }
      if (!os.exists(otherPath)) {
        println(
          s"Can't find '$name' check command line"
        )
        throw new Exception(s"Can't find path for '$name'")
      }
      otherPath
    }

    /** For use with modules that should successfully be elaborated by the
      * frontend, and which can be turned into executables with assertions.
      */
    def execute(
      t:                    () => BasicTester,
      additionalVResources: Seq[String] = Seq(),
      annotations:          AnnotationSeq = Seq(),
      nameHint:             Option[String] = None,
      processLogger:        ProcessLogger = loggingProcessLogger
    ): Boolean = {
      val pm = new PhaseManager(
        targets = Seq(
          Dependency[AddImplicitTesterDirectory],
          Dependency[Emitter],
          Dependency[Convert],
          Dependency[MaybeInjectingPhase]
        )
      )

      val annotationsFromPhase1 = pm.transform(ChiselGeneratorAnnotation(finishWrapper(t)) +: annotations)

      val target: String = annotationsFromPhase1.collectFirst { case FirrtlCircuitAnnotation(cir) => cir.main }.get
      val path = annotationsFromPhase1.collectFirst { case TargetDirAnnotation(dir) => dir }.map(new File(_)).get

      // Copy CPP harness and other Verilog sources from resources into files
      val cppHarness = new File(path, "top.cpp")
      copyResourceToFile("/chisel3/top.cpp", cppHarness)
      // NOTE: firrtl.Driver.execute() may end up copying these same resources in its BlackBoxSourceHelper code.
      // As long as the same names are used for the output files, and we avoid including duplicate files
      //  in BackendCompilationUtilities.verilogToCpp(), we should be okay.
      // To that end, we use the same method to write the resource to the target directory.
      val additionalVFiles = additionalVResources.map((name: String) => {
        writeResourceToDirectory(name, path)
      })

      val dirName = annotationsFromPhase1.collectFirst { case TargetDirAnnotation(dirName) => dirName }.getOrElse(".")
      val verilog = circt.stage.ChiselStage.emitSystemVerilog(t())
      val verilogPath = ensureExistingAbsolutePath(path.toString) / (target + ".v")
      os.write.over(verilogPath, verilog)
      // Use sys.Process to invoke a bunch of backend stuff, then run the resulting exe
      if (
        (verilogToCpp(target, path, additionalVFiles, cppHarness) #&&
          cppToExe(target, path)).!(processLogger) == 0
      ) {
        executeExpectingSuccess(target, path)
      } else {
        false
      }
    }
  }

  val defaultBackend: Backend = VerilatorBackend

  /** Use this to force a test to be run only with backends that are restricted to verilator backend
    */
  def verilatorOnly: AnnotationSeq = Seq(VerilatorBackend)

  /** Set the target directory to the name of the top module after elaboration */
  final class AddImplicitTesterDirectory extends Phase {
    override def prerequisites = Seq(Dependency[Elaborate])
    override def optionalPrerequisites = Seq.empty
    override def optionalPrerequisiteOf = Seq(Dependency[Emitter])
    override def invalidates(a: Phase) = false

    override def transform(a: AnnotationSeq) = a.flatMap {
      case a @ ChiselCircuitAnnotation(circuit) =>
        Seq(
          a,
          TargetDirAnnotation(
            firrtl.util.BackendCompilationUtilities.createTestDirectory(circuit.name).getAbsolutePath.toString
          )
        )
      case a => Seq(a)
    }
  }

  /** For use with modules that should successfully be elaborated by the
    * frontend, and which can be turned into executables with assertions.
    */
  def execute(
    t:                    () => BasicTester,
    additionalVResources: Seq[String] = Seq(),
    annotations:          AnnotationSeq = Seq(),
    nameHint:             Option[String] = None,
    /** Logger used for forked processes, useful for capturing Verilator process stdout and stderr */
    processLogger: ProcessLogger = loggingProcessLogger
  ): Boolean = {

    val backendAnnotations = annotations.collect { case anno: Backend => anno }
    val backendAnnotation = if (backendAnnotations.length == 1) {
      backendAnnotations.head
    } else if (backendAnnotations.isEmpty) {
      defaultBackend
    } else {
      throw new ChiselException(s"Only one backend annotation allowed, found: ${backendAnnotations.mkString(", ")}")
    }
    backendAnnotation.execute(t, additionalVResources, annotations, nameHint, processLogger)
  }

  /**
    * Calls the finish method of an BasicTester or a class that extends it.
    * The finish method is a hook for code that augments the circuit built in the constructor.
    */
  def finishWrapper(test: () => BasicTester): () => BasicTester = { () =>
    {
      val tester = test()
      tester.finish()
      tester
    }
  }

}
