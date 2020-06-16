// See LICENSE for license details.

package chisel3.testers

import chisel3._
import java.io._

import chisel3.aop.Aspect
import chisel3.experimental.RunFirrtlTransform
import chisel3.stage.phases.{AspectPhase, Convert, Elaborate, Emitter}
import chisel3.stage.{
  ChiselCircuitAnnotation,
  ChiselGeneratorAnnotation,
  ChiselOutputFileAnnotation,
  ChiselStage,
  DesignAnnotation
}
import firrtl.{Driver => _, _}
import firrtl.options.{Dependency, Phase, PhaseManager}
import firrtl.stage.{FirrtlCircuitAnnotation, FirrtlStage}
import firrtl.transforms.BlackBoxSourceHelper.writeResourceToDirectory

object TesterDriver extends BackendCompilationUtilities {

  /** Set the target directory to the name of the top module after elaboration */
  final class AddImplicitTesterDirectory extends Phase {
    override def prerequisites = Seq(Dependency[Elaborate])
    override def optionalPrerequisites = Seq.empty
    override def optionalPrerequisiteOf = Seq(Dependency[Emitter])
    override def invalidates(a: Phase) = false

    override def transform(a: AnnotationSeq) = a.flatMap {
      case a@ ChiselCircuitAnnotation(circuit) =>
        Seq(a, TargetDirAnnotation(
              firrtl.util.BackendCompilationUtilities.createTestDirectory(circuit.name)
                .getAbsolutePath
                .toString))
      case a => Seq(a)
    }
  }

  /** For use with modules that should successfully be elaborated by the
    * frontend, and which can be turned into executables with assertions. */
  def execute(t: () => BasicTester,
              additionalVResources: Seq[String] = Seq(),
              annotations: AnnotationSeq = Seq()
             ): Boolean = {
    val pm = new PhaseManager(
      targets = Seq(Dependency[AddImplicitTesterDirectory],
                    Dependency[Emitter],
                    Dependency[Convert]))

    val annotationsx = pm.transform(ChiselGeneratorAnnotation(t) +: annotations)

    val target: String = annotationsx.collectFirst { case FirrtlCircuitAnnotation(cir) => cir.main }.get
    val path = annotationsx.collectFirst { case TargetDirAnnotation(dir) => dir }.map(new File(_)).get

    // Copy CPP harness and other Verilog sources from resources into files
    val cppHarness =  new File(path, "top.cpp")
    copyResourceToFile("/chisel3/top.cpp", cppHarness)
    // NOTE: firrtl.Driver.execute() may end up copying these same resources in its BlackBoxSourceHelper code.
    // As long as the same names are used for the output files, and we avoid including duplicate files
    //  in BackendCompilationUtilities.verilogToCpp(), we should be okay.
    // To that end, we use the same method to write the resource to the target directory.
    val additionalVFiles = additionalVResources.map((name: String) => {
      writeResourceToDirectory(name, path)
    })

    (new FirrtlStage).execute(Array("--compiler", "verilog"), annotationsx)

    // Use sys.Process to invoke a bunch of backend stuff, then run the resulting exe
    if ((verilogToCpp(target, path, additionalVFiles, cppHarness) #&&
        cppToExe(target, path)).! == 0) {
      executeExpectingSuccess(target, path)
    } else {
      false
    }
  }
  /**
    * Calls the finish method of an BasicTester or a class that extends it.
    * The finish method is a hook for code that augments the circuit built in the constructor.
    */
  def finishWrapper(test: () => BasicTester): () => BasicTester = {
    () => {
      val tester = test()
      tester.finish()
      tester
    }
  }
}
