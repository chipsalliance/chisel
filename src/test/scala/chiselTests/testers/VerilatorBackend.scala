// SPDX-License-Identifier: Apache-2.0

package chiselTests.testers

import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.stage.phases.{Convert, Emitter}
import chiselTests.testers.TesterDriver.{AddImplicitTesterDirectory, cppToExe, executeExpectingSuccess, finishWrapper, verilogToCpp}
import firrtl.AnnotationSeq
import firrtl.options.{Dependency, PhaseManager, TargetDirAnnotation}
import firrtl.stage.{FirrtlCircuitAnnotation, FirrtlStage}
import firrtl.transforms.BlackBoxSourceHelper.writeResourceToDirectory
import firrtl.util.BackendCompilationUtilities.copyResourceToFile

import java.io._

case object VerilatorBackend extends TesterDriver.Backend {
  /** For use with modules that should successfully be elaborated by the
    * frontend, and which can be turned into executables with assertions. */
  def execute(t: () => BasicTester,
              additionalVResources: Seq[String] = Seq(),
              annotations: AnnotationSeq = Seq(),
              nameHint: Option[String] = None
             ): Boolean = {
    val pm = new PhaseManager(
      targets = Seq(Dependency[AddImplicitTesterDirectory],
        Dependency[Emitter],
        Dependency[Convert]))

    val annotationsx = pm.transform(ChiselGeneratorAnnotation(finishWrapper(t)) +: annotations)

    val target: String = annotationsx.collectFirst { case FirrtlCircuitAnnotation(cir) => cir.main }.get
    val path = annotationsx.collectFirst { case TargetDirAnnotation(dir) => dir }.map(new File(_)).get

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

    (new FirrtlStage).execute(Array("--compiler", "verilog"), annotationsx)

    // Use sys.Process to invoke a bunch of backend stuff, then run the resulting exe
    if ((verilogToCpp(target, path, additionalVFiles, cppHarness) #&&
      cppToExe(target, path)).! == 0) {
      executeExpectingSuccess(target, path)
    } else {
      false
    }
  }
}