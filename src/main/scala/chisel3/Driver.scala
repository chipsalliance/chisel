// SPDX-License-Identifier: Apache-2.0

package chisel3

import internal.firrtl._
import firrtl._
import firrtl.util.{BackendCompilationUtilities => FirrtlBackendCompilationUtilities}
import java.io._
import _root_.logger.LazyLogging

@deprecated("Use object firrtl.util.BackendCompilationUtilities instead", "Chisel 3.5")
trait BackendCompilationUtilities extends LazyLogging {

  import scala.sys.process.{ProcessBuilder, ProcessLogger, _}

  // Inlined from old trait firrtl.util.BackendCompilationUtilities
  lazy val TestDirectory = FirrtlBackendCompilationUtilities.TestDirectory
  def timeStamp:            String = FirrtlBackendCompilationUtilities.timeStamp
  def loggingProcessLogger: ProcessLogger = FirrtlBackendCompilationUtilities.loggingProcessLogger
  def copyResourceToFile(name:      String, file: File): Unit = FirrtlBackendCompilationUtilities.copyResourceToFile(name, file)
  def createTestDirectory(testName: String): File = FirrtlBackendCompilationUtilities.createTestDirectory(testName)
  def makeHarness(template:         String => String, post: String)(f: File): File =
    FirrtlBackendCompilationUtilities.makeHarness(template, post)(f)
  def firrtlToVerilog(prefix: String, dir: File): ProcessBuilder =
    FirrtlBackendCompilationUtilities.firrtlToVerilog(prefix, dir)
  def verilogToCpp(
    dutFile:          String,
    dir:              File,
    vSources:         Seq[File],
    cppHarness:       File,
    suppressVcd:      Boolean = false,
    resourceFileName: String = firrtl.transforms.BlackBoxSourceHelper.defaultFileListName
  ): ProcessBuilder = {
    FirrtlBackendCompilationUtilities.verilogToCpp(dutFile, dir, vSources, cppHarness, suppressVcd, resourceFileName)
  }
  def cppToExe(prefix: String, dir: File): ProcessBuilder = FirrtlBackendCompilationUtilities.cppToExe(prefix, dir)
  def executeExpectingFailure(
    prefix:       String,
    dir:          File,
    assertionMsg: String = ""
  ): Boolean = {
    FirrtlBackendCompilationUtilities.executeExpectingFailure(prefix, dir, assertionMsg)
  }
  def executeExpectingSuccess(prefix: String, dir: File): Boolean =
    FirrtlBackendCompilationUtilities.executeExpectingSuccess(prefix, dir)

  /** Compile Chirrtl to Verilog by invoking Firrtl inside the same JVM
    *
    * @param prefix basename of the file
    * @param dir    directory where file lives
    * @return       true if compiler completed successfully
    */
  @deprecated("Use ChiselStage instead", "Chisel 3.5")
  def compileFirrtlToVerilog(prefix: String, dir: File): Boolean = {

    // ====== Implemented by inlining logic from ExecutionsOptionManager.toAnnotations =====
    import firrtl.stage.InfoModeAnnotation
    import firrtl.stage.phases.DriverCompatibility.TopNameAnnotation
    import _root_.logger.LogLevelAnnotation
    val annos: AnnotationSeq = List(
      InfoModeAnnotation("append"),
      TopNameAnnotation(prefix),
      TargetDirAnnotation(dir.getAbsolutePath),
      LogLevelAnnotation(_root_.logger.LogLevel.None)
    )

    // ******************* Implemented by inlining firrtl.Driver.execute ***************************
    import firrtl.stage.phases.DriverCompatibility
    import firrtl.stage.FirrtlStage
    import firrtl.options.{Dependency, Phase, PhaseManager}
    import firrtl.options.phases.DeletedWrapper

    val phases: Seq[Phase] = {
      import DriverCompatibility._
      new PhaseManager(
        List(
          Dependency[AddImplicitFirrtlFile],
          Dependency[AddImplicitAnnotationFile],
          Dependency[AddImplicitOutputFile],
          Dependency[AddImplicitEmitter],
          Dependency[FirrtlStage]
        )
      ).transformOrder
        .map(DeletedWrapper(_))
    }

    val annosx =
      try {
        phases.foldLeft(annos)((a, p) => p.transform(a))
      } catch {
        case _: firrtl.options.OptionsException => return false
      }
    // *********************************************************************************************

    val options = annosx

    // ********** Implemented by inlining firrtl.stage.FirrtlExecutionResultView.view **************
    import firrtl.stage.FirrtlCircuitAnnotation

    options.collectFirst { case a: FirrtlCircuitAnnotation => a.circuit } match {
      case None => false
      case Some(_) => true
    }
    // *********************************************************************************************
  }
}

