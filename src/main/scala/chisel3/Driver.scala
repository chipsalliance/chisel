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
  def copyResourceToFile(name: String, file: File): Unit =
    FirrtlBackendCompilationUtilities.copyResourceToFile(name, file)
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

}
