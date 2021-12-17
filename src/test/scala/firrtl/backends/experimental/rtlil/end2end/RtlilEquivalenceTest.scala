// SPDX-License-Identifier: Apache-2.0

package firrtl.backends.experimental.rtlil.end2end

import firrtl.stage.{FirrtlStage, OutputFileAnnotation}
import firrtl.util.BackendCompilationUtilities
import logger.LazyLogging
import org.scalatest.flatspec.AnyFlatSpec

import java.io.File
import scala.sys.process.{Process, ProcessLogger}

class RtlilEquivalenceTest extends AnyFlatSpec with LazyLogging {
  it should "generate identical RTLIL for Can_Core" in { compileAndParse("CanTop") }
  it should "generate identical RTLIL for RotationCordic" in { compileAndParse("RotationCordic") }

  private def compileAndParse(name: String): Unit = {
    val testDir = BackendCompilationUtilities.createTestDirectory(name + "-rtlil")
    val inputFile = new File(testDir, s"${name}.fir")
    BackendCompilationUtilities.copyResourceToFile(s"/rtlil_equiv_check/${name}.fir", inputFile)

    val args = Array(
      "-ll",
      "error",
      "--target-dir",
      testDir.toString,
      "-i",
      inputFile.toString,
      "-E",
      "experimental-rtlil",
      "-E",
      "low",
      "-E",
      "verilog"
    )
    val res = (new FirrtlStage).execute(args, Seq())
    val fileName = res.collectFirst { case OutputFileAnnotation(file) => file }.get

    val rtlilFile = testDir.toString + "/" + fileName + ".il"
    val verilogFile = testDir.toString + "/" + fileName + ".v"

    val log = ProcessLogger(
      logger.info(_),
      logger.warn(_)
    )

    val yosysArgs = Array(
      s"read_rtlil ${rtlilFile};",
      s"prep -flatten -top ${name};",
      "design -stash gate;",
      s"read_verilog ${verilogFile};",
      s"prep -flatten -top ${name};",
      "design -stash gold;",
      s"design -copy-from gold -as gold ${name};",
      s"design -copy-from gate -as gate ${name};",
      "select gold gate;",
      "opt -full -fine;",
      "memory;",
      "async2sync;",
      "equiv_make gold gate equiv;",
      "select equiv;",
      "equiv_struct;",
      "equiv_simple -seq 1;",
      "equiv_status;",
      "equiv_induct -seq 1 -undef;",
      "equiv_status -assert"
    )
    val yosysRet = Process(Seq("yosys", "-p", yosysArgs.mkString(" "))).run(log).exitValue()
    assert(yosysRet == 0, s"Unable to prove equivalence of design ${name}.")
  }
}
