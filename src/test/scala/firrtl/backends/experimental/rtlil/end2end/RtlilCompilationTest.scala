// SPDX-License-Identifier: Apache-2.0

package firrtl.backends.experimental.rtlil.end2end

import firrtl.stage.{FirrtlStage, OutputFileAnnotation}
import firrtl.util.BackendCompilationUtilities
import logger.LazyLogging
import org.scalatest.flatspec.AnyFlatSpec

import java.io.File
import scala.sys.process.{Process, ProcessLogger}

class RtlilCompilationTest extends AnyFlatSpec with LazyLogging {
  it should "generate valid RTLIL for AddNot" in { compileAndParse("AddNot") }
  it should "generate valid RTLIL for FPU" in { compileAndParse("FPU") }
  it should "generate valid RTLIL for HwachaSequencer" in { compileAndParse("HwachaSequencer") }
  it should "generate valid RTLIL for ICache" in { compileAndParse("ICache") }
  it should "generate valid RTLIL for Ops" in { compileAndParse("Ops") }
  it should "generate valid RTLIL for Rob" in { compileAndParse("Rob") }
  it should "generate valid RTLIL for RocketCore" in { compileAndParse("RocketCore") }

  private def compileAndParse(name: String): Unit = {
    val testDir = BackendCompilationUtilities.createTestDirectory(name + "-rtlil")
    val inputFile = new File(testDir, s"${name}.fir")
    BackendCompilationUtilities.copyResourceToFile(s"/regress/${name}.fir", inputFile)

    val args = Array(
      "-ll",
      "error", // surpress warnings to keep test output clean
      "--target-dir",
      testDir.toString,
      "-i",
      inputFile.toString,
      "-E",
      "experimental-rtlil",
      "-E",
      "verilog",
      "-E",
      "low"
    )
    val res = (new FirrtlStage).execute(args, Seq())
    val fileName = res.collectFirst { case OutputFileAnnotation(file) => file }.get

    val rtlilFile = testDir.toString + "/" + fileName + ".il"
    val log = ProcessLogger(_ => (), logger.error(_))
    //memory_collect is here to verify that the emitted $mem(wr|rd) cells are proper.
    val yosysRet = Process(Seq("yosys", "-p", s"read_rtlil ${rtlilFile}; memory_collect")).run(log).exitValue()
    assert(yosysRet == 0, s"Failed to parse RTLIL file $rtlilFile generated for $name")
  }
}
