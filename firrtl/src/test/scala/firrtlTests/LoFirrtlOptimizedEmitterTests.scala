// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl._
import firrtl.stage._
import firrtl.util.BackendCompilationUtilities
import org.scalatest.flatspec.AnyFlatSpec

class LoFirrtlOptimizedEmitterTests extends AnyFlatSpec {
  behavior.of("LoFirrtlOptimizedEmitter")

  it should "generate valid firrtl for AddNot" in { compileAndParse("AddNot") }
  it should "generate valid firrtl for FPU" in { compileAndParse("FPU") }
  it should "generate valid firrtl for HwachaSequencer" in { compileAndParse("HwachaSequencer") }
  it should "generate valid firrtl for ICache" in { compileAndParse("ICache") }
  it should "generate valid firrtl for Ops" in { compileAndParse("Ops") }
  it should "generate valid firrtl for Rob" in { compileAndParse("Rob") }
  it should "generate valid firrtl for RocketCore" in { compileAndParse("RocketCore") }

  private def compileAndParse(name: String): Unit = {
    val testDir = os.RelPath(
      BackendCompilationUtilities.createTestDirectory(
        "LoFirrtlOptimizedEmitter_should_generate_valid_firrtl_for" + name
      )
    )
    val inputFile = testDir / s"$name.fir"
    val outputFile = testDir / s"$name.opt.lo.fir"

    BackendCompilationUtilities.copyResourceToFile(s"/regress/${name}.fir", (os.pwd / inputFile).toIO)

    val stage = new FirrtlStage
    // run low-opt emitter
    val args = Array(
      "-ll",
      "error", // surpress warnings to keep test output clean
      "--target-dir",
      testDir.toString,
      "-i",
      inputFile.toString,
      "-E",
      "low-opt"
    )
    val res = stage.execute(args, Seq())

    // load in result to check
    stage.execute(Array("--target-dir", testDir.toString, "-i", outputFile.toString()), Seq())
  }
}
