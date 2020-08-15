// See LICENSE for license details.

package firrtl.backends.experimental.smt.end2end

import java.io.File

import firrtl.stage.{FirrtlStage, OutputFileAnnotation}
import firrtl.util.BackendCompilationUtilities
import logger.LazyLogging
import org.scalatest.flatspec.AnyFlatSpec

import scala.sys.process.{Process, ProcessLogger}

/** compiles the regression tests to SMTLib and parses the result with z3 */
class SMTCompilationTest extends AnyFlatSpec with LazyLogging {
  it should "generate valid SMTLib for AddNot" taggedAs(RequiresZ3) in { compileAndParse("AddNot") }
  it should "generate valid SMTLib for FPU" taggedAs(RequiresZ3) in { compileAndParse("FPU") }
  // we get a stack overflow in Scala 2.11 because of a deeply nested and(...) expression in the sequencer
  it should "generate valid SMTLib for HwachaSequencer" taggedAs(RequiresZ3) ignore { compileAndParse("HwachaSequencer") }
  it should "generate valid SMTLib for ICache" taggedAs(RequiresZ3) in { compileAndParse("ICache") }
  it should "generate valid SMTLib for Ops" taggedAs(RequiresZ3) in { compileAndParse("Ops") }
  // TODO: enable Rob test once we support more than 2 write ports on a memory
  it should "generate valid SMTLib for Rob" taggedAs(RequiresZ3) ignore { compileAndParse("Rob") }
  it should "generate valid SMTLib for RocketCore" taggedAs(RequiresZ3) in { compileAndParse("RocketCore") }

  private def compileAndParse(name: String): Unit = {
    val testDir = BackendCompilationUtilities.createTestDirectory(name + "-smt")
    val inputFile = new File(testDir, s"${name}.fir")
    BackendCompilationUtilities.copyResourceToFile(s"/regress/${name}.fir", inputFile)

    val args = Array(
      "-ll", "error", // surpress warnings to keep test output clean
      "--target-dir", testDir.toString,
      "-i", inputFile.toString,
      "-E", "experimental-smt2"
      // "-fct", "firrtl.backends.experimental.smt.StutteringClockTransform"
    )
    val res = (new FirrtlStage).execute(args, Seq())
    val fileName = res.collectFirst{ case OutputFileAnnotation(file) => file }.get

    val smtFile = testDir.toString + "/" + fileName + ".smt2"
    val log = ProcessLogger(_ => (), logger.error(_))
    val z3Ret = Process(Seq("z3", smtFile)).run(log).exitValue()
    assert(z3Ret == 0, s"Failed to parse SMTLib file $smtFile generated for $name")
  }
}
