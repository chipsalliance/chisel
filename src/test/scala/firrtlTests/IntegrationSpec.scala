// See LICENSE for license details.

package firrtlTests

import firrtl._
import org.scalatest._
import org.scalatest.prop._

import java.io.File

class GCDExecutionTest extends ExecutionTest("GCDTester", "/integration")
class RightShiftExecutionTest extends ExecutionTest("RightShiftTester", "/integration")
class MemExecutionTest extends ExecutionTest("MemTester", "/integration")
class PipeExecutionTest extends ExecutionTest("PipeTester", "/integration")

// This is a bit custom some kind of one off
class GCDSplitEmissionExecutionTest extends FirrtlFlatSpec {
  "GCDTester" should "work even when the modules are emitted to different files" in {
    val top = "GCDTester"
    val testDir = createTestDirectory("GCDTesterSplitEmission")
    val sourceFile = new File(testDir, s"$top.fir")
    copyResourceToFile(s"/integration/$top.fir", sourceFile)

    val optionsManager = new ExecutionOptionsManager("GCDTesterSplitEmission") with HasFirrtlOptions {
      commonOptions = CommonOptions(topName = top, targetDirName = testDir.getPath)
      firrtlOptions = FirrtlExecutionOptions(
                        inputFileNameOverride = sourceFile.getPath,
                        compilerName = "verilog",
                        infoModeName = "ignore",
                        emitOneFilePerModule = true)
    }
    firrtl.Driver.execute(optionsManager)

    // expected filenames
    val dutFile = new File(testDir, "DecoupledGCD.v")
    val topFile = new File(testDir, s"$top.v")
    dutFile should exist
    topFile should exist

    // Copy harness over
    val harness = new File(testDir, s"testTop.cpp")
    copyResourceToFile(cppHarnessResourceName, harness)

    // topFile will be compiled by Verilator command by default but we need to also include dutFile
    verilogToCpp(top, testDir, Seq(dutFile), harness).!
    cppToExe(top, testDir).!
    assert(executeExpectingSuccess(top, testDir))
  }
}

class RobCompilationTest extends CompilationTest("Rob", "/regress")
class RocketCoreCompilationTest extends CompilationTest("RocketCore", "/regress")
class ICacheCompilationTest extends CompilationTest("ICache", "/regress")
class FPUCompilationTest extends CompilationTest("FPU", "/regress")

