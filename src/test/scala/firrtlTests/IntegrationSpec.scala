// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl._
import firrtl.stage.FirrtlStage
import firrtl.testutils._

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

    (new FirrtlStage).execute(
      Array(
        "--target-dir",
        testDir.getPath,
        "--input-file",
        sourceFile.getPath,
        "--info-mode",
        "ignore",
        "--emit-modules",
        "verilog"
      ),
      Seq()
    )

    // expected filenames
    val dutFile = new File(testDir, "DecoupledGCD.v")
    val topFile = new File(testDir, s"$top.v")
    dutFile should exist
    topFile should exist

    // Copy harness over
    val harness = new File(testDir, s"testTop.cpp")
    copyResourceToFile(cppHarnessResourceName, harness)

    // topFile will be compiled by Verilator command by default but we need to also include dutFile
    verilogToCpp(top, testDir, Seq(dutFile), harness) #&&
      cppToExe(top, testDir) ! loggingProcessLogger
    assert(executeExpectingSuccess(top, testDir))
  }
}

class RobCompilationTest extends CompilationTest("Rob", "/regress")
class RocketCoreCompilationTest extends CompilationTest("RocketCore", "/regress")
class ICacheCompilationTest extends CompilationTest("ICache", "/regress")
class FPUCompilationTest extends CompilationTest("FPU", "/regress")
class HwachaSequencerCompilationTest extends CompilationTest("HwachaSequencer", "/regress")

abstract class CommonSubexprEliminationEquivTest(name: String, dir: String)
    extends EquivalenceTest(Seq(firrtl.passes.CommonSubexpressionElimination), name, dir)
abstract class DeadCodeEliminationEquivTest(name: String, dir: String)
    extends EquivalenceTest(Seq(new firrtl.transforms.DeadCodeElimination), name, dir)
abstract class ConstantPropagationEquivTest(name: String, dir: String)
    extends EquivalenceTest(Seq(new firrtl.transforms.ConstantPropagation), name, dir)
abstract class LowFirrtlOptimizationEquivTest(name: String, dir: String)
    extends EquivalenceTest(Seq(new LowFirrtlOptimization), name, dir)

class OpsCommonSubexprEliminationTest extends CommonSubexprEliminationEquivTest("Ops", "/regress")
class OpsDeadCodeEliminationTest extends DeadCodeEliminationEquivTest("Ops", "/regress")
class OpsConstantPropagationTest extends ConstantPropagationEquivTest("Ops", "/regress")
class OpsLowFirrtlOptimizationTest extends LowFirrtlOptimizationEquivTest("Ops", "/regress")
