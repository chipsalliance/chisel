// SPDX-License-Identifier: Apache-2.0

package firrtl.fuzzer

import com.pholser.junit.quickcheck.From
import com.pholser.junit.quickcheck.generator.{Generator, GenerationStatus}
import com.pholser.junit.quickcheck.random.SourceOfRandomness

import edu.berkeley.cs.jqf.fuzz.{Fuzz, JQF};

import firrtl._
import firrtl.annotations.{Annotation, CircuitTarget, ModuleTarget, Target}
import firrtl.ir.Circuit
import firrtl.options.Dependency
import firrtl.stage.{FirrtlCircuitAnnotation, InfoModeAnnotation, OutputFileAnnotation, TransformManager}
import firrtl.stage.Forms.{VerilogMinimumOptimized, VerilogOptimized}
import firrtl.stage.phases.WriteEmitted
import firrtl.transforms.{InlineBooleanExpressions, ManipulateNames}
import firrtl.util.BackendCompilationUtilities

import java.io.{File, FileWriter, PrintWriter, StringWriter}
import java.io.{File, FileWriter}

import org.junit.Assert
import org.junit.runner.RunWith

object FirrtlEquivalenceTestUtils {

  private class AddSuffixToTop(suffix: String) extends ManipulateNames[AddSuffixToTop] {
    override def manipulate = (a: String, b: Namespace) => Some(b.newName(a + suffix))

    override def execute(state: CircuitState): CircuitState = {
      val block = (_: Target) => false
      val allow: Target => Boolean = {
        case _: ModuleTarget => true
        case _: CircuitTarget => true
        case _: Target => false
      }
      val renames = RenameMap()
      val circuitx = run(state.circuit, renames, block, allow)
      state.copy(circuit = circuitx, renames = Some(renames))
    }
  }

  private def writeEmitted(state: CircuitState, outputFile: String): Unit = {
    (new WriteEmitted).transform(state.annotations :+ OutputFileAnnotation(outputFile))
  }

  def firrtlEquivalenceTestPass(
    circuit: Circuit,
    referenceCompiler: TransformManager,
    referenceAnnos: Seq[Annotation],
    customCompiler: TransformManager,
    customAnnos: Seq[Annotation],
    testDir: File,
    timesteps: Int = 1): Boolean = {
    val baseAnnos = Seq(
      InfoModeAnnotation("ignore"),
      FirrtlCircuitAnnotation(circuit)
    )

    testDir.mkdirs()

    val customTransforms = Seq(
      customCompiler,
      new AddSuffixToTop("_custom"),
      new VerilogEmitter
    )
    val customResult = customTransforms.foldLeft(CircuitState(
      circuit,
      ChirrtlForm,
      baseAnnos ++: EmitCircuitAnnotation(classOf[VerilogEmitter]) +: customAnnos
    )) { case (state, transform) => transform.transform(state) }
    val customName = customResult.circuit.main
    val customOutputFile = new File(testDir, s"$customName.v")
    writeEmitted(customResult, customOutputFile.toString)

    val referenceTransforms = Seq(
      referenceCompiler,
      new AddSuffixToTop("_reference"),
      new MinimumVerilogEmitter
    )
    val referenceResult = referenceTransforms.foldLeft(CircuitState(
      circuit,
      ChirrtlForm,
      baseAnnos ++: EmitCircuitAnnotation(classOf[MinimumVerilogEmitter]) +: referenceAnnos
    )) { case (state, transform) => transform.transform(state) }
    val referenceName = referenceResult.circuit.main
    val referenceOutputFile = new File(testDir, s"$referenceName.v")
    writeEmitted(referenceResult, referenceOutputFile.toString)

    BackendCompilationUtilities.yosysExpectSuccess(customName, referenceName, testDir, timesteps)
  }
}

import ExprGen._
class InlineBooleanExprsCircuitGenerator extends SingleExpressionCircuitGenerator (
  ExprGenParams(
    maxDepth = 50,
    maxWidth = 31,
    generators = ExprGenParams.defaultGenerators ++ Map(
      LtDoPrimGen -> 10,
      LeqDoPrimGen -> 10,
      GtDoPrimGen -> 10,
      GeqDoPrimGen -> 10,
      EqDoPrimGen -> 10,
      NeqDoPrimGen -> 10,
      AndDoPrimGen -> 10,
      OrDoPrimGen -> 10,
      XorDoPrimGen -> 10,
      AndrDoPrimGen -> 10,
      OrrDoPrimGen -> 10,
      XorrDoPrimGen -> 10,
      BitsDoPrimGen -> 10,
      HeadDoPrimGen -> 10,
      TailDoPrimGen -> 10,
      MuxGen -> 10
    )
  )
)

@RunWith(classOf[JQF])
class FirrtlEquivalenceTests {
  private val lowFirrtlCompiler = new LowFirrtlCompiler()
  private val header = "=" * 50 + "\n"
  private val footer = header
  private def message(c: Circuit, t: Throwable): String = {
    val sw = new StringWriter()
    val pw = new PrintWriter(sw)
    t.printStackTrace(pw)
    pw.flush()
    header + c.serialize + "\n" + sw.toString + footer
  }
  private val baseTestDir = new File("fuzzer/test_run_dir")

  private def runTest(c: Circuit, referenceCompiler: TransformManager, customCompiler: TransformManager) = {
    val testDir = new File(baseTestDir, f"${c.hashCode}%08x")
    testDir.mkdirs()
    val fileWriter = new FileWriter(new File(testDir, s"${c.main}.fir"))
    fileWriter.write(c.serialize)
    fileWriter.close()
    val passed = try {
      FirrtlEquivalenceTestUtils.firrtlEquivalenceTestPass(
        circuit = c,
        referenceCompiler = referenceCompiler,
        referenceAnnos = Seq(),
        customCompiler = customCompiler,
        customAnnos = Seq(),
        testDir = testDir
      )
    } catch {
      case e: Throwable => {
        Assert.assertTrue(s"exception thrown on input ${testDir}:\n${message(c, e)}", false)
        throw e
      }
    }

    if (!passed) {
      Assert.assertTrue(
        s"not equivalent to reference compiler on input ${testDir}:\n${c.serialize}\n", false)
    }
  }

  @Fuzz
  def testOptimized(@From(value = classOf[FirrtlCompileCircuitGenerator]) c: Circuit) = {
    runTest(
      c = c,
      referenceCompiler = new TransformManager(VerilogMinimumOptimized),
      customCompiler = new TransformManager(VerilogOptimized)
    )
  }

  @Fuzz
  def testInlineBooleanExpressions(@From(value = classOf[InlineBooleanExprsCircuitGenerator]) c: Circuit) = {
    runTest(
      c = c,
      referenceCompiler = new TransformManager(VerilogMinimumOptimized),
      customCompiler = new TransformManager(VerilogMinimumOptimized :+ Dependency[InlineBooleanExpressions])
    )
  }
}
