package chiselTests

import chisel3.Module
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage}
import firrtl.annotations.Annotation
import firrtl.options.{OutputAnnotationFileAnnotation, TargetDirAnnotation}
import firrtl.stage.OutputFileAnnotation
import firrtl.util.BackendCompilationUtilities.timeStamp
import logger.{LazyLogging, LogLevel, LogLevelAnnotation}
import org.scalatest.flatspec.AnyFlatSpec
import os._
import scala.util.Properties

/** [[SMTModelCheckingSpec]] use z3 and [[firrtl.backends.experimental.smt]] library
  * to solve `assert/assume` in [[chisel3.experimental.verification]],
  * It is a copy&paste version from `firrtl.backends.experimental.smt.end2end.EndToEndSMTBaseSpec` from firrtl
  * Useful to check combinational logic and some small test.
  */
abstract class SMTModelCheckingSpec extends AnyFlatSpec {
  def success = MCSuccess

  def fail(k: Int) = MCFail(k)

  def test(dut: () => Module, name: String, expected: MCResult, kmax: Int = 0, annos: Seq[Annotation] = Seq()): Unit = {
    expected match {
      case MCFail(k) =>
        assert(kmax >= k, s"Please set a kmax that includes the expected failing step! ($kmax < $expected)")
      case _ =>
    }
    // @todo rewrite BackendCompilationUtilities
    val testBaseDir = os.pwd / "test_run_dir" / name
    os.makeDir.all(testBaseDir)
    val testDir = os.temp.dir(testBaseDir, timeStamp, deleteOnExit = false)
    val res = (new ChiselStage).execute(
      Array("-E", "experimental-smt2"),
      Seq(
        LogLevelAnnotation(LogLevel.Error), // silence warnings for tests
        ChiselGeneratorAnnotation(dut),
        TargetDirAnnotation(testDir.toString)
      ) ++ annos
    )
    val top = res.collectFirst{case OutputAnnotationFileAnnotation(top) => top}.get
    assert(res.collectFirst { case _: OutputFileAnnotation => true }.isDefined)
    val r = Z3ModelChecker.bmc(testDir, top, kmax)
    assert(r == expected)
  }
}

private object Z3ModelChecker extends LazyLogging {
  def bmc(testDir: Path, main: String, kmax: Int): MCResult = {
    assert(kmax >= 0 && kmax < 50, "Trying to keep kmax in a reasonable range.")
    Seq.tabulate(kmax + 1) { k =>
      val stepFile = testDir / s"${main}_step$k.smt2"
      os.copy(testDir / s"$main.smt2", stepFile)
      os.write.append(stepFile,
        s"""${step(main, k)}
           |(check-sat)
           |""".stripMargin)
      val success = executeStep(stepFile)
      if (!success) return MCFail(k)
    }
    MCSuccess
  }

  private def executeStep(path: Path): Boolean = {
    val (out, ret) = executeCmd(path.toString)
    assert(ret == 0, s"expected success (0), not $ret: `$out`\nz3 ${path.toString}")
    assert(out == "sat" + Properties.lineSeparator || out == "unsat" + Properties.lineSeparator, s"Unexpected output: $out")
    out == "unsat" + Properties.lineSeparator
  }

  private def executeCmd(cmd: String): (String, Int) = {
    val process = os.proc("z3", cmd).call(stderr = ProcessOutput.Readlines(logger.warn(_)))
    (process.out.chunks.mkString, process.exitCode)
  }

  private def step(main: String, k: Int): String = {
    // define all states
    (0 to k).map(ii => s"(declare-fun s$ii () $main$StateTpe)") ++
      // assert that init holds in state 0
      List(s"(assert ($main$Init s0))") ++
      // assert transition relation
      (0 until k).map(ii => s"(assert ($main$Transition s$ii s${ii + 1}))") ++
      // assert that assumptions hold in all states
      (0 to k).map(ii => s"(assert ($main$Assumes s$ii))") ++
      // assert that assertions hold for all but last state
      (0 until k).map(ii => s"(assert ($main$Asserts s$ii))") ++
      // check to see if we can violate the assertions in the last state
      List(s"(assert (not ($main$Asserts s$k)))")
  }.mkString("\n")

  // the following suffixes have to match the ones in [[SMTTransitionSystemEncoder]]
  private val Transition = "_t"
  private val Init = "_i"
  private val Asserts = "_a"
  private val Assumes = "_u"
  private val StateTpe = "_s"
}
sealed trait MCResult

case object MCSuccess extends MCResult

case class MCFail(k: Int) extends MCResult

