package chisel3.simulator.scalatest

import scala.util.control.NoStackTrace
import org.scalatest.ConfigMap
import org.scalatest.funspec.AnyFunSpec
import chisel3._
import chisel3.experimental.inlinetest._
import firrtl.options.StageUtils.dramaticMessage

/** Provides APIs for running inline tests inside of a scalatest spec. */
trait InlineTests extends AnyFunSpec with ChiselSim {
  def runInlineTests(
    parametrizationName: String
  )(
    gen: => RawModule with HasTests
  )(
    tests:   TestChoice.Type = TestChoice.All,
    timeout: Int = 1000
  ): Unit = {
    val testChoicePhrase = tests match {
      case TestChoice.All              => "all tests"
      case TestChoice.Names(testNames) => "tests named " + testNames.map(n => s"'${n}'").mkString(", ")
      case TestChoice.Globs(globs)     => "tests matching " + globs.map(g => s"'${g}'").mkString(", ")
    }

    it(s"should pass ${testChoicePhrase} for ${parametrizationName}") {
      val simulated = simulateTests(gen, tests, timeout)
      val failures = simulated.filter(!_.success)
      if (failures.nonEmpty) {
        fail(failures.map { failure =>
          s"\t- ${failure.testName} failed: ${failure.result.asInstanceOf[TestResult.Failure].message}"
        }.mkString("\n"))
      }
    }
  }
}
