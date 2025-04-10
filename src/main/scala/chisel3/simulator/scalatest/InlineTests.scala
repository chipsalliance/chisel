package chisel3.simulator.scalatest

import scala.util.control.NoStackTrace
import org.scalatest.ConfigMap
import org.scalatest.funspec.AnyFunSpec
import chisel3._
import chisel3.experimental.inlinetest._
import firrtl.options.StageUtils.dramaticMessage

private case class FailedTest(
  testName: String,
  expected: TestResult.Type,
  actual:   TestResult.Type
)

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
      val results = simulateTests(gen, tests, timeout)

      val failures: Seq[FailedTest] =
        results.flatMap { result =>
          val expected = result.elaboratedTest.expectedResult
          val actual = result.actualResult
          Option.when(actual != expected) {
            FailedTest(result.elaboratedTest.params.testName, expected, actual)
          }
        }

      if (failures.nonEmpty) {
        val failuresList = failures.map { case FailedTest(testName, expected, actual) =>
          val expectedVerbPhrase = expected match {
            case TestResult.Success => "succeed"
            case failure: TestResult.Failure => s"${failure.description}"
          }
          val actualVerbPhrase = actual match {
            case TestResult.Success => "succeeded"
            case failure: TestResult.Failure => s"${failure.description}"
          }
          s"${testName} expected ${expectedVerbPhrase}, saw ${actualVerbPhrase}"
        }.mkString("\n")
        fail(failuresList)
      }
    }
  }
}
