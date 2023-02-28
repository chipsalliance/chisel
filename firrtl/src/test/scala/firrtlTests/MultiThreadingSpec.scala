// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl.FileUtils
import firrtl.{ChirrtlForm, CircuitState}
import firrtl.testutils._

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}

class MultiThreadingSpec extends FirrtlPropSpec {

  // TODO Test with annotations and source locator
  property("The FIRRTL compiler should be thread safe") {
    // Run the compiler we're testing
    def runCompiler(input: Seq[String], compiler: firrtl.Compiler): String = {
      val parsedInput = firrtl.Parser.parse(input)
      val res = compiler.compileAndEmit(CircuitState(parsedInput, ChirrtlForm))
      res.getEmittedCircuit.value
    }
    // The parameters we're testing with
    val compilers = Seq(
      new firrtl.HighFirrtlCompiler,
      new firrtl.MiddleFirrtlCompiler,
      new firrtl.LowFirrtlCompiler
    )
    val inputFilePath = s"/integration/GCDTester.fir" // arbitrary
    val numThreads = 64 // arbitrary

    // Begin the actual test

    val inputStrings = FileUtils.getLinesResource(inputFilePath).toSeq

    import ExecutionContext.Implicits.global
    try { // Use try-catch because error can manifest in many ways
      // Execute for each compiler
      val compilerResults = compilers.map { compiler =>
        // Run compiler serially once
        val serialResult = runCompiler(inputStrings, compiler)
        Future {
          val threadFutures = (0 until numThreads).map { i =>
            Future {
              runCompiler(inputStrings, compiler) == serialResult
            }
          }
          Await.result(Future.sequence(threadFutures), Duration.Inf)
        }
      }
      val results = Await.result(Future.sequence(compilerResults), Duration.Inf)
      assert(results.flatten.reduce(_ && _)) // check all true (ie. success)
    } catch {
      case _: Throwable => fail("The Compiler is not thread safe")
    }
  }
}
