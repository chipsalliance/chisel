// See LICENSE for license details.

package firrtlTests

import firrtl.{ChirrtlForm, CircuitState, Compiler, annotations}

import scala.concurrent.{Future, Await, ExecutionContext}
import scala.concurrent.duration.Duration

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
      new firrtl.LowFirrtlCompiler,
      new firrtl.VerilogCompiler)
    val inputFilePath = s"/integration/GCDTester.fir" // arbitrary
    val numThreads = 64 // arbitrary

    // Begin the actual test
    val inputStream = getClass().getResourceAsStream(inputFilePath)
    val inputStrings = io.Source.fromInputStream(inputStream).getLines().toList

    import ExecutionContext.Implicits.global
    try { // Use try-catch because error can manifest in many ways
      // Execute for each compiler
      val compilerResults = compilers map { compiler =>
        // Run compiler serially once
        val serialResult = runCompiler(inputStrings, compiler)
        Future {
          val threadFutures = (0 until numThreads) map { i =>
              Future {
                runCompiler(inputStrings, compiler) == serialResult
              }
            }
          Await.result(Future.sequence(threadFutures), Duration.Inf)
        }
      }
      val results = Await.result(Future.sequence(compilerResults), Duration.Inf)
      assert(results.flatten reduce (_ && _)) // check all true (ie. success)
    } catch {
      case _: Throwable => fail("The Compiler is not thread safe")
    }
  }
}
