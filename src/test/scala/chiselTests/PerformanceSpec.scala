// See LICENSE for license details.

package chiselTests

import org.scalatest.time.SpanSugar._
import org.scalatest.concurrent.ScalaFutures
import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global

import chisel3._
import firrtl._

/** Parent class for tests that check for bad performance bugs in Chisel and Firrtl */
abstract class PerformancePathologySpec extends ChiselFlatSpec with ScalaFutures {
  /** Name to be printed by ScalaTest
    *
    * Default is this.getClass.getSimpleName
    */
  def name: String = this.getClass.getSimpleName
  /** Should the time of each run be printed
    *
    * Default is false
    */
  def displayRuntime: Boolean = false

  /** Timeout in seconds */
  def timeout: Int
  require(timeout > 0, "Timeout must be a positive number of seconds")
  /** Configuration to use for warming up */
  def warmupConfig: () => Module
  /** Configuration to use for benchmarking */
  def benchmarkConfig: () => Module

  /** Firrtl Resulting from Chisel invocation */
  def firrtlResult: String = _firrtlResult
  /** Verilog Resulting from Firrtl invocation */
  def verilogResult: String = _verilogResult

  // Useful timing function to help decide what timeout to give
  private def time[R](name: String)(block: => R): R = {
    val t0 = if (displayRuntime) {
      println(s"Starting $name")
      System.nanoTime()
    } else 0.0
    val result = block
    if (displayRuntime) {
      val t1 = System.nanoTime()
      println(s"Finished $name")
      val timeMillis = (t1 - t0) / 1000000.0
      println(f"$name took $timeMillis%.1f ms\n")
    }
    result
  }

  private var _firrtlResult = ""
  private var _verilogResult = ""

  name should s"complete within $timeout seconds" in {
    val manager = new ExecutionOptionsManager("perf") with HasFirrtlOptions with HasChiselExecutionOptions

    // Warmup
    for (i <- 0 until 5) {
      time(s"Warmup $i") { chisel3.Driver.execute(manager, warmupConfig) }
    }

    val result = Future {
      time(s"Benchmarked run") { chisel3.Driver.execute(manager, benchmarkConfig) }
    }
    assert(result.isReadyWithin(timeout seconds)) // Do timeout

    // TODO replace with something more Scala-y
    result foreach { // Execute upon success
      case ChiselExecutionSucccess(_, fresult, Some(firrtlExecutionResult)) =>
        _firrtlResult = fresult
        firrtlExecutionResult match {
          case FirrtlExecutionSuccess(_, vresult) =>
            _verilogResult = vresult
          case FirrtlExecutionFailure(msg) => fail(msg)
        }
      case ChiselExecutionSucccess(_, _, None) => fail("Firrtl did not run for some reason")
      case ChiselExecutionFailure(msg) => fail(msg)
    }
  }
}

