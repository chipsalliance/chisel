// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental.{ChiselRange, Interval}
import chisel3.internal.firrtl.KnownIntervalRange
import chisel3.testers.BasicTester
import cookbook.CookbookTester
import logger.LogLevel
import org.scalatest.{FreeSpec, Matchers}

class IntervalTest1 extends Module {
  val io = IO(new Bundle {
    val in1 = Input(Interval(6.W, 3.BP, range"[0,4]"))
    val in2 = Input(Interval(6.W, 3.BP, range"[0,4]"))
    val out = Output(Interval(8.W, 3.BP, range"[0,8]"))
  })

  io.out := io.in1 + io.in2
}
class IntervalTester extends CookbookTester(10) {
  val dut = Module(new IntervalTest1)

  dut.io.in1 := 4.I()
  dut.io.in2 := 4.I()
  assert(dut.io.out === 8.I())

  val i = Interval(range"[0,10)")
  stop()
}

class SIntTest1 extends Module {
  val io = IO(new Bundle {
    val in1 = Input(SInt(6.W))
    val in2 = Input(SInt(6.W))
    val out = Output(SInt(8.W))
  })

  io.out := io.in1 + io.in2
}
class SIntTest1Tester extends CookbookTester(10) {
  val dut = Module(new SIntTest1)

  dut.io.in1 := 4.S
  dut.io.in2 := 4.S
  assert(dut.io.out === 8.S)

  val i = SInt(range"[0,10)")
  stop()
}

class IntervalAddTester extends BasicTester {
  logger.Logger.globalLevel = LogLevel.Info

  val in1 = Wire(Interval(range"[0,4]"))
  val in2 = Wire(Interval(range"[0,4]"))

  in1 := 2.I
  in2 := 2.I

  val result = in1 +& in2

  assert(result === 4.I)

  stop()

}

class IntervalSpec extends FreeSpec with Matchers with ChiselRunners {
  "Test a simple interval add" in {
    assertTesterPasses{ new IntervalAddTester }
  }
  "Intervals can be created" in {
    assertTesterPasses{ new IntervalTester }
  }
  "SInt for use comparing to Interval" in {
    assertTesterPasses{ new SIntTest1Tester }
  }
  "show the firrtl" in {
    Driver.execute(Array("--no-run-firrtl"), () => new IntervalTester) match {
      case result: ChiselExecutionSuccess =>
        println(result.emitted)
      case _ =>
        assert(false, "Failed to generate chirrtl")
    }
  }
}
