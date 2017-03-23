// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental.{ChiselRange, Interval}
import chisel3.internal.firrtl.KnownSIntRange
import chisel3.testers.BasicTester
import cookbook.CookbookTester
import org.scalatest.{FreeSpec, Matchers}

class IntervalTest1 extends Module {
  val io = IO(new Bundle {
    val in1 = Interval(6.W, 3.BP, range"[0,4]")
    val in2 = Interval(6.W, 3.BP, range"[0,4]")
    val out = Interval(8.W, 3.BP, range"[0,8]")
  })
}
class IntervalTester extends CookbookTester(10) {
  val dut = Module(new IntervalTest1)

  dut.io.in1 := 4.I()
  dut.io.in2 := 4.I()
  assert(dut.io.out === 8.I())

  val i = Interval(range"[0,10)")
  stop()
}

class IntervalSpec extends FreeSpec with Matchers with ChiselRunners {
  "Intervals can be created" in {
    assertTesterPasses{ new IntervalTester }
  }
}
