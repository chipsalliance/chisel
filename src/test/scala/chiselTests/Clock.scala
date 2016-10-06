// See LICENSE for license details.

package chiselTests

import org.scalatest._
import org.scalatest.prop._

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._

class ClockAsUIntTester extends BasicTester {
  assert(Bool(true).asClock.asUInt === UInt(1))
  stop()
}


class ClockSpec extends ChiselPropSpec {
  property("Bool.asClock.asUInt should pass a signal through unaltered") {
    assertTesterPasses { new ClockAsUIntTester }
  }
}
