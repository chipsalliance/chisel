// See LICENSE for license details.

package chiselTests

import org.scalatest._
import Chisel._
import Chisel.testers.BasicTester

class WhenTester() extends BasicTester {
  val cnt = Counter(4)
  when(Bool(true)) { cnt.inc() }

  val out = Wire(UInt(width=3))
  when(cnt.value === UInt(0)) {
    out := UInt(1)
  } .elsewhen (cnt.value === UInt(1)) {
    out := UInt(2)
  } .elsewhen (cnt.value === UInt(2)) {
    out := UInt(3)
  } .otherwise {
    out := UInt(0)
  }

  assert(out === cnt.value + UInt(1))

  when(cnt.value === UInt(3)) {
    stop()
  }
}

class OverlappedWhenTester() extends BasicTester {
  val cnt = Counter(4)
  when(Bool(true)) { cnt.inc() }

  val out = Wire(UInt(width=3))
  when(cnt.value <= UInt(0)) {
    out := UInt(1)
  } .elsewhen (cnt.value <= UInt(1)) {
    out := UInt(2)
  } .elsewhen (cnt.value <= UInt(2)) {
    out := UInt(3)
  } .otherwise {
    out := UInt(0)
  }

  assert(out === cnt.value + UInt(1))

  when(cnt.value === UInt(3)) {
    stop()
  }
}

class WhenSpec extends ChiselFlatSpec {
  "When, elsewhen, and otherwise with orthogonal conditions" should "work" in {
    assertTesterPasses{ new WhenTester }
  }
  "When, elsewhen, and otherwise with overlapped conditions" should "work" in {
    assertTesterPasses{ new OverlappedWhenTester }
  }
}
