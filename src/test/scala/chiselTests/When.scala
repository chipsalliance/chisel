// See LICENSE for license details.

package chiselTests

import org.scalatest._
import Chisel._
import Chisel.testers.BasicTester

class WhenTester() extends BasicTester {
  val cnt = Counter(4)
  when(true.asBool) { cnt.inc() }

  val out = Wire(UInt(width=3))
  when(cnt.value === 0.asUInt) {
    out := 1.asUInt
  } .elsewhen (cnt.value === 1.asUInt) {
    out := 2.asUInt
  } .elsewhen (cnt.value === 2.asUInt) {
    out := 3.asUInt
  } .otherwise {
    out := 0.asUInt
  }

  assert(out === cnt.value + 1.asUInt)

  when(cnt.value === 3.asUInt) {
    stop()
  }
}

class OverlappedWhenTester() extends BasicTester {
  val cnt = Counter(4)
  when(true.asBool) { cnt.inc() }

  val out = Wire(UInt(width=3))
  when(cnt.value <= 0.asUInt) {
    out := 1.asUInt
  } .elsewhen (cnt.value <= 1.asUInt) {
    out := 2.asUInt
  } .elsewhen (cnt.value <= 2.asUInt) {
    out := 3.asUInt
  } .otherwise {
    out := 0.asUInt
  }

  assert(out === cnt.value + 1.asUInt)

  when(cnt.value === 3.asUInt) {
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
