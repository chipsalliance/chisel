// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._

class CountTester(max: Int) extends BasicTester {
  val cnt = Counter(max)
  when(true.B) { cnt.inc() }
  when(cnt.value === (max-1).asUInt) {
    stop()
  }
}

class EnableTester(seed: Int) extends BasicTester {
  val ens = RegInit(seed.asUInt)
  ens := ens >> 1

  val (cntEnVal, _) = Counter(ens(0), 32)
  val (_, done) = Counter(true.B, 33)

  when(done) {
    assert(cntEnVal === popCount(seed).asUInt)
    stop()
  }
}

class WrapTester(max: Int) extends BasicTester {
  val (cnt, wrap) = Counter(true.B, max)
  when(wrap) {
    assert(cnt === (max - 1).asUInt)
    stop()
  }
}

class RangeTester(r: Range) extends BasicTester {
  val (cnt, wrap) = Counter(r)
  when(wrap) {
    assert(cnt === r.last.U)
    stop()
  }
}

class CounterSpec extends ChiselPropSpec {
  property("Counter should count up") {
    forAll(smallPosInts) { (max: Int) => assertTesterPasses{ new CountTester(max) } }
  }

  property("Counter can be en/disabled") {
    forAll(safeUInts) { (seed: Int) => whenever(seed >= 0) { assertTesterPasses{ new EnableTester(seed) } } }
  }

  property("Counter should wrap") {
    forAll(smallPosInts) { (max: Int) => assertTesterPasses{ new WrapTester(max) } }
  }

  property("Counter should handle a range") {
    forAll(posRange) { (r: Range) =>
      assertTesterPasses{ new RangeTester(r) }
    }
  }
}
