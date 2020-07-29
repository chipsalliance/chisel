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

class ResetTester(n: Int) extends BasicTester {
  val triggerReset = WireInit(false.B)
  val wasReset = RegNext(triggerReset)
  val (value, _) = Counter(0 until 8, reset = triggerReset)

  triggerReset := value === (n-1).U

  when(wasReset) {
    assert(value === 0.U)
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
  val checkWrap = RegInit(false.B)

  when(checkWrap) {
    assert(cnt === r.head.U)
    stop()
  }.elsewhen(wrap) {
    assert(cnt === r.last.U)
    checkWrap := true.B
  }
}

class CounterSpec extends ChiselPropSpec {
  property("Counter should count up") {
    forAll(smallPosInts) { (max: Int) => assertTesterPasses{ new CountTester(max) } }
  }

  property("Counter can be en/disabled") {
    forAll(safeUInts) { (seed: Int) => whenever(seed >= 0) { assertTesterPasses{ new EnableTester(seed) } } }
  }

  property("Counter can be reset") {
    forAll(smallPosInts) { (seed: Int) => assertTesterPasses{ new ResetTester(seed) } }
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
