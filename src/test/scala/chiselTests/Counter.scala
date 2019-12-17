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
    when(! cntEnVal === popCount(seed).asUInt) {
      printf(s"XXXXXXXX $seed  cntEnVal %d popCount(seed) is ${popCount(seed)}\n", cntEnVal)
    }
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

class CounterSpec extends ChiselPropSpec {
  property("Counter should count up") {
    forAll(smallPosInts) { (max: Int) => assertTesterPasses{ new CountTester(max) } }
  }

  property("Counter can be en/disabled") {
    assertTesterPasses{ new EnableTester(4) }
//    forAll(safeUInts) { (seed: Int) => whenever(seed >= 0) {
//      println(f"Testing with seed $seed%d $seed%x")
//      assertTesterPasses{ new EnableTester(seed) }
//    }}
  }

  property("Counter should wrap") {
    forAll(smallPosInts) { (max: Int) => assertTesterPasses{ new WrapTester(max) } }
  }
}
