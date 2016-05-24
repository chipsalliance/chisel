// See LICENSE for license details.

package chiselTests
import Chisel._
import org.scalatest._
import org.scalatest.prop._
import Chisel.testers.BasicTester

class CountTester(max: Int) extends BasicTester {
  val cnt = Counter(max)
  when(true.asBool) { cnt.inc() }
  when(cnt.value === (max-1).asUInt) {
    stop()
  }
}

class EnableTester(seed: Int) extends BasicTester {
  val ens = Reg(init = seed.asUInt)
  ens := ens >> 1

  val (cntEnVal, _) = Counter(ens(0), 32)
  val (_, done) = Counter(true.asBool, 33)

  when(done) {
    assert(cntEnVal === popCount(seed).asUInt)
    stop()
  }
}

class WrapTester(max: Int) extends BasicTester {
  val (cnt, wrap) = Counter(true.asBool, max)
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
    forAll(safeUInts) { (seed: Int) => whenever(seed >= 0) { assertTesterPasses{ new EnableTester(seed) } } }
  }

  property("Counter should wrap") {
    forAll(smallPosInts) { (max: Int) => assertTesterPasses{ new WrapTester(max) } }
  }
}
