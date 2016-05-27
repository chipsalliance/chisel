// See LICENSE for license details.

package chiselTests

import org.scalatest._
import org.scalatest.prop._

import chisel._
import chisel.testers.BasicTester
import chisel.util._

class CountTester(max: Int) extends BasicTester {
  val cnt = Counter(max)
  when(Bool(true)) { cnt.inc() }
  when(cnt.value === UInt(max-1)) {
    stop()
  }
}

class EnableTester(seed: Int) extends BasicTester {
  val ens = Reg(init = UInt(seed))
  ens := ens >> 1

  val (cntEnVal, _) = Counter(ens(0), 32)
  val (_, done) = Counter(Bool(true), 33)

  when(done) {
    assert(cntEnVal === UInt(popCount(seed)))
    stop()
  }
}

class WrapTester(max: Int) extends BasicTester {
  val (cnt, wrap) = Counter(Bool(true), max)
  when(wrap) {
    assert(cnt === UInt(max - 1))
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
