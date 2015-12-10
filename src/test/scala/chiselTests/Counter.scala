// See LICENSE for license details.

package chiselTests
import Chisel._
import org.scalatest._
import org.scalatest.prop._
import Chisel.testers.BasicTester

class CountTester(max: Int) extends BasicTester {
  val cnt = Counter(max)
  when(Bool(true)) { cnt.inc() }
  when(cnt.value === UInt(max-1)) { io.done := Bool(true) }
}

class EnableTester(seed: Int) extends BasicTester {
  val ens = Reg(init = UInt(seed))
  ens := ens >> 1
  val (cntEn, cntWrap) = Counter(ens(0), 32)
  val cnt = Counter(Bool(true), 32)._1
  when(cnt === UInt(31)) {
    io.done := Bool(true)
    io.error := cnt != UInt(popCount(seed))
  }
}

class WrapTester(max: Int) extends BasicTester {
  val (cnt, wrap) = Counter(Bool(true), max)
  when(wrap) { io.done := Bool(true); io.error := cnt != UInt(max) }
}

class CounterSpec extends ChiselPropSpec {

  property("Counter should count up") {
    forAll(smallPosInts) { (max: Int) => assert(execute{ new CountTester(max) }) }
  }

  property("Counter can be en/disabled") {
    forAll(safeUInts) { (seed: Int) => assert(execute{ new EnableTester(seed) }) }
  }

  property("Counter should wrap") {
    forAll(smallPosInts) { (max: Int) => assert(execute{ new WrapTester(max) }) }
  }
}
