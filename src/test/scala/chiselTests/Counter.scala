package chiselTests
import Chisel._
import org.scalatest._
import org.scalatest.prop._
import Chisel.testers.BasicTester

class CounterSpec extends ChiselPropSpec {

  class CountTester(max: Int) extends BasicTester {
    val cnt = Counter(max)
    when(cnt.value === UInt(max)) { io.done := Bool(true) }
  }

  property("Counter should count up") {
    forAll(smallPosInts) { (max: Int) => assert(execute{ new CountTester(max) }) }
  }

  class EnableTester(seed: Int) extends BasicTester {
    val ens = Reg(init = UInt(seed))
    ens := ens >> 1
    val (cntEn, cntWrap) = Counter(ens(0), 32)
    val cnt = Counter(32)
    when(cnt.value === UInt(31)) {
      io.done := Bool(true) 
      io.error := cnt.value != UInt(popCount(seed))
    }
  }

  property("Counter can be en/disabled") {
    forAll(safeUInts) { (seed: Int) => assert(execute{ new EnableTester(seed) }) }
  }

  class WrapTester(max: Int) extends BasicTester {
    val (cnt, wrap) = Counter(Bool(true), max)
    when(wrap) { io.done := Bool(true); io.error := cnt != UInt(max) }
  }

  property("Counter should wrap") {
    forAll(smallPosInts) { (max: Int) => assert(execute{ new WrapTester(max) }) }
  }
}
