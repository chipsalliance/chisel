// See LICENSE for license details.

package chiselTests

import org.scalatest._
import org.scalatest.prop._

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._

import scala.math.{min, max}

class CountTester(max: Int) extends BasicTester {
  val cnt = Counter(max)
  when(true.B) { cnt.inc() }
  when(cnt.value === (max-1).asUInt) {
    stop()
  }
}

class EnableTester(seed: Int) extends BasicTester {
  val ens = Reg(init = seed.asUInt)
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

class InitTester(max: Int, init: Int) extends BasicTester {
  val cntFromInit = Counter(max, init)
  cntFromInit.inc()
  
  val cntFromZero = Counter(max)
  val wrap = cntFromZero.inc()

  val restart = if (init > 0) cntFromZero.value === (init - 1).U else false.B
  when (restart) { cntFromInit.restart() }

  assert(! (cntFromZero.value >  init.U) || (cntFromInit.value === cntFromZero.value) )

  when (wrap) { stop() }
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

  property("Counter should init and restart") {
    forAll(smallPosInts, smallPosInts) { (x: Int, y: Int) => if (x > y) assertTesterPasses{
      new InitTester(x, y) 
    } }
  }
}
