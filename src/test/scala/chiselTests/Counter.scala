// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.util.Counter
import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers

class CountTester(max: Int) extends Module {
  val cnt = Counter(max)
  assert(cnt.n == max)
  when(true.B) { cnt.inc() }
  val expected = if (max == 0) 0.U else (max - 1).U
  when(cnt.value === expected) {
    stop()
  }
}

class EnableTester(seed: Int) extends Module {
  val ens = RegInit(seed.asUInt)
  ens := ens >> 1

  val (cntEnVal, _) = Counter(ens(0), 32)
  val (_, done) = Counter(true.B, 33)

  private def popCount(n: Long): Int = n.toBinaryString.count(_ == '1')

  when(done) {
    assert(cntEnVal === popCount(seed).asUInt)
    stop()
  }
}

class ResetTester(n: Int) extends Module {
  val triggerReset = WireInit(false.B)
  val wasReset = RegNext(triggerReset)
  val (value, _) = Counter(0 until 8, reset = triggerReset)

  triggerReset := value === (n - 1).U

  when(wasReset) {
    assert(value === 0.U)
    stop()
  }
}

class WrapTester(max: Int) extends Module {
  val (cnt, wrap) = Counter(true.B, max)
  when(wrap) {
    assert(cnt === (max - 1).asUInt)
    stop()
  }
}

class RangeTester(r: Range) extends Module {
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

class CounterSpec extends AnyPropSpec with PropertyUtils with ChiselSim {
  property("Counter should count up") {
    for (i <- 0 until 4) {
      simulate(new CountTester(i))(RunUntilFinished(i + 2))
    }
  }

  property("Counter can be en/disabled") {
    forAll(safeUInts) { (seed: Int) =>
      whenever(seed >= 0) { simulate { new EnableTester(seed) }(RunUntilFinished(34)) }
    }
  }

  property("Counter can be reset") {
    forAll(smallPosInts) { (seed: Int) => simulate { new ResetTester(seed) }(RunUntilFinished(1024 * 10)) }
  }

  property("Counter should wrap") {
    forAll(smallPosInts) { (max: Int) => simulate { new WrapTester(max) }(RunUntilFinished(max + 1)) }
  }

  property("Counter should handle a range") {
    forAll(posRange) { (r: Range) =>
      simulate { new RangeTester(r) }(RunUntilFinished(1024 * 10))
    }
  }
}
