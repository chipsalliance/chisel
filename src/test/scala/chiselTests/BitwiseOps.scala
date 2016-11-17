// See LICENSE for license details.

package chiselTests

import chisel3._
import org.scalatest._
import org.scalatest.prop._
import chisel3.testers.BasicTester

class BitwiseOpsTester(w: Int, _a: Int, _b: Int) extends BasicTester {
  val mask = (1 << w) - 1
  val a = _a.asUInt(w.W)
  val b = _b.asUInt(w.W)
  assert(~a === UInt(mask & ~_a))
  assert((a & b) === UInt(_a & _b))
  assert((a | b) === UInt(_a | _b))
  assert((a ^ b) === UInt(_a ^ _b))
  stop()
}

class BitwiseOpsSpec extends ChiselPropSpec {
  property("All bit-wise ops should return the correct result") {
    forAll(safeUIntPair) { case(w: Int, a: Int, b: Int) =>
      assertTesterPasses{ new BitwiseOpsTester(w, a, b) }
    }
  }
}
