// See LICENSE for license details.

package chiselTests

import Chisel._
import org.scalatest._
import org.scalatest.prop._
import Chisel.testers.BasicTester

class BitwiseOpsTester(w: Int, _a: Int, _b: Int) extends BasicTester {
  val mask = (1 << w) - 1
  val a = _a.asUInt(w)
  val b = _b.asUInt(w)
  assert(~a === (mask & ~_a).asUInt)
  assert((a & b) === (_a & _b).asUInt)
  assert((a | b) === (_a | _b).asUInt)
  assert((a ^ b) === (_a ^ _b).asUInt)
  stop()
}

class BitwiseOpsSpec extends ChiselPropSpec {
  property("All bit-wise ops should return the correct result") {
    forAll(safeUIntPair) { case(w: Int, a: Int, b: Int) =>
      assertTesterPasses{ new BitwiseOpsTester(w, a, b) }
    }
  }
}
