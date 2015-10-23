// See LICENSE for license details.

package chiselTests

import Chisel._
import org.scalatest._
import org.scalatest.prop._
import Chisel.testers.BasicTester

class BitwiseOpsSpec extends ChiselPropSpec {

  class BitwiseOpsTester(w: Int, _a: Int, _b: Int) extends BasicTester {
    io.done := Bool(true)
    val mask = (1 << w) - 1
    val a = UInt(_a)
    val b = UInt(_b)
    when(~a != UInt(mask & ~_a)) { io.error := UInt(1) }
    when((a & b) != UInt(mask & (_a & _b))) { io.error := UInt(2) }
    when((a | b)  != UInt(mask & (_a | _b))) { io.error := UInt(3) }
    when((a ^ b) != UInt(mask & (_a ^ _b))) { io.error := UInt(4) }
  }

  property("All bit-wise ops should return the correct result") {
    forAll(safeUIntPair) { case(w: Int, a: Int, b: Int) =>
      assert(execute{ new BitwiseOpsTester(w, a, b) })
    }
  }
}
