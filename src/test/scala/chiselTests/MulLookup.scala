// See LICENSE for license details.

package chiselTests

import chisel3._
import org.scalatest._
import org.scalatest.prop._
import chisel3.testers.BasicTester

class MulLookup(val w: Int) extends Module {
  val io = new Bundle {
    val x   = UInt(INPUT,  w)
    val y   = UInt(INPUT,  w)
    val z   = UInt(OUTPUT, 2 * w)
  }
  val tbl = Vec(
    for {
      i <- 0 until 1 << w
      j <- 0 until 1 << w
    } yield UInt(i * j, 2 * w)
  )
  io.z := tbl(((io.x << w) | io.y))
}

class MulLookupTester(w: Int, x: Int, y: Int) extends BasicTester {
  val dut = Module(new MulLookup(w))
  dut.io.x := UInt(x)
  dut.io.y := UInt(y)
  assert(dut.io.z === UInt(x * y))
  stop()
}

class MulLookupSpec extends ChiselPropSpec {

  property("Mul lookup table should return the correct result") {
    forAll(smallPosInts, smallPosInts) { (x: Int, y: Int) =>
      assertTesterPasses{ new MulLookupTester(3, x, y) }
    }
  }
}
