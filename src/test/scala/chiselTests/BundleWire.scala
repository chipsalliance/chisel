// See LICENSE for license details.

package chiselTests
import Chisel._
import org.scalatest._
import org.scalatest.prop._
import Chisel.testers.BasicTester

class Coord extends Bundle {
  val x = UInt(width = 32)
  val y = UInt(width = 32)
}

class BundleWire(n: Int) extends Module {
  val io = new Bundle {
    val in   = (new Coord).asInput
    val outs = Vec(new Coord, n).asOutput
  }
  val coords = Wire(Vec(new Coord, n))
  for (i <- 0 until n) {
    coords(i)  := io.in
    io.outs(i) := coords(i)
  }
}

class BundleWireTester(n: Int, x: Int, y: Int) extends BasicTester {
  val dut = Module(new BundleWire(n))
  dut.io.in.x := UInt(x)
  dut.io.in.y := UInt(y)
  for (elt <- dut.io.outs) {
    assert(elt.x === UInt(x))
    assert(elt.y === UInt(y))
  }
  stop()
}

class BundleWireSpec extends ChiselPropSpec {

  property("All vec elems should match the inputs") {
    forAll(vecSizes, safeUInts, safeUInts) { (n: Int, x: Int, y: Int) =>
      assert(execute{ new BundleWireTester(n, x, y) })
    }
  }
}
