// See LICENSE for license details.

package chiselTests
import chisel3._
import org.scalatest._
import org.scalatest.prop._
import chisel3.testers.BasicTester
//import chisel3.core.ExplicitCompileOptions.Strict

class Coord extends Bundle {
  val x = UInt.width( 32)
  val y = UInt.width( 32)
}

class BundleWire(n: Int) extends Module {
  val io = IO(new Bundle {
    val in   = Input(new Coord)
    val outs = Output(Vec(n, new Coord))
  })
  val coords = Wire(Vec(n, new Coord))
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
      assertTesterPasses{ new BundleWireTester(n, x, y) }
    }
  }
}
