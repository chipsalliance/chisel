// See LICENSE for license details.

package chiselTests

import Chisel._
import Chisel.testers.BasicTester
import org.scalatest._
import org.scalatest.prop._

class GCD extends Module {
  val io = new Bundle {
    val a  = UInt(INPUT, 32)
    val b  = UInt(INPUT, 32)
    val e  = Bool(INPUT)
    val z  = UInt(OUTPUT, 32)
    val v  = Bool(OUTPUT)
  }
  val x = Reg(UInt(width = 32))
  val y = Reg(UInt(width = 32))
  when (x > y)   { x := x -% y }
  .otherwise     { y := y -% x }
  when (io.e) { x := io.a; y := io.b }
  io.z := x
  io.v := y === UInt(0)
}

class GCDTester(a: Int, b: Int, z: Int) extends BasicTester {
  val dut = Module(new GCD)
  val first = Reg(init=Bool(true))
  dut.io.a := UInt(a)
  dut.io.b := UInt(b)
  dut.io.e := first
  when(first) { first := Bool(false) }
  when(dut.io.v) {
    io.done := Bool(true)
    io.error := (dut.io.z != UInt(z)).toUInt
  }
}

class GCDSpec extends ChiselPropSpec {
  
  //TODO: use generators and this function to make z's
  def gcd(a: Int, b: Int): Int = if(b == 0) a else gcd(b, a%b)

  val gcds = Table(
    ("a", "b", "z"),  // First tuple defines column names
    ( 64,  48,  16),  // Subsequent tuples define the data
    ( 12,   9,   3),
    ( 48,  64,  12))

  property("GCD should return the correct result") {
    forAll (gcds) { (a: Int, b: Int, z: Int) =>
      assert(execute{ new GCDTester(a, b, z) })
    }
  }
}
