// See LICENSE for license details.

package unitTests

import Chisel._
import Chisel.testers.{BasicTester, UnitTester}
import chiselTests.ChiselFlatSpec

class GCD extends Module {
  val io = new Bundle {
    val a  = UInt(INPUT,  16)
    val b  = UInt(INPUT,  16)
    val e  = Bool(INPUT)
    val z  = UInt(OUTPUT, 16)
    val v  = Bool(OUTPUT)
  }
  val x  = Reg(UInt())
  val y  = Reg(UInt())
  when   (x > y) { x := x - y }
  unless (x > y) { y := y - x }
  when (io.e) { x := io.a; y := io.b }
  io.z := x
  io.v := y === UInt(0)
}

class GCDUnitTester extends UnitTester {
  def compute_gcd(a: Int, b: Int): Tuple2[Int, Int] = {
    var x = a
    var y = b
    var depth = 1
    while(y > 0 ) {
      if (x > y) {
        x -= y
      }
      else {
        y -= x
      }
      depth += 1
    }
    return (x, depth)
  }

  val (a, b, z) = (64, 48, 16)
  val device_under_test = Module(new GCD)
  val gcd = device_under_test

  testBlock {
    poke(gcd.io.a, a)
    poke(gcd.io.b, b)
    poke(gcd.io.e, 1)
    step(1)
    poke(gcd.io.e, 0)

    val (expected_gcd, steps) = compute_gcd(a, b)

    step(steps - 1) // -1 is because we step(1) already to toggle the enable
    expect(gcd.io.z, expected_gcd)
    expect(gcd.io.v, 1)
  }
}

class GCDTester extends ChiselFlatSpec {
  "a" should "b" in {
    assert( execute { new GCDUnitTester } )
  }
}

//object GCDUnitTest {
//  def main(args: Array[String]): Unit = {
//    val tutorial_args = args.slice(1, args.length)
//
//    new GCDTester
//  }
//}
//
