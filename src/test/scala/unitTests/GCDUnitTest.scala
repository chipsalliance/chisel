// See LICENSE for license details.

package unitTests

import Chisel._
import Chisel.testers.{UnitTestRunners, UnitTester}

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
      if(y > 0) depth += 1
    }
    return (x, depth)
  }

  val gcd = Module(new GCD)

  for {
    value_1 <- 4 to 8
    value_2 <- 2 to 4
  } {
    poke(gcd.io.a, value_1)
    poke(gcd.io.b, value_2)
    poke(gcd.io.e, 1)
    step(1)
    poke(gcd.io.e, 0)

    val (expected_gcd, steps) = compute_gcd(value_1, value_2)

    step(steps-1) // -1 is because we step(1) already to toggle the enable
    expect(gcd.io.z, expected_gcd)
    expect(gcd.io.v, 1 )
  }

  install(gcd)

}

class GCDTester extends UnitTestRunners {
  execute { new GCDUnitTester }
}

object GCDUnitTest {
  def main(args: Array[String]): Unit = {
    val tutorial_args = args.slice(1, args.length)

    new GCDTester
  }
}

