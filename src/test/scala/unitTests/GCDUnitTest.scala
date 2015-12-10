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
  def compute_gcd(a: Int, b: Int): Int = if(b == 0) a else compute_gcd(b, a%b)

  val gcd = Module(new GCD)

  poke(gcd.io.a, UInt(17))

  for {
    value_1 <- 0 to 2
    value_2 <- 0 to 2
  } {
    poke(gcd.io.a, UInt(value_1))
    poke(gcd.io.b, UInt(value_2))

    expect(gcd.io.v, UInt(compute_gcd(value_1, value_2)))
    step(1)
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

