package unitTests

import Chisel._
import Chisel.testers.UnitTester
import scala.util.Random

class Adder(val w: Int) extends Module {
  val io = new Bundle {
    val in0 = UInt(INPUT,  w)
    val in1 = UInt(INPUT,  w)
    val out = UInt(OUTPUT, w)
  }
  io.out := io.in0 + io.in1
}

class AdderTests extends UnitTester {
  val c = Module( new Adder(10) )

  for (i <- 0 until 10) {
    val in0 = Random.nextInt(1 << c.w)
    val in1 = Random.nextInt(1 << c.w)
    poke(c.io.in0, in0)
    poke(c.io.in1, in1)
    step(1)
    expect(c.io.out, (in0 + in1)&((1 << c.w)-1))
  }

  install(c)
}
