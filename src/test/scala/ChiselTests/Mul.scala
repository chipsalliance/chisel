package ChiselTests
import Chisel._
import Chisel.testers._
import scala.collection.mutable.ArrayBuffer

class Mul(val w: Int) extends Module {
  val io = new Bundle {
    val x   = UInt(INPUT,  w)
    val y   = UInt(INPUT,  w)
    val z   = UInt(OUTPUT, 2 * w)
  }
  val muls = new ArrayBuffer[UInt]()

  val n = 1 << w

  for (i <- 0 until n)
    for (j <- 0 until n)
      muls += UInt(i * j, 2 * w)
  val tbl = Vec(muls)
  // val ad = (io.x << w) | io.y
  io.z := tbl(((io.x << w) | io.y).toUInt)
}


class MulTester(c: Mul) extends Tester(c) {
  val maxInt  = 1 << c.w
  for (i <- 0 until 10) {
    val x = rnd.nextInt(maxInt)
    val y = rnd.nextInt(maxInt)
    poke(c.io.x, x)
    poke(c.io.y, y)
    step(1)
    expect(c.io.z, (x * y))
  }
}
