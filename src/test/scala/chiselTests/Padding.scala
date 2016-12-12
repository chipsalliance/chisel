// See LICENSE for license details.

package chiselTests

import chisel3._

class Padder extends Module {
  val io = IO(new Bundle {
    val a   = Input(UInt(4.W))
    val asp = Output(SInt(8.W))
    val aup = Output(UInt(8.W))
  })
  io.asp := io.a.asSInt
  io.aup := io.a.asUInt
}

/*
class PadsTester(c: Pads) extends Tester(c) {
  def pads(x: BigInt, s: Int, w: Int) = {
    val sign  = (x & (1 << (s-1)))
    val wmask = (1 << w) - 1
    val bmask = (1 << s) - 1
    if (sign == 0) x else ((~bmask | x) & wmask)
  }
  for (t <- 0 until 16) {
    val test_a = rnd.nextInt(1 << 4)
    poke(c.io.a, test_a)
    step(1)
    expect(c.io.asp, pads(test_a, 4, 8))
    expect(c.io.aup, test_a)
  }
}
*/

class PadderSpec extends ChiselPropSpec {

  property("Padder should elaborate") {
    elaborate { new Padder }
  }

  ignore("PadderTester should return the correct result") { }
}
