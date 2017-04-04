// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._

class LFSR16 extends Module {
  val io = IO(new Bundle {
    val inc = Input(Bool())
    val out = Output(UInt(16.W))
  })
  val res = RegInit(1.U(16.W))
  when (io.inc) {
    val nxt_res = Cat(res(0)^res(2)^res(3)^res(5), res(15,1))
    res := nxt_res
  }
  io.out := res
}


/*
class LFSR16Tester(c: LFSR16) extends Tester(c) {
  var res = 1
  for (t <- 0 until 16) {
    val inc = rnd.nextInt(2)
    poke(c.io.inc, inc)
    step(1)
    if (inc == 1) {
      val bit = ((res >> 0) ^ (res >> 2) ^ (res >> 3) ^ (res >> 5) ) & 1;
      res = (res >> 1) | (bit << 15);
    }
    expect(c.io.out, res)
  }
}
*/

//TODO: Use chisel3.util version instead?

class LFSRSpec extends ChiselPropSpec {

  property("LFSR16 should elaborate") {
    elaborate { new LFSR16 }
  }

  ignore("LFSR16 should return the correct result") { }
}
