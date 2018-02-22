// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.testers.BasicTester

class MemorySearch extends Module {
  val io = IO(new Bundle {
    val target  = Input(UInt(4.W))
    val en      = Input(Bool())
    val done    = Output(Bool())
    val address = Output(UInt(3.W))
  })
  val vals  = Array(0, 4, 15, 14, 2, 5, 13)
  val index = RegInit(0.U(3.W))
  val elts  = VecInit(vals.map(_.asUInt(4.W)))
  // val elts  = Mem(UInt(32.W), 8) TODO ????
  val elt  = elts(index)
  val end  = !io.en && ((elt === io.target) || (index === 7.U))
  when (io.en) {
    index := 0.U
  } .elsewhen (!end) {
    index := index +% 1.U
  }
  io.done    := end
  io.address := index
}

/*
class MemorySearchTester(c: MemorySearch) extends Tester(c) {
  val list = c.vals
  val n = 8
  val maxT = n * (list.length + 3)
  for (k <- 0 until n) {
    val target = rnd.nextInt(16)
    poke(c.io.en,     1)
    poke(c.io.target, target)
    step(1)
    poke(c.io.en,     0)
    do {
      step(1)
    } while (peek(c.io.done) == 0 && t < maxT)
    val addr = peek(c.io.address).toInt
    expect(addr == list.length || list(addr) == target,
           "LOOKING FOR " + target + " FOUND " + addr)
  }
}
*/

class MemorySearchSpec extends ChiselPropSpec {

  property("MemorySearch should elaborate") {
    elaborate { new EnableShiftRegister }
  }

  ignore("MemorySearch should return the correct result") { }
}
