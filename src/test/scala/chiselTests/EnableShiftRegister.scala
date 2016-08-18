// See LICENSE for license details.

package chiselTests
import chisel3._
import chisel3.testers.BasicTester

class EnableShiftRegister extends Module {
  val io = IO(new Bundle {
    val in    = Input(UInt.width(4))
    val shift = Input(Bool())
    val out   = Output(UInt.width(4))
  })
  val r0 = Reg(init = UInt(0, 4))
  val r1 = Reg(init = UInt(0, 4))
  val r2 = Reg(init = UInt(0, 4))
  val r3 = Reg(init = UInt(0, 4))
  when(io.shift) {
    r0 := io.in
    r1 := r0
    r2 := r1
    r3 := r2
  }
  io.out := r3
}

/*
class EnableShiftRegisterTester(c: EnableShiftRegister) extends Tester(c) {
  val reg = Array.fill(4){ 0 }
  for (t <- 0 until 16) {
    val in    = rnd.nextInt(16)
    val shift = rnd.nextInt(2)
    println("SHIFT " + shift + " IN " + in)   // scalastyle:ignore regex
    poke(c.io.in,    in)
    poke(c.io.shift, shift)
    step(1)
    if (shift == 1) {
      for (i <- 3 to 1 by -1)
        reg(i) = reg(i-1)
      reg(0) = in
    }
    expect(c.io.out, reg(3))
  }
}
*/

class EnableShiftRegisterSpec extends ChiselPropSpec {

  property("EnableShiftRegister should elaborate") {
    elaborate { new EnableShiftRegister }
  }

  ignore("EnableShiftRegisterTester should return the correct result") { }
}
