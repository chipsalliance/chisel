// See LICENSE for license details.

package chiselTests
import chisel3._
import chisel3.testers.BasicTester

class EnableShiftRegister extends Module {
  val io = IO(new Bundle {
    val in    = Input(UInt(4.W))
    val shift = Input(Bool())
    val out   = Output(UInt(4.W))
  })
  val r0 = RegInit(0.U(4.W))
  val r1 = RegInit(0.U(4.W))
  val r2 = RegInit(0.U(4.W))
  val r3 = RegInit(0.U(4.W))
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
