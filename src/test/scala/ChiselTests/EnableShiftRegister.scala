package ChiselTests
import Chisel._

class EnableShiftRegister extends Module {
  val io = new Bundle {
    val in    = UInt(INPUT, 4)
    val shift = Bool(INPUT)
    val out   = UInt(OUTPUT, 4)
  }
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

class EnableShiftRegisterTester(c: EnableShiftRegister) extends Tester(c) {
  val reg = Array.fill(4){ 0 }
  for (t <- 0 until 16) {
    val in    = rnd.nextInt(16)
    val shift = rnd.nextInt(2)
    println("SHIFT " + shift + " IN " + in)
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
