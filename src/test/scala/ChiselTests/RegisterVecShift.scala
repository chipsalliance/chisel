package ChiselTests
import Chisel._
import Chisel.testers._

class RegisterVecShift extends Module {
  val io = new Bundle {
    val ins   = Vec(UInt(INPUT, 4), 4)
    // val ins   = Vec.fill(4){ UInt(INPUT, 4) }
    val load  = Bool(INPUT)
    val shift = Bool(INPUT)
    val out   = UInt(OUTPUT, 4)
  }
  // val delays = Reg( init = Vec.fill(4){ UInt(0, 4) } )
  val delays = Reg( Vec(UInt(width = 4), 4) )
  when (reset) {
    delays := Vec.fill(4){ UInt(0, 4) }
  }
  when (io.load) {
    delays(0) := io.ins(0)
    delays(1) := io.ins(1)
    delays(2) := io.ins(2)
    delays(3) := io.ins(3)
  } .elsewhen(io.shift) {
    delays(0) := io.ins(0)
    delays(1) := delays(0)
    delays(2) := delays(1)
    delays(3) := delays(2)
  }
  io.out := delays(3)
}


class RegisterVecShiftTester(c: RegisterVecShift) extends Tester(c) {
  val reg = Array.fill(4){ 0 }
  val ins = Array.fill(4){ 0 }
  for (t <- 0 until 16) {
    for (i <- 0 until 4)
      ins(i) = rnd.nextInt(16)
    val shift = rnd.nextInt(2)
    val load  = rnd.nextInt(2)
    for (i <- 0 until 4)
      poke(c.io.ins(i), ins(i))
    poke(c.io.load,  load)
    poke(c.io.shift, shift)
    step(1)
    if (load == 1) {
      for (i <- 0 until 4) 
        reg(i) = ins(i)
    } else if (shift == 1) {
      for (i <- 3 to 1 by -1)
        reg(i) = reg(i-1)
      reg(0) = ins(0)
    }
    expect(c.io.out, reg(3))
  }
}
