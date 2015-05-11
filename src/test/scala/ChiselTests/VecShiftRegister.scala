package ChiselTests
import Chisel._

class VecShiftRegister extends Module {
  val io = new Bundle {
    val ins   = Vec(UInt(INPUT, 4), 4)
    val load  = Bool(INPUT)
    val shift = Bool(INPUT)
    val out   = UInt(OUTPUT, 4)
  }
  val delays = Reg(Vec(UInt(width = 4), 4))
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


class VecShiftRegisterTester(c: VecShiftRegister) extends Tester(c) {
}
