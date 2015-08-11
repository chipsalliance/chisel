package ChiselTests
import Chisel._
import Chisel.testers._

object Insts {
  def ADD = BitPat("b0000000??????????000?????0110011")
}

class Decoder extends Module {
  val io = new Bundle {
    val inst  = UInt(INPUT, 32)
    val isAdd = Bool(OUTPUT)
  }
  io.isAdd := (Insts.ADD === io.inst)
}

class DecoderTester(c: Decoder) extends Tester(c) {
  poke(c.io.inst, 0x1348533)
  step(1)
  expect(c.io.isAdd, int(true))
}
