package chiselTests

import Chisel._
import Chisel.testers.BasicTester

class BadNameReg extends Module {
  val io  = new Bundle{ val x = UInt(INPUT, width = 32)}
  val reg = Reg(init = UInt(0, width = 32))

  when(io.x > UInt(0)) {
    reg   := UInt(3)
  }
}

class BadNameTester extends BasicTester {
  val inst = Module( new BadNameReg )

  inst.io.x := UInt(42)
  stop()
}

class RegIsABadNameForReg extends ChiselPropSpec {
  property("devices that wire things during IO construction should run") {
    assert(execute { new BadNameTester })
  }
}
