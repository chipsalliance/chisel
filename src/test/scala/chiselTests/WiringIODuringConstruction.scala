package chiselTests

import Chisel._
import Chisel.testers.BasicTester

class SelfWiringIO extends Bundle {
  val data  = UInt(INPUT, width = 32)
  val ready = Bool(OUTPUT)
  val valid = Bool(INPUT)

  ready := Bool(true)
}

class SelfWiringPacket extends Bundle {
  val addr = UInt(width = 32)
  val data = UInt(width = 32)
}

class SelfWiring extends Module {
  val io  = new Bundle {
    val in = new SelfWiringIO
    val outs                         = Vec(4, new EnqIO(new SelfWiringPacket))
  }
  val buffer = Reg(init = UInt(0, width = 32))

  when(io.in.valid && io.in.ready) {
    buffer   := io.in.data
    io.in.ready := Bool(false)
  }
}

class SelfWiringTester extends BasicTester {
  val c = Module( new SelfWiring )

  when(c.io.in.ready) {
    c.io.in.data := UInt(42)
  }
  stop()
}

class WiringIODuringConstruction extends ChiselPropSpec {
  property("devices that wire things during IO construction should run") {
    assert(execute { new SelfWiringTester })
  }
}
