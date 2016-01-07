// See LICENSE for license details.

package chiselTests

import Chisel._
import Chisel.testers.BasicTester

class ByteSelector extends Module {
  val io = new Bundle {
    val in     = UInt(INPUT, 32)
    val offset = UInt(INPUT, 2)
    val out    = UInt(OUTPUT, 8)
  }
  io.out := UInt(0, width=8)
  when (io.offset === UInt(0, width=2)) {
    io.out := io.in(7,0)
  } .elsewhen (io.offset === UInt(1)) {
    io.out := io.in(15,8)
  } .elsewhen (io.offset === UInt(2)) {
    io.out := io.in(23,16)
  } .otherwise {
    io.out := io.in(31,24)
  }
}

class ByteSelectorUnitTester extends BasicTester {
  val c = Module(new ByteSelector)

  c.io.in := UInt(0x12345678)
  c.io.offset := UInt(1)
  stop()
}

class BytesSelectorSpec extends ChiselPropSpec {
  property("All bit-wise ops should return the correct result") {
    assert(execute { new ByteSelectorUnitTester})
  }
}

