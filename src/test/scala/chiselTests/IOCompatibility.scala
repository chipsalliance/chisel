// See LICENSE for license details.

package chiselTests

import Chisel._
import chisel3.NotStrict.CompileOptions

class IOCSimpleIO extends Bundle {
  val in  = UInt(INPUT,  32)
  val out = UInt(OUTPUT, 32)
}

class IOCPlusOne extends Module {
  val io = new IOCSimpleIO
  io.out := io.in + UInt(1)
}

class IOCModuleVec(val n: Int) extends Module {
  val io = new Bundle {
    val ins  = Vec(n, UInt(INPUT,  32))
    val outs = Vec(n, UInt(OUTPUT, 32))
  }
  val pluses = Vec.fill(n){ Module(new IOCPlusOne).io }
  for (i <- 0 until n) {
    pluses(i).in := io.ins(i)
    io.outs(i)   := pluses(i).out
  }
}

class IOCModuleWire extends Module {
  val io = new IOCSimpleIO
  val inc = Wire(Module(new IOCPlusOne).io.chiselCloneType)
  inc.in := io.in
  io.out := inc.out
}

class IOCompatibilitySpec extends ChiselPropSpec {

  property("IOCModuleVec should elaborate") {
    elaborate { new IOCModuleVec(2) }
  }

  property("IOCModuleWire should elaborate") {
    elaborate { new IOCModuleWire }
  }
}
