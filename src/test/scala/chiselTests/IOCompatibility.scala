// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.core.Binding.BindingException
import org.scalatest._

class IOCSimpleIO extends Bundle {
  val in  = Input(UInt(width=32))
  val out = Output(UInt(width=32))
}

class IOCPlusOne extends Module {
  val io = IO(new IOCSimpleIO)
  io.out := io.in + UInt(1)
}

class IOCModuleVec(val n: Int) extends Module {
  val io = IO(new Bundle {
    val ins  = Vec(n, Input(UInt(width=32)))
    val outs = Vec(n, Output(UInt(width=32)))
  })
  val pluses = Vec.fill(n){ Module(new IOCPlusOne).io }
  for (i <- 0 until n) {
    pluses(i).in := io.ins(i)
    io.outs(i)   := pluses(i).out
  }
}

class IOCModuleWire extends Module {
  val io = IO(new IOCSimpleIO)
  val inc = Wire(Module(new IOCPlusOne).io.chiselCloneType)
  inc.in := io.in
  io.out := inc.out
}

class IOCompatibilitySpec extends ChiselPropSpec with Matchers {

  property("IOCModuleVec should elaborate") {
    elaborate { new IOCModuleVec(2) }
  }

  property("IOCModuleWire should elaborate") {
    elaborate { new IOCModuleWire }
  }


  class IOUnwrapped extends Module {
    val io = new IOCSimpleIO
    io.out := io.in
  }

  property("Unwrapped IO should generate an exception") {
    a [BindingException] should be thrownBy {
      elaborate(new IOUnwrapped)
    }
  }
}
