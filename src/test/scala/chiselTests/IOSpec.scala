// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage
import org.scalatest.matchers.should.Matchers

class IOCSimpleIO extends Bundle {
  val in = Input(UInt(32.W))
  val out = Output(UInt(32.W))
}

class IOCPlusOne extends Module {
  val io = IO(new IOCSimpleIO)
  io.out := io.in + 1.U
}

class IOCModuleVec(val n: Int) extends Module {
  val io = IO(new Bundle {
    val ins = Vec(n, Input(UInt(32.W)))
    val outs = Vec(n, Output(UInt(32.W)))
  })
  val pluses = VecInit(Seq.fill(n) { Module(new IOCPlusOne).io })
  for (i <- 0 until n) {
    pluses(i).in := io.ins(i)
    io.outs(i) := pluses(i).out
  }
}

class IOCModuleWire extends Module {
  val io = IO(new IOCSimpleIO)
  val inc = Wire(chiselTypeOf(Module(new IOCPlusOne).io))
  inc.in := io.in
  io.out := inc.out
}

class IOSpec extends ChiselPropSpec with Matchers with Utils with MatchesAndOmits {

  property("IOCModuleVec should elaborate") {
    ChiselStage.emitCHIRRTL { new IOCModuleVec(2) }
  }

  property("IOCModuleWire should elaborate") {
    ChiselStage.emitCHIRRTL { new IOCModuleWire }
  }

  class IOUnwrapped extends Module {
    val io = new IOCSimpleIO
    io.out := io.in
  }

  property("Unwrapped IO should generate an exception") {
    a[BindingException] should be thrownBy extractCause[BindingException] {
      ChiselStage.emitCHIRRTL(new IOUnwrapped)
    }
  }

  property("Incoming/Outgoing should generate an input/output without coercing alignment") {
    class Test extends Module {
      val in = Incoming(new util.DecoupledIO(UInt(32.W)))
      val out = Outgoing(new util.DecoupledIO(UInt(32.W)))
      out :<>= in
    }
    val result = ChiselStage.emitCHIRRTL { new Test }
    matchesAndOmits(result)("input in", "output out", "flip ready")()
  }

  property("Incoming/Outgoing should fail an input/output") {
    class Test extends Module {
      val in = Incoming(Flipped(new util.DecoupledIO(UInt(32.W))))
      val out = Outgoing(Flipped(new util.DecoupledIO(UInt(32.W))))
      in :<>= out
    }
    val (err, _) = grabLog { intercept[ChiselException] { ChiselStage.emitCHIRRTL { new Test } } }
    matchesAndOmits(err)(
      "Incoming(..) cannot accept a chisel typed which is flipped",
      "Outgoing(..) cannot accept a chisel typed which is flipped"
    )()
  }
}
