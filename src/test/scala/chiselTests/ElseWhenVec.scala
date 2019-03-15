// See LICENSE for license details.

package chiselTests

import chisel3._

trait DebugModuleBundle extends Bundle {
  val nComponents = 4
  val debugUnavail = Input(Vec(nComponents, Bool()))
}

class ElseWhenVec extends Module {
  val io = IO(new DebugModuleBundle {
    val in = Input(UInt(16.W))
    val out = Output(UInt(16.W))
    val trigger = Input(Bool())
  })
  when (io.in === 0.U) {
    io.out := io.in
  } .elsewhen ( io.debugUnavail(0.U) ) {
    io.out := 1.U
  }
}

class ElseWhenVecSpec extends ChiselPropSpec {
  property("ElseWhenVec should elaborate") {
    elaborate { new ElseWhenVec }
  }
}
