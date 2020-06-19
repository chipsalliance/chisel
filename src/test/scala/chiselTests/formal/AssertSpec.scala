package chiselTests.formal

import chisel3._
import chisel3.formal.assert
import chiselTests.ChiselPropSpec

class AssertModule extends Module {
  val io = IO(new Bundle{
    val in = Input(UInt(8.W))
    val out = Output(UInt(8.W))
  })
  io.out := io.in
  when (io.in === 3.U) {
    assert(io.out === io.in)
  }
}

class AssertSpec extends ChiselPropSpec {
  property("basic equality check should work") {
    println(generateFirrtl(new AssertModule))
  }
}
