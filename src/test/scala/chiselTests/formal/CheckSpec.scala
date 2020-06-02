package chiselTests.formal

import chisel3._
import chisel3.formal.check
import chiselTests.ChiselPropSpec

class CheckModule extends Module {
  val io = IO(new Bundle{
    val in = Input(UInt(8.W))
    val out = Output(UInt(8.W))
  })
  io.out := io.in
  check(io.out === io.in)
}

class CheckSpec extends ChiselPropSpec {
  property("basic equality check should work") {
    println(compile(new CheckModule))
  }
}
