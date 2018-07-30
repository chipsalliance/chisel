package chisel3.tests

import chisel3._

class PassthroughModule[T <: Data](ioType: T) extends Module {
  val io = IO(new Bundle {
    val in = Input(ioType)
    val out = Output(ioType)
  })
  io.out := io.in
}
