// Useful utilities for tests

package chiselTests

import chisel3._
import chisel3.experimental._

class PassthroughModuleIO extends Bundle {
  val in = Input(UInt(32.W))
  val out = Output(UInt(32.W))
}

trait AbstractPassthroughModule extends RawModule {
  val io = IO(new PassthroughModuleIO)
  io.out := io.in
}

class PassthroughModule extends Module with AbstractPassthroughModule
class PassthroughMultiIOModule extends MultiIOModule with AbstractPassthroughModule
class PassthroughRawModule extends RawModule with AbstractPassthroughModule


