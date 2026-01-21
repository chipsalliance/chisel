// SPDX-License-Identifier: Apache-2.0

// Useful utilities for tests

package chiselTests

import chisel3._
import chisel3.internal.firrtl.Width
import _root_.firrtl.{ir => firrtlir}

class PassthroughModuleIO extends Bundle {
  val in = Input(UInt(32.W))
  val out = Output(UInt(32.W))
}

trait AbstractPassthroughModule extends RawModule {
  val io = IO(new PassthroughModuleIO)
  io.out := io.in
}

class PassthroughModule extends Module with AbstractPassthroughModule
class PassthroughMultiIOModule extends Module with AbstractPassthroughModule
class PassthroughRawModule extends RawModule with AbstractPassthroughModule
