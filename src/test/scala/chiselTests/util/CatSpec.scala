// SPDX-License-Identifier: Apache-2.0

package chiselTests.util

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.util.Cat

import chiselTests.ChiselFlatSpec

object CatSpec {

  class JackIsATypeSystemGod extends Module {
    val in  = IO(Input (Vec(0, UInt(8.W))))
    val out = IO(Output(UInt(8.W)))

    out := Cat(in)
  }

}

class CatSpec extends ChiselFlatSpec {

  import CatSpec._

  behavior of "util.Cat"

  it should "not fail to elaborate a zero-element Vec" in {

    ChiselStage.elaborate(new JackIsATypeSystemGod)

  }

}
