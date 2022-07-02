// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental

import chisel3._
import chisel3.stage.ChiselStage
import chiselTests.ChiselFlatSpec

class ConstpropSpec extends ChiselFlatSpec {
  behavior.of("Constprop")

  it should "propagate UInt constants" in {
    class MyModule extends RawModule {
      val in = IO(Input(UInt(8.W)))
      ("b11".U ## (("b1000".U | "b0100".U) ^ 1.U) & "b1111".U).litValue should be("b1101".U.litValue)
      (in & 0.U(8.W)).litValue should be(0)
      (in | 0.U(8.W)) should be(in)
    }
    ChiselStage.elaborate(new MyModule)
  }
}
