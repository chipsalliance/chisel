// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage

class WireSpec extends ChiselFlatSpec {
  "WireDefault.apply" should "work" in {
    assertCompiles("WireDefault(UInt(4.W), 2.U)")
  }
  it should "allow DontCare" in {
    assertCompiles("WireDefault(UInt(4.W), DontCare)")
  }
  it should "not allow DontCare to affect type inference" in {
    assertCompiles("val x: UInt = WireDefault(UInt(4.W), DontCare)")
  }
  it should "not allow init argument to affect type inference" in {
    assertDoesNotCompile("val x: UInt = WireDefault(UInt(4.W), 2.S)")
  }
  it should "have source locator information on wires" in {
    class Dummy extends chisel3.Module {
      val in = IO(Input(Bool()))
      val out = IO(Output(Bool()))

      val wire = WireInit(Bool(), true.B)
      val wire2 = Wire(Bool())
      wire2 := in
      out := in & wire & wire2
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new Dummy)
    chirrtl should include("wire wire : UInt<1>")
    chirrtl should include("wire wire2 : UInt<1>")
  }
}
