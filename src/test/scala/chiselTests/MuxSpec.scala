// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.testers.BasicTester

class MuxTester extends BasicTester {
  assert(Mux(0.B, 1.U, 2.U) === 2.U)
  assert(Mux(1.B, 1.U, 2.U) === 1.U)
  val dontCareMux1 = Wire(UInt())
  dontCareMux1 := Mux(0.B, DontCare, 4.U)  // note: Mux output of type Element
  assert(dontCareMux1 === 4.U)

  val dontCareMux2 = Wire(UInt())
  dontCareMux2 := Mux(1.B, 3.U, DontCare)  // note: Mux output of type Element
  assert(dontCareMux2 === 3.U)

  Mux(0.B, 3.U, DontCare)  // just to make sure nothing crashes, any result is valid
  stop()
}

class MuxSpec extends ChiselFlatSpec {
  "Mux" should "pass basic checks" in {
    assertTesterPasses { new MuxTester }
  }
}
