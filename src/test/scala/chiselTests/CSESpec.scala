// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental.chiselName
import chisel3.stage.ChiselStage

@chiselName
class CSEModule extends RawModule {
  val in1 = IO(Input(Bool()))
  val in2 = IO(Input(Bool()))
  val in3 = IO(Input(Bool()))
  val in4 = IO(Input(Bool()))
  val out1 = IO(Output(Bool()))
  val out2 = IO(Output(Bool()))

  val foo = in1 ^ in2 ^ in3

  out1 := foo

  out2 := false.B
  when (in4) {
    val bar = in1 ^ in2 ^ in3 // these expressions should be CSE'd with foo
    out2 := bar
  }
}

class CSESpec extends ChiselFlatSpec with Utils {

  behavior of "CSE"

  it should "remove a redundant combinational logic expression" in {
    val fir = (new ChiselStage).emitChirrtl(new CSEModule)
    fir should include ("foo")
    fir should not include ("bar")
  }
}
