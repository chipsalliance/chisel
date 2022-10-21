package chiselTests

import chisel3._
import circt.stage.ChiselStage

class PortSpec extends ChiselFreeSpec {

  class DummyIO extends Bundle {
    val foo = Input(Bool())
    val bar = Input(UInt(8.W))
  }

  class Dummy extends Module {
    val in = IO(new DummyIO)
    val out = IO(Output(Bool()))
    out := in.foo.asUInt + in.bar
  }

  "Ports now have source locators" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new Dummy)
    // Automatic clock and reset coming from Module do not get source locators
    chirrtl should include("input clock : Clock")
    chirrtl should include("input reset : UInt<1>")
    // other ports get source locators
    chirrtl should include(
      "output in : { flip foo : UInt<1>, flip bar : UInt<8>} @[src/test/scala/chiselTests/PortSpec.scala"
    )
    chirrtl should include("output out : UInt<1> @[src/test/scala/chiselTests/PortSpec.scala")
  }
}
