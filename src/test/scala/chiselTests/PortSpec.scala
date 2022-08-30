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
    println(chirrtl)

    chirrtl should include ("input clock : Clock @[Module.scala 120:30]")
    chirrtl should include ("input reset : UInt<1> @[Module.scala 121:30]")
    chirrtl should include ("output in : { flip foo : UInt<1>, flip bar : UInt<8>} @[PortSpec.scala 14:16]")
    chirrtl should include ("output out : UInt<1> @[PortSpec.scala 15:17]")
  }
}
