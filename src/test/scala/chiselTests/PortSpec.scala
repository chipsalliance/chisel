package chiselTests

import chisel3._
import chisel3.testing.scalatest.FileCheck
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PortSpec extends AnyFlatSpec with Matchers with FileCheck {

  class DummyIO extends Bundle {
    val foo = Input(Bool())
    val bar = Input(UInt(8.W))
  }

  class Dummy extends Module {
    val in = IO(new DummyIO)
    val out = IO(Output(Bool()))
    out := in.foo.asUInt + in.bar
  }

  behavior of "Ports"

  they should "have source locators" in {
    info("Module-provided ports (clock and reset) point at the line with `Module`")
    info("User-defined ports point at the correct lines")
    ChiselStage
      .emitCHIRRTL(new Dummy)
      .fileCheck()(
        """|CHECK:      public module Dummy :
           |CHECK-NEXT:   input clock : Clock @[src/test/scala/chiselTests/PortSpec.scala 16:
           |CHECK-NEXT:   input reset : UInt<1> @[src/test/scala/chiselTests/PortSpec.scala 16:
           |CHECK-NEXT:   output in : { flip foo : UInt<1>, flip bar : UInt<8>} @[src/test/scala/chiselTests/PortSpec.scala 17:
           |CHECK-NEXT:   output out : UInt<1> @[src/test/scala/chiselTests/PortSpec.scala 18:
           |""".stripMargin
      )
  }
}
