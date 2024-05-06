package chiselTest.experimental

import chisel3.experimental.AutoBlackBox
import chiseltest.ChiselScalatestTester
import org.scalatest.flatspec.AnyFlatSpec

class AutoBlackBoxSpec extends AnyFlatSpec with ChiselScalatestTester {
  "AutoBlackBox" should "generate IO, module name and parameter" in {
    assert(chisel3.stage.ChiselStage.emitChirrtl(
      new chisel3.Module {
        val inst = chisel3.Module(new AutoBlackBox {
          def verilog =
            """module BlackBoxPassthrough
              |#(
              |parameter int width = 2,
              |parameter real pi = 3.14 + 1,
              |parameter string str = "abc"
              |)
              |(
              |    input  [width-1:0] in,
              |    output [width-1:0] out,
              |    inout VDD,
              |    inout VSS
              |);
              |  assign out = in;
              |endmodule
              |""".stripMargin

          override def signalFilter: String => Boolean = {
            case "VDD" => false
            case "VSS" => false
            case "GND" => false
            case _     => true
          }
        })
      }
    ).contains(
      """extmodule BlackBoxPassthrough :
        |    output out : UInt<2>
        |    input in : UInt<2>
        |    defname = BlackBoxPassthrough
        |    parameter pi = 4.140000000000001
        |    parameter str = "abc"
        |    parameter width = 2
        |""".stripMargin)
    )
  }
}
