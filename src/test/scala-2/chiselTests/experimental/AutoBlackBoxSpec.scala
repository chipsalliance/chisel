// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental

import chisel3.experimental.SourceLine
import chisel3.experimental.hierarchy.Instantiate
import chisel3.util.experimental.AutoBlackBox
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AutoBlackBoxSpec extends AnyFlatSpec with Matchers {
  "AutoBlackBox" should "generate IO, module name and parameter" in {
    assert(
      ChiselStage
        .emitCHIRRTL(
          new chisel3.Module {
            implicit val info = SourceLine("Foo.scala", 1, 2)
            Instantiate(
              new AutoBlackBox(
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
                  |""".stripMargin,
                {
                  case "VDD" => false
                  case "VSS" => false
                  case _     => true
                }
              )
            )
          }
        )
        .contains(
          """extmodule BlackBoxPassthrough : @[Foo.scala 1:2]
            |    output out : UInt<2>
            |    input in : UInt<2>
            |    defname = BlackBoxPassthrough
            |""".stripMargin
        )
    )
  }
}
