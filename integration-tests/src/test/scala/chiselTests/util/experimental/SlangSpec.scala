// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.experimental
import chisel3.experimental.{DoubleParam, IntParam, StringParam}
import chisel3.util.experimental.SlangUtils
import org.scalatest.flatspec.AnyFlatSpec

class SlangSpec extends AnyFlatSpec {
  val verilog =
    """module BlackBoxPassthrough
      |#(
      |parameter int width = 2,
      |parameter real pi1 = 3.14 + 1,
      |parameter string str = "abc"
      |)
      |(
      |    input  [width-1:0] in,
      |    output [width-1:0] out
      |);
      |  assign out = in;
      |endmodule
      |""".stripMargin
  "slang" should "parse" in {
    SlangUtils.getVerilogAst(verilog)
  }
  "slang" should "get port of verilog" in {
    val bundle = SlangUtils.verilogModuleIO(SlangUtils.getVerilogAst(verilog)).map(e => (e._1, e._2.toString))
    assert(bundle.contains(("in", "UInt<2>")))
    assert(bundle.contains(("out", "UInt<2>")))
  }
  "slang" should "get module of verilog" in {
    assert(SlangUtils.verilogModuleName(SlangUtils.getVerilogAst(verilog)) == "BlackBoxPassthrough")
  }
  "slang" should "get parameter of module" in {
    assert(
      SlangUtils.verilogParameter(SlangUtils.getVerilogAst(verilog)) == List(
        "width" -> IntParam(2),
        "pi1" -> DoubleParam(3.14 + 1),
        "str" -> StringParam("abc")
      )
    )
  }
}
