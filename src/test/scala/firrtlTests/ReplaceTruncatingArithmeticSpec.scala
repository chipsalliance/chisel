// See LICENSE for license details.

package firrtlTests

import firrtl._
import firrtl.ir._
import FirrtlCheckers._

class ReplaceTruncatingArithmeticSpec extends FirrtlFlatSpec {
  def compile(input: String): CircuitState =
    (new VerilogCompiler).compileAndEmit(CircuitState(parse(input), ChirrtlForm), List.empty)
  def compileBody(body: String) = {
    val str = """
      |circuit Test :
      |  module Test :
      |""".stripMargin + body.split("\n").mkString("    ", "\n    ", "")
    compile(str)
  }

  "Truncting addition" should "be inferred and emitted in Verilog" in {
    val result = compileBody(s"""
      |input x : UInt<8>
      |input y : UInt<8>
      |output z : UInt<8>
      |z <= tail(add(x, y), 1)""".stripMargin
    )
    result should containLine (s"assign z = x + y;")
  }
  it should "be inferred and emitted in Verilog even with an intermediate node" in {
    val result = compileBody(s"""
      |input x : UInt<8>
      |input y : UInt<8>
      |output z : UInt<8>
      |node n = add(x, y)
      |z <= tail(n, 1)""".stripMargin
    )
    result should containLine (s"assign z = x + y;")
  }
  "Truncting subtraction" should "be inferred and emitted in Verilog" in {
    val result = compileBody(s"""
      |input x : UInt<8>
      |input y : UInt<8>
      |output z : UInt<8>
      |z <= tail(sub(x, y), 1)""".stripMargin
    )
    result should containLine (s"assign z = x - y;")
  }
  "Tailing more than 1" should "not result in a truncating operator" in {
    val result = compileBody(s"""
      |input x : UInt<8>
      |input y : UInt<8>
      |output z : UInt<7>
      |node n = sub(x, y)
      |z <= tail(n, 2)""".stripMargin
    )
    result should containLine (s"assign n = x - y;")
    result should containLine (s"assign z = n[6:0];")
  }

}
