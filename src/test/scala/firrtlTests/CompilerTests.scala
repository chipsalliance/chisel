// See LICENSE for license details.

package firrtlTests

import org.scalatest.FlatSpec
import org.scalatest.Matchers
import org.scalatest.junit.JUnitRunner

import firrtl.ir.Circuit
import firrtl.{
  ChirrtlForm,
  CircuitState,
  Compiler,
  HighFirrtlCompiler,
  MiddleFirrtlCompiler,
  LowFirrtlCompiler,
  Parser,
  VerilogCompiler
}

/**
 * An example methodology for testing Firrtl compilers.
 *
 * Given an input Firrtl circuit (expressed as a string),
 * the compiler is executed. The output of the compiler
 * should be compared against the check string.
 */
abstract class CompilerSpec extends FlatSpec {
   def parse (s: String): Circuit = Parser.parse(s.split("\n").toIterator)
   def compiler: Compiler
   def input: String
   def check: String
   def getOutput: String = {
     val res = compiler.compileAndEmit(CircuitState(parse(input), ChirrtlForm))
     res.getEmittedCircuit.value
   }
}

/**
 * An example test for testing the HighFirrtlCompiler.
 *
 * Given an input Firrtl circuit (expressed as a string),
 * the compiler is executed. The output of the compiler
 * is parsed again and compared (in-memory) to the parsed
 * input.
 */
class HighFirrtlCompilerSpec extends CompilerSpec with Matchers {
   val compiler = new HighFirrtlCompiler()
   val input =
"""circuit Top :
  module Top :
    input a : UInt<1>[2]
    node x = a
"""
   val check = input
   "Any circuit" should "match exactly to its input" in {
      (parse(getOutput)) should be (parse(check))
   }
}

/**
  * An example test for testing the MiddleFirrtlCompiler.
  *
  * Given an input Firrtl circuit (expressed as a string),
  * the compiler is executed. The output of the compiler is
  * a lowered (to MidForm) version of the input circuit. The output is
  * string compared to the correct lowered circuit.
  */
class MiddleFirrtlCompilerSpec extends CompilerSpec with Matchers {
   val compiler = new MiddleFirrtlCompiler()
   val input =
      """
circuit Top :
  module Top :
    input reset : UInt<1>
    input a : UInt<1>[2]
    wire b : UInt
    b <= a[0]
    when reset :
      b <= UInt(0)
"""
   // Verify that Vecs are retained, but widths are inferred and whens are expanded.
   val check = Seq(
      "circuit Top :",
      "  module Top :",
      "    input reset : UInt<1>",
      "    input a : UInt<1>[2]",
      "    wire b : UInt<1>",
      "    node _GEN_0 = mux(reset, UInt<1>(\"h0\"), a[0])",
      "    b <= _GEN_0\n\n"
   ).reduce(_ + "\n" + _)
   "A circuit" should "match exactly to its MidForm state" in {
      (parse(getOutput)) should be (parse(check))
   }
}

/**
 * An example test for testing the LoweringCompiler.
 *
 * Given an input Firrtl circuit (expressed as a string),
 * the compiler is executed. The output of the compiler is
 * a lowered version of the input circuit. The output is
 * string compared to the correct lowered circuit.
 */
class LowFirrtlCompilerSpec extends CompilerSpec with Matchers {
   val compiler = new LowFirrtlCompiler()
   val input =
"""
circuit Top :
  module Top :
    input a : UInt<1>[2]
    node x = a
"""
   val check = Seq(
      "circuit Top :",
      "  module Top :",
      "    input a_0 : UInt<1>",
      "    input a_1 : UInt<1>",
      "    node x_0 = a_0",
      "    node x_1 = a_1\n\n"
   ).reduce(_ + "\n" + _)
   "A circuit" should "match exactly to its lowered state" in {
      (parse(getOutput)) should be (parse(check))
   }
}

/**
 * An example test for testing the VerilogCompiler.
 *
 * Given an input Firrtl circuit (expressed as a string),
 * the compiler is executed. The output of the compiler is
 * the corresponding Verilog. The output is string compared
 * to the correct Verilog.
 */
class VerilogCompilerSpec extends CompilerSpec with Matchers {
   val input = """circuit Top :
                 |  module Top :
                 |    input a : UInt<1>[2]
                 |    output b : UInt<1>[2]
                 |    b <= a""".stripMargin
   val check = """module Top(
                 |  input   a_0,
                 |  input   a_1,
                 |  output  b_0,
                 |  output  b_1
                 |);
                 |  assign b_0 = a_0;
                 |  assign b_1 = a_1;
                 |endmodule
                 |""".stripMargin
   def compiler = new VerilogCompiler()
   "A circuit's verilog output" should "match the given string and not have RANDOMIZE if no invalids" in {
      getOutput should be (check)
   }
}
