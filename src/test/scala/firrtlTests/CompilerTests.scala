// See LICENSE for license details.

package firrtlTests

import java.io.StringWriter

import org.scalatest.FlatSpec
import org.scalatest.Matchers
import org.scalatest.junit.JUnitRunner

import firrtl.ir.Circuit
import firrtl.{
  ChirrtlForm,
  CircuitState,
  Compiler,
  HighFirrtlCompiler,
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
   val writer = new StringWriter()
   def compiler: Compiler
   def input: String
   def check: String
   def getOutput: String = {
      compiler.compile(CircuitState(parse(input), ChirrtlForm), writer)
      writer.toString()
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
   val compiler = new VerilogCompiler()
   val input =
"""
circuit Top :
  module Top :
    input a : UInt<1>[2]
    output b : UInt<1>[2]
    b <= a
"""
   val check = Seq(
      "`ifdef RANDOMIZE_GARBAGE_ASSIGN",
      "`define RANDOMIZE",
      "`endif",
      "`ifdef RANDOMIZE_INVALID_ASSIGN",
      "`define RANDOMIZE",
      "`endif",
      "`ifdef RANDOMIZE_REG_INIT",
      "`define RANDOMIZE",
      "`endif",
      "`ifdef RANDOMIZE_MEM_INIT",
      "`define RANDOMIZE",
      "`endif",
      "",
      "module Top(",
      "  input   a_0,",
      "  input   a_1,",
      "  output  b_0,",
      "  output  b_1",
      ");",
      "  assign b_0 = a_0;",
      "  assign b_1 = a_1;",
      "endmodule\n"
   ).reduce(_ + "\n" + _)
   "A circuit's verilog output" should "match the given string" in {
      (getOutput) should be (check)
   }
}
