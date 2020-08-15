// See LICENSE for license details.

package firrtlTests.transforms

import firrtl._
import firrtl.testutils._
import firrtl.testutils.FirrtlCheckers.containLine

class LegalizeClocksTransformSpec extends FirrtlFlatSpec {
  def compile(input: String): CircuitState =
    (new MinimumVerilogCompiler).compileAndEmit(CircuitState(parse(input), ChirrtlForm), Nil)

  behavior.of("LegalizeClocksTransform")

  it should "not emit @(posedge 1'h0) for stop" in {
    val input =
      """circuit test :
        |  module test :
        |    stop(asClock(UInt(1)), UInt(1), 1)
        |""".stripMargin
    val result = compile(input)
    result should containLine(s"always @(posedge _GEN_0) begin")
    result.getEmittedCircuit.value shouldNot include("always @(posedge 1")
  }

  it should "not emit @(posedge 1'h0) for printf" in {
    val input =
      """circuit test :
        |  module test :
        |    printf(asClock(UInt(1)), UInt(1), "hi")
        |""".stripMargin
    val result = compile(input)
    result should containLine(s"always @(posedge _GEN_0) begin")
    result.getEmittedCircuit.value shouldNot include("always @(posedge 1")
  }

  it should "not emit @(posedge 1'h0) for reg" in {
    val input =
      """circuit test :
        |  module test :
        |    output out : UInt<8>
        |    input in : UInt<8>
        |    reg r : UInt<8>, asClock(UInt(0))
        |    r <= in
        |    out <= r
        |""".stripMargin
    val result = compile(input)
    result should containLine(s"always @(posedge _GEN_0) begin")
    result.getEmittedCircuit.value shouldNot include("always @(posedge 1")
  }

  it should "deduplicate injected nodes for literal clocks" in {
    val input =
      """circuit test :
        |  module test :
        |    printf(asClock(UInt(1)), UInt(1), "hi")
        |    stop(asClock(UInt(1)), UInt(1), 1)
        |""".stripMargin
    val result = compile(input)
    result should containLine(s"wire  _GEN_0 = 1'h1;")
    // Check that there's only 1 _GEN_0 instantiation
    val verilog = result.getEmittedCircuit.value
    val matches = "wire\\s+_GEN_0\\s+=\\s+1'h1".r.findAllIn(verilog)
    matches.size should be(1)

  }
}
