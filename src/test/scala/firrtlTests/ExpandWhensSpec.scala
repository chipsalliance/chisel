// See LICENSE for license details.

package firrtlTests

import java.io._
import org.scalatest._
import org.scalatest.prop._
import firrtl._
import firrtl.passes._
import firrtl.ir._
import firrtl.Parser.IgnoreInfo

class ExpandWhensSpec extends FirrtlFlatSpec {
  private val transforms = Seq(
    ToWorkingIR,
    CheckHighForm,
    ResolveKinds,
    InferTypes,
    CheckTypes,
    Uniquify,
    ResolveKinds,
    InferTypes,
    ResolveGenders,
    CheckGenders,
    InferWidths,
    CheckWidths,
    PullMuxes,
    ExpandConnects,
    RemoveAccesses,
    ExpandWhens)
  private def executeTest(input: String, check: String, expected: Boolean) = {
    val circuit = Parser.parse(input.split("\n").toIterator)
    val result = transforms.foldLeft(CircuitState(circuit, UnknownForm)) {
      (c: CircuitState, p: Transform) => p.runTransform(c)
    }
    val c = result.circuit
    val lines = c.serialize.split("\n") map normalized

    if (expected) {
      c.serialize.contains(check) should be (true)
    } else {
      lines.foreach(_.contains(check) should be (false))
    }
  }
  "Expand Whens" should "not emit INVALID" in {
    val input =
  """|circuit Tester :
     |  module Tester :
     |    input p : UInt<1>
     |    when p :
     |      wire a : {b : UInt<64>, c : UInt<64>}
     |      a is invalid
     |      a.b <= UInt<64>("h04000000000000000")""".stripMargin
    val check = "INVALID"
    executeTest(input, check, false)
  }
  it should "void unwritten memory fields" in {
    val input =
  """|circuit Tester :
     |  module Tester :
     |    input clk : Clock
     |    mem memory:
     |      data-type => UInt<32>
     |      depth => 32
     |      reader => r0
     |      writer => w0
     |      read-latency => 0
     |      write-latency => 1
     |      read-under-write => undefined
     |    memory.r0.addr <= UInt<1>(1)
     |    memory.r0.en <= UInt<1>(1)
     |    memory.r0.clk <= clk
     |    memory.w0.addr <= UInt<1>(1)
     |    memory.w0.data <= UInt<1>(1)
     |    memory.w0.en <= UInt<1>(1)
     |    memory.w0.clk <= clk
     |    """.stripMargin
    val check = "VOID"
    executeTest(input, check, true)
  }
  it should "replace 'is invalid' with validif for wires that have a connection" in {
    val input =
      """|circuit Tester :
         |  module Tester :
         |    input p : UInt<1>
         |    output out : UInt
         |    wire w : UInt<32>
         |    w is invalid
         |    out <= w
         |    when p :
         |      w <= UInt(123)
         """.stripMargin
    val check = "validif(p"
    executeTest(input, check, true)
  }
  it should "leave 'is invalid' for wires that don't have a connection" in {
    val input =
      """|circuit Tester :
         |  module Tester :
         |    input p : UInt<1>
         |    output out : UInt
         |    wire w : UInt<32>
         |    w is invalid
         |    out <= w
         """.stripMargin
    val check = "w is invalid"
    executeTest(input, check, true)
  }
  it should "delete 'is invalid' for attached Analog wires" in {
    val input =
      """|circuit Tester :
         |  extmodule Child :
         |    input bus : Analog<32>
         |  module Tester :
         |    input bus : Analog<32>
         |    inst c of Child
         |    wire w : Analog<32>
         |    attach (w, bus)
         |    attach (w, c.bus)
         |    w is invalid
         """.stripMargin
    val check = "w is invalid"
    executeTest(input, check, false)
  }
  it should "correctly handle submodule inputs" in {
    val input =
      """circuit Test :
        |  module Child :
        |    input in : UInt<32>
        |  module Test :
        |    input in : UInt<32>[2]
        |    input p : UInt<1>
        |    inst c of Child
        |    when p :
        |      c.in <= in[0]
        |    else :
        |      c.in <= in[1]""".stripMargin
    val check = "mux(p, in[0], in[1])"
    executeTest(input, check, true)
  }
}

class ExpandWhensExecutionTest extends ExecutionTest("ExpandWhens", "/passes/ExpandWhens")

