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
  private def executeTest(input: String, check: String, transforms: Seq[Transform], expected: Boolean) = {
    val circuit = Parser.parse(input.split("\n").toIterator)
    val result = transforms.foldLeft(CircuitState(circuit, UnknownForm)) {
      (c: CircuitState, p: Transform) => p.runTransform(c)
    }
    val c = result.circuit
    val lines = c.serialize.split("\n") map normalized
    println(c.serialize)

    if(expected) {
      c.serialize.contains(check) should be (true)
    } else {
      lines foreach { l => l.contains(check) should be (false) }
    }
  }
  "Expand Whens" should "not emit INVALID" in {
    val transforms = Seq(
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
    val input =
  """|circuit Tester : 
     |  module Tester :
     |    input p : UInt<1>
     |    when p :
     |      wire a : {b : UInt<64>, c : UInt<64>}
     |      a is invalid
     |      a.b <= UInt<64>("h04000000000000000")""".stripMargin
    val check = "INVALID"
    executeTest(input, check, transforms, false)
  }
  "Expand Whens" should "void unwritten memory fields" in {
    val transforms = Seq(
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
    executeTest(input, check, transforms, true)
  }
}

class ExpandWhensExecutionTest extends ExecutionTest("ExpandWhens", "/passes/ExpandWhens")

