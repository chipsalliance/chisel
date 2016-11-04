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
  private def executeTest(input: String, notExpected: String, passes: Seq[Pass]) = {
    val c = passes.foldLeft(Parser.parse(input.split("\n").toIterator)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val lines = c.serialize.split("\n") map normalized

    lines foreach { l =>
      l.contains(notExpected) should be (false)
    }
  }
  "Expand Whens" should "not emit INVALID" in {
    val passes = Seq(
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
    executeTest(input, check, passes)
  }
}

class ExpandWhensExecutionTest extends ExecutionTest("ExpandWhens", "/passes/ExpandWhens")

