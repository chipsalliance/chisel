// See LICENSE for license details.

package firrtlTests

import java.io._
import org.scalatest._
import org.scalatest.prop._
import firrtl._
import firrtl.ir.Circuit
import firrtl.passes._
import firrtl.Parser.IgnoreInfo

class WidthSpec extends FirrtlFlatSpec {
  private def executeTest(input: String, expected: Seq[String], passes: Seq[Pass]) = {
    val c = passes.foldLeft(Parser.parse(input.split("\n").toIterator)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val lines = c.serialize.split("\n") map normalized

    expected foreach { e =>
      lines should contain(e)
    }
  }

  "Add of UInt<2> and SInt<2>" should "return SInt<4>" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      InferWidths)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input x: UInt<2>
        |    input y: SInt<2>
        |    output z: SInt
        |    z <= add(x, y)""".stripMargin
    val check = Seq( "output z : SInt<4>")
    executeTest(input, check, passes)
  }
  "SInt<2> - UInt<3>" should "return SInt<5>" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      InferWidths)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input x: UInt<3>
        |    input y: SInt<2>
        |    output z: SInt
        |    z <= sub(y, x)""".stripMargin
    val check = Seq( "output z : SInt<5>")
    executeTest(input, check, passes)
  }
  "Dshl by 32 bits" should "result in an error" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      InferWidths,
      CheckWidths)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input x: UInt<3>
        |    input y: UInt<32>
        |    output z: UInt
        |    z <= dshl(x, y)""".stripMargin
    intercept[CheckWidths.WidthTooBig] {
      executeTest(input, Nil, passes)
    }
  }
  "Circular reg depending on reg + 1" should "error" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      InferWidths,
      CheckWidths)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input clock: Clock
        |    input reset: UInt<1>
        |    reg r : UInt, clock with :
        |      reset => (reset, UInt(3))
        |    node T_7 = add(r, r)
        |    r <= T_7
        |""".stripMargin
    intercept[CheckWidths.UninferredWidth] {
      executeTest(input, Nil, passes)
    }
  }
}
