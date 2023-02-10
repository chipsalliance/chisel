// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl._
import firrtl.passes._
import firrtl.testutils._

class WidthSpec extends FirrtlFlatSpec {
  private def executeTest(input: String, expected: Seq[String], passes: Seq[Transform]) = {
    val c = passes
      .foldLeft(CircuitState(Parser.parse(input.split("\n").toIterator), UnknownForm)) {
        (c: CircuitState, p: Transform) => p.runTransform(c)
      }
      .circuit
    val lines = c.serialize.split("\n").map(normalized)

    expected.foreach { e =>
      lines should contain(e)
    }
  }

  private val inferPasses = Seq(ToWorkingIR)

  private val inferAndCheckPasses = inferPasses

  case class LiteralWidthCheck(lit: BigInt, uIntWidth: Option[BigInt], sIntWidth: BigInt)
  val litChecks = Seq(
    LiteralWidthCheck(-4, None, 3),
    LiteralWidthCheck(-3, None, 3),
    LiteralWidthCheck(-2, None, 2),
    LiteralWidthCheck(-1, None, 1),
    LiteralWidthCheck(0, Some(1), 1), // TODO https://github.com/freechipsproject/firrtl/pull/530
    LiteralWidthCheck(1, Some(1), 2),
    LiteralWidthCheck(2, Some(2), 3),
    LiteralWidthCheck(3, Some(2), 3),
    LiteralWidthCheck(4, Some(3), 4)
  )
  for (LiteralWidthCheck(lit, uwo, sw) <- litChecks) {
    import firrtl.ir.{IntWidth, SIntLiteral, UIntLiteral}
    s"$lit" should s"have signed width $sw" in {
      SIntLiteral(lit).width should equal(IntWidth(sw))
    }
    uwo.foreach { uw =>
      it should s"have unsigned width $uw" in {
        UIntLiteral(lit).width should equal(IntWidth(uw))
      }
    }
  }

  "Dshl by 20 bits" should "result in an error" in {
    val input =
      """circuit Unit :
        |  module Unit :
        |    input x: UInt<3>
        |    input y: UInt<32>
        |    output z: UInt
        |    z <= dshl(x, y)""".stripMargin
    // Throws both DshlTooBig and WidthTooBig
    // TODO check message
    intercept[PassExceptions] {
      executeTest(input, Nil, inferAndCheckPasses)
    }
  }

  "Add of UInt<2> and SInt<2>" should "error" in {
    val input =
      """circuit Unit :
        |  module Unit :
        |    input x: UInt<2>
        |    input y: SInt<2>
        |    output z: SInt
        |    z <= add(x, y)""".stripMargin
    val check = Seq("output z : SInt<4>")
    intercept[PassExceptions] {
      executeTest(input, check, inferPasses)
    }
  }

  "SInt<2> - UInt<3>" should "error" in {
    val input =
      """circuit Unit :
        |  module Unit :
        |    input x: UInt<3>
        |    input y: SInt<2>
        |    output z: SInt
        |    z <= sub(y, x)""".stripMargin
    val check = Seq("output z : SInt<5>")
    intercept[PassExceptions] {
      executeTest(input, check, inferPasses)
    }
  }

}
