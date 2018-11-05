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
    import firrtl.ir.{UIntLiteral, SIntLiteral, IntWidth}
    s"$lit" should s"have signed width $sw" in {
      SIntLiteral(lit).width should equal (IntWidth(sw))
    }
    uwo.foreach { uw =>
      it should s"have unsigned width $uw" in {
        UIntLiteral(lit).width should equal (IntWidth(uw))
      }
    }
  }

  "Dshl by 20 bits" should "result in an error" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveGenders,
      InferWidths,
      CheckWidths)
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
      executeTest(input, Nil, passes)
    }
  }
  "Width >= MaxWidth" should "result in an error" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveGenders,
      InferWidths,
      CheckWidths)
    val input =
     s"""circuit Unit :
        |  module Unit :
        |    input x: UInt<${CheckWidths.MaxWidth}>
      """.stripMargin
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
      ResolveGenders,
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

  "Add of UInt<2> and SInt<2>" should "error" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveGenders,
      InferWidths)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input x: UInt<2>
        |    input y: SInt<2>
        |    output z: SInt
        |    z <= add(x, y)""".stripMargin
    val check = Seq( "output z : SInt<4>")
    intercept[PassExceptions] {
      executeTest(input, check, passes)
    }
  }

  "SInt<2> - UInt<3>" should "error" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveGenders,
      InferWidths)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input x: UInt<3>
        |    input y: SInt<2>
        |    output z: SInt
        |    z <= sub(y, x)""".stripMargin
    val check = Seq( "output z : SInt<5>")
    intercept[PassExceptions] {
      executeTest(input, check, passes)
    }
  }

  behavior of "CheckWidths.UniferredWidth"

  it should "provide a good error message with a full target if a user forgets an assign" in {
    val passes = Seq(
      ToWorkingIR,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveGenders,
      InferWidths,
      CheckWidths)
    val input =
      """|circuit Foo :
         |  module Foo :
         |    input clock : Clock
         |    inst bar of Bar
         |  module Bar :
         |    wire a: { b : UInt<1>, c : { d : UInt<1>, e : UInt } }
         |""".stripMargin
    val msg = intercept[CheckWidths.UninferredWidth] { executeTest(input, Nil, passes) }
      .getMessage should include ("""|    circuit Foo:
                                     |    └── module Bar:
                                     |        └── a.c.e""".stripMargin)
  }
}
