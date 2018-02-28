// See LICENSE for license details.

package firrtlTests
package fixed

import firrtl._
import firrtl.ir.Circuit
import firrtl.passes._
import firrtl.Parser.IgnoreInfo

class RemoveFixedTypeSpec extends FirrtlFlatSpec {
  private def executeTest(input: String, expected: Seq[String], passes: Seq[Pass]) = {
    val c = passes.foldLeft(Parser.parse(input.split("\n").toIterator)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val lines = c.serialize.split("\n") map normalized
    println(c.serialize)

    expected foreach { e =>
      lines should contain(e)
    }
  }

  "Fixed types" should "be removed" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveGenders,
      CheckGenders,
      InferWidths,
      CheckWidths,
      ConvertFixedToSInt)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input a : Fixed<10><<2>>
        |    input b : Fixed<10>
        |    input c : Fixed<4><<3>>
        |    output d : Fixed<<5>>
        |    d <= add(a, add(b, c))""".stripMargin
    val check =
      """circuit Unit :
         |  module Unit :
         |    input a : SInt<10>
         |    input b : SInt<10>
         |    input c : SInt<4>
         |    output d : SInt<15>
         |    d <= shl(add(shl(a, 1), add(shl(b, 3), c)), 2)""".stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }
  "Fixed types" should "be removed, even with a bulk connect" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveGenders,
      CheckGenders,
      InferWidths,
      CheckWidths,
      ConvertFixedToSInt)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input a : Fixed<10><<2>>
        |    input b : Fixed<10>
        |    input c : Fixed<4><<3>>
        |    output d : Fixed<<5>>
        |    d <- add(a, add(b, c))""".stripMargin
    val check =
      """circuit Unit :
         |  module Unit :
         |    input a : SInt<10>
         |    input b : SInt<10>
         |    input c : SInt<4>
         |    output d : SInt<15>
         |    d <- shl(add(shl(a, 1), add(shl(b, 3), c)), 2)""".stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }

  "Fixed types" should "remove binary point shift correctly" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveGenders,
      CheckGenders,
      InferWidths,
      CheckWidths,
      ConvertFixedToSInt)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input a : Fixed<10><<2>>
        |    output d : Fixed<12><<4>>
        |    d <= bpshl(a, 2)""".stripMargin
    val check =
      """circuit Unit :
        |  module Unit :
        |    input a : SInt<10>
        |    output d : SInt<12>
        |    d <= shl(a, 2)""".stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }

  "Fixed types" should "remove binary point shift correctly in reverse" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveGenders,
      CheckGenders,
      InferWidths,
      CheckWidths,
      ConvertFixedToSInt)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input a : Fixed<10><<2>>
        |    output d : Fixed<9><<1>>
        |    d <= bpshr(a, 1)""".stripMargin
    val check =
      """circuit Unit :
        |  module Unit :
        |    input a : SInt<10>
        |    output d : SInt<9>
        |    d <= shr(a, 1)""".stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }

  "Fixed types" should "remove an absolutely set binary point correctly" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveGenders,
      CheckGenders,
      InferWidths,
      CheckWidths,
      ConvertFixedToSInt)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input a : Fixed<10><<2>>
        |    output d : Fixed
        |    d <= bpset(a, 3)""".stripMargin
    val check =
      """circuit Unit :
        |  module Unit :
        |    input a : SInt<10>
        |    output d : SInt<11>
        |    d <= shl(a, 1)""".stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }

  "Fixed point numbers" should "allow binary point to be set to zero at creation" in {
    val input =
      """
        |circuit Unit :
        |  module Unit :
        |    input clock : Clock
        |    input reset : UInt<1>
        |    input io_in : Fixed<6><<0>>
        |    output io_out : Fixed
        |
        |    io_in is invalid
        |    io_out is invalid
        |    io_out <= io_in
      """.stripMargin

    class CheckChirrtlTransform extends SeqTransform {
      def inputForm = ChirrtlForm
      def outputForm = ChirrtlForm
      val transforms = Seq(passes.CheckChirrtl)
    }

    val chirrtlTransform = new CheckChirrtlTransform
    chirrtlTransform.execute(CircuitState(parse(input), ChirrtlForm))
  }

  "Fixed point numbers" should "remove nested AsFixedPoint" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveGenders,
      CheckGenders,
      InferWidths,
      CheckWidths,
      ConvertFixedToSInt)
    val input =
      """
        |circuit Unit :
        |  module Unit :
        |    node x = asFixedPoint(asFixedPoint(UInt(3), 0), 1)
      """.stripMargin
    val check =
      """
        |circuit Unit :
        |  module Unit :
        |    node x = asSInt(asSInt(UInt<2>("h3")))
      """.stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }
}

