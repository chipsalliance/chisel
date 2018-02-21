// See LICENSE for license details.

package firrtlTests
package fixed

import java.io._
import firrtl._
import firrtl.ir.Circuit
import firrtl.passes._
import firrtl.Parser.IgnoreInfo

class FixedTypeInferenceSpec extends FirrtlFlatSpec {
  private def executeTest(input: String, expected: Seq[String], passes: Seq[Pass]) = {
    val c = passes.foldLeft(Parser.parse(input.split("\n").toIterator)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val lines = c.serialize.split("\n") map normalized

    expected foreach { e =>
      lines should contain(e)
    }
  }

  "Fixed types" should "infer add correctly" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveGenders,
      CheckGenders,
      InferWidths,
      CheckWidths)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input a : Fixed<10><<2>>
        |    input b : Fixed<10>
        |    input c : Fixed<4><<3>>
        |    output d : Fixed
        |    d <= add(a, add(b, c))""".stripMargin
    val check =
      """circuit Unit :
        |  module Unit :
        |    input a : Fixed<10><<2>>
        |    input b : Fixed<10><<0>>
        |    input c : Fixed<4><<3>>
        |    output d : Fixed<15><<3>>
        |    d <= add(a, add(b, c))""".stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }

  "Fixed types" should "be correctly shifted left" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveGenders,
      CheckGenders,
      InferWidths,
      CheckWidths)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input a : Fixed<10><<2>>
        |    output d : Fixed
        |    d <= shl(a, 2)""".stripMargin
    val check =
      """circuit Unit :
        |  module Unit :
        |    input a : Fixed<10><<2>>
        |    output d : Fixed<12><<2>>
        |    d <= shl(a, 2)""".stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }

  "Fixed types" should "be correctly shifted right" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveGenders,
      CheckGenders,
      InferWidths,
      CheckWidths)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input a : Fixed<10><<2>>
        |    output d : Fixed
        |    d <= shr(a, 2)""".stripMargin
    val check =
      """circuit Unit :
        |  module Unit :
        |    input a : Fixed<10><<2>>
        |    output d : Fixed<8><<2>>
        |    d <= shr(a, 2)""".stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }

  "Fixed types" should "relatively move binary point left" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveGenders,
      CheckGenders,
      InferWidths,
      CheckWidths)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input a : Fixed<10><<2>>
        |    output d : Fixed
        |    d <= bpshl(a, 2)""".stripMargin
    val check =
      """circuit Unit :
        |  module Unit :
        |    input a : Fixed<10><<2>>
        |    output d : Fixed<12><<4>>
        |    d <= bpshl(a, 2)""".stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }

  "Fixed types" should "relatively move binary point right" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveGenders,
      CheckGenders,
      InferWidths,
      CheckWidths)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input a : Fixed<10><<2>>
        |    output d : Fixed
        |    d <= bpshr(a, 2)""".stripMargin
    val check =
      """circuit Unit :
        |  module Unit :
        |    input a : Fixed<10><<2>>
        |    output d : Fixed<8><<0>>
        |    d <= bpshr(a, 2)""".stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }

  "Fixed types" should "absolutely set binary point correctly" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveGenders,
      CheckGenders,
      InferWidths,
      CheckWidths)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input a : Fixed<10><<2>>
        |    output d : Fixed
        |    d <= bpset(a, 3)""".stripMargin
    val check =
      """circuit Unit :
        |  module Unit :
        |    input a : Fixed<10><<2>>
        |    output d : Fixed<11><<3>>
        |    d <= bpset(a, 3)""".stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }

  "Fixed types" should "cat, head, tail, bits" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveGenders,
      CheckGenders,
      InferWidths,
      CheckWidths)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input a : Fixed<10><<2>>
        |    input b : Fixed<7><<3>>
        |    output cat : UInt
        |    output head : UInt
        |    output tail : UInt
        |    output bits : UInt
        |    cat <= cat(a, b)
        |    head <= head(a, 3)
        |    tail <= tail(a, 3)
        |    bits <= bits(a, 6, 3)""".stripMargin
    val check =
      """circuit Unit :
        |  module Unit :
        |    input a : Fixed<10><<2>>
        |    input b : Fixed<7><<3>>
        |    output cat : UInt<17>
        |    output head : UInt<3>
        |    output tail : UInt<7>
        |    output bits : UInt<4>
        |    cat <= cat(a, b)
        |    head <= head(a, 3)
        |    tail <= tail(a, 3)
        |    bits <= bits(a, 6, 3)""".stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }

  "Fixed types" should "be cast to" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveGenders,
      CheckGenders,
      InferWidths,
      CheckWidths)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input a : SInt<10>
        |    output d : Fixed
        |    d <= asFixedPoint(a, 2)""".stripMargin
    val check =
      """circuit Unit :
        |  module Unit :
        |    input a : SInt<10>
        |    output d : Fixed<10><<2>>
        |    d <= asFixedPoint(a, 2)""".stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }

  "Fixed types" should "support binary point of zero" in {
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
        |    input clock : Clock
        |    input reset : UInt<1>
        |    input io_in : Fixed<6><<0>>
        |    output io_out : Fixed<6><<0>>
        |
        |    io_in is invalid
        |    io_out is invalid
        |    io_out <= io_in
      """.stripMargin
    val check =
      """
        |circuit Unit :
        |  module Unit :
        |    input clock : Clock
        |    input reset : UInt<1>
        |    input io_in : SInt<6>
        |    output io_out : SInt<6>
        |
        |    io_out <= io_in
        |
      """.stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }
  "Fixed types" should "work with mems" in {
    def input(memType: String): String =
      s"""
        |circuit Unit :
        |  module Unit :
        |    input clock : Clock
        |    input in : Fixed<16><<8>>
        |    input ridx : UInt<3>
        |    output out : Fixed<16><<8>>
        |    input widx : UInt<3>
        |    $memType mem : Fixed<16><<8>>[8]
        |    infer mport min = mem[ridx], clock
        |    min <= in
        |    infer mport mout = mem[widx], clock
        |    out <= mout
      """.stripMargin
    def check(readLatency: Int, moutEn: Int, minEn: Int): String =
      s"""
        |circuit Unit :
        |  module Unit :
        |    input clock : Clock
        |    input in : SInt<16>
        |    input ridx : UInt<3>
        |    output out : SInt<16>
        |    input widx : UInt<3>
        |
        |    mem mem :
        |      data-type => SInt<16>
        |      depth => 8
        |      read-latency => $readLatency
        |      write-latency => 1
        |      reader => mout
        |      writer => min
        |      read-under-write => undefined
        |    out <= mem.mout.data
        |    mem.mout.addr <= widx
        |    mem.mout.en <= UInt<1>("h$moutEn")
        |    mem.mout.clk <= clock
        |    mem.min.addr <= ridx
        |    mem.min.en <= UInt<1>("h$minEn")
        |    mem.min.clk <= clock
        |    mem.min.data <= in
        |    mem.min.mask <= UInt<1>("h1")
      """.stripMargin
    executeTest(input("smem"), check(1, 0, 1).split("\n") map normalized, new LowFirrtlCompiler)
    executeTest(input("cmem"), check(0, 1, 1).split("\n") map normalized, new LowFirrtlCompiler)
  }
}

