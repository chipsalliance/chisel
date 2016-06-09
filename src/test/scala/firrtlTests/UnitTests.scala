/*
Copyright (c) 2014 - 2016 The Regents of the University of
California (Regents). All Rights Reserved.  Redistribution and use in
source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
   * Redistributions of source code must retain the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer.
   * Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer in the documentation and/or other materials
     provided with the distribution.
   * Neither the name of the Regents nor the names of its contributors
     may be used to endorse or promote products derived from this
     software without specific prior written permission.
IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF
ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION
TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
MODIFICATIONS.
*/

package firrtlTests

import java.io._
import org.scalatest._
import org.scalatest.prop._
import firrtl._
import firrtl.passes._
import firrtl.Parser.IgnoreInfo

class UnitTests extends FirrtlFlatSpec {
  def parse (input:String) = Parser.parse(input.split("\n").toIterator, IgnoreInfo)
  private def executeTest(input: String, expected: Seq[String], passes: Seq[Pass]) = {
    val c = passes.foldLeft(Parser.parse(input.split("\n").toIterator)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val lines = c.serialize.split("\n") map normalized

    expected foreach { e =>
      lines should contain(e)
    }
  }

  "Connecting bundles of different types" should "throw an exception" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input y: {a : UInt<1>}
        |    output x: {a : UInt<1>, b : UInt<1>}
        |    x <= y""".stripMargin
    intercept[CheckTypes.InvalidConnect] {
      passes.foldLeft(parse(input)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
    }
  }

  "Initializing a register with a different type" should "throw an exception" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes)
    val input =
     """circuit Unit :
       |  module Unit :
       |    input clk : Clock
       |    input reset : UInt<1>
       |    wire x : { valid : UInt<1> }
       |    reg y : { valid : UInt<1>, bits : UInt<3> }, clk with :
       |      reset => (reset, x)""".stripMargin
    intercept[CheckTypes.InvalidRegInit] {
      passes.foldLeft(parse(input)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
    }
  }

  "Partial connection two bundle types whose relative flips don't match but leaf node directions do" should "connect correctly" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ExpandConnects)
    val input =
     """circuit Unit :
       |  module Unit :
       |    wire x : { flip a: { b: UInt<32> } }
       |    wire y : { a: { flip b: UInt<32> } }
       |    x <- y""".stripMargin
    val check =
     """circuit Unit :
       |  module Unit :
       |    wire x : { flip a: { b: UInt<32> } }
       |    wire y : { a: { flip b: UInt<32> } }
       |    y.a.b <= x.a.b""".stripMargin
    val c_result = passes.foldLeft(parse(input)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val writer = new StringWriter()
    FIRRTLEmitter.run(c_result,writer)
    (parse(writer.toString())) should be (parse(check))
  }

  val splitExpTestCode =
     """
       |circuit Unit :
       |  module Unit :
       |    input a : UInt<1>
       |    input b : UInt<2>
       |    input c : UInt<2>
       |    output out : UInt<1>
       |    out <= bits(mux(a, b, c), 0, 0)
       |""".stripMargin

  "Emitting a nested expression" should "throw an exception" in {
    val passes = Seq(
      ToWorkingIR,
      InferTypes)
    intercept[PassException] {
      val c = Parser.parse(splitExpTestCode.split("\n").toIterator)
      val c2 = passes.foldLeft(c)((c, p) => p run c)
      new VerilogEmitter().run(c2, new OutputStreamWriter(new ByteArrayOutputStream))
    }
  }

  "After splitting, emitting a nested expression" should "compile" in {
    val passes = Seq(
      ToWorkingIR,
      SplitExpressions,
      InferTypes)
    val c = Parser.parse(splitExpTestCode.split("\n").toIterator)
    val c2 = passes.foldLeft(c)((c, p) => p run c)
    new VerilogEmitter().run(c2, new OutputStreamWriter(new ByteArrayOutputStream))
  }

  "Simple compound expressions" should "be split" in {
    val passes = Seq(
      ToWorkingIR,
      ResolveKinds,
      InferTypes,
      ResolveGenders,
      InferWidths,
      SplitExpressions
    )
    val input =
      """circuit Top :
         |  module Top :
         |    input a : UInt<32>
         |    input b : UInt<32>
         |    input d : UInt<32>
         |    output c : UInt<1>
         |    c <= geq(add(a, b),d)""".stripMargin
    val check = Seq(
      "node GEN_0 = add(a, b)",
      "c <= geq(GEN_0, d)"
    )
    executeTest(input, check, passes)
  }

  "Smaller widths" should "be explicitly padded" in {
    val passes = Seq(
      ToWorkingIR,
      ResolveKinds,
      InferTypes,
      ResolveGenders,
      InferWidths,
      PadWidths
    )
    val input =
      """circuit Top :
         |  module Top :
         |    input a : UInt<32>
         |    input b : UInt<20>
         |    input pred : UInt<1>
         |    output c : UInt<32>
         |    c <= mux(pred,a,b)""".stripMargin
     val check = Seq("c <= mux(pred, a, pad(b, 32))")
     executeTest(input, check, passes)
  }
}
