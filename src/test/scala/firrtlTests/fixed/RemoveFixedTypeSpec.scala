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
package fixed

import firrtl.Annotations.AnnotationMap
import firrtl._
import firrtl.ir.Circuit
import firrtl.passes._
import firrtl.Parser.IgnoreInfo

class RemoveFixedTypeSpec extends FirrtlFlatSpec {
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
        |    input clk : Clock
        |    input reset : UInt<1>
        |    input io_in : Fixed<6><<0>>
        |    output io_out : Fixed
        |
        |    io_in is invalid
        |    io_out is invalid
        |    io_out <= io_in
      """.stripMargin

    class CheckChirrtlTransform extends Transform with SimpleRun {
      val passSeq = Seq(passes.CheckChirrtl)
      def execute (circuit: Circuit, annotationMap: AnnotationMap): TransformResult =
        run(circuit, passSeq)
    }

    val chirrtlTransform = new CheckChirrtlTransform
    chirrtlTransform.execute(parse(input), new AnnotationMap(Seq.empty))
  }
}

