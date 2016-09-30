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
import firrtl.Annotations._
import firrtl.ir.Circuit
import firrtl.passes._
import firrtl.Parser.IgnoreInfo

class InoutVerilog extends FirrtlFlatSpec {
  private def executeTest(input: String, expected: Seq[String], compiler: Compiler) = {
    val writer = new StringWriter()
    compiler.compile(CircuitState(parse(input), ChirrtlForm), writer)
    val lines = writer.toString().split("\n") map normalized
    expected foreach { e =>
      lines should contain(e)
    }
  }
  "Circuit" should "attach a module input source" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit Attaching : 
         |  module Attaching : 
         |    input an: Analog<3>
         |    inst a of A
         |    inst b of B
         |    attach an to (a.an1, b.an2)
         |  module A: 
         |    input an1: Analog<3>
         |  module B:
         |    input an2: Analog<3> """.stripMargin
     val check = 
      """module Attaching(
        |  inout  [2:0] an
        |);
        |  A a (
        |    .an1(an)
        |  );
        |  B b (
        |    .an2(an)
        |  );
        |endmodule
        |module A(
        |  inout  [2:0] an1
        |);
        |endmodule
        |module B(
        |  inout  [2:0] an2
        |);
        |endmodule
        |""".stripMargin.split("\n") map normalized
     executeTest(input, check, compiler)
   }

  "Circuit" should "attach a module output source" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit Attaching : 
         |  module Attaching : 
         |    output an: Analog<3>
         |    inst a of A
         |    inst b of B
         |    attach an to (a.an1, b.an2)
         |  module A: 
         |    input an1: Analog<3>
         |  module B:
         |    input an2: Analog<3> """.stripMargin
     val check = 
      """module Attaching(
        |  inout  [2:0] an
        |);
        |  A a (
        |    .an1(an)
        |  );
        |  B b (
        |    .an2(an)
        |  );
        |endmodule
        |module A(
        |  inout  [2:0] an1
        |);
        |endmodule
        |module B(
        |  inout  [2:0] an2
        |);
        |endmodule
        |""".stripMargin.split("\n") map normalized
     executeTest(input, check, compiler)
   }

  "Circuit" should "not attach an instance input source" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit Attaching : 
         |  module Attaching : 
         |    inst a of A
         |    inst b of B
         |    attach a.an to (b.an)
         |  module A: 
         |    input an: Analog<3>
         |  module B:
         |    input an: Analog<3> """.stripMargin
     intercept[CheckTypes.IllegalAttachSource] {
       executeTest(input, Seq.empty, compiler)
     }
   }

  "Circuit" should "attach an instance output source" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit Attaching : 
         |  module Attaching : 
         |    inst a of A
         |    inst b of B
         |    attach b.an to (a.an)
         |  module A: 
         |    input an: Analog<3>
         |  module B:
         |    input an: Analog<3> """.stripMargin
    intercept[CheckTypes.IllegalAttachSource] {
      executeTest(input, Seq.empty, compiler)
    }
  }

  "Circuit" should "attach a wire source" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit Attaching : 
         |  module Attaching : 
         |    wire x: Analog
         |    inst a of A
         |    attach x to (a.an)
         |  module A: 
         |    input an: Analog<3> """.stripMargin
    val check = 
      """module Attaching(
        |);
        |  wire [2:0] x;
        |  A a (
        |    .an(x)
        |  );
        |endmodule
        |""".stripMargin.split("\n") map normalized
    executeTest(input, check, compiler)
  }
}

class AttachAnalogSpec extends FirrtlFlatSpec {
  private def executeTest(input: String, expected: Seq[String], passes: Seq[Pass]) = {
    val c = passes.foldLeft(Parser.parse(input.split("\n").toIterator)) {
      (c: Circuit, p: Pass) => p.run(c)
    }
    val lines = c.serialize.split("\n") map normalized

    expected foreach { e =>
      lines should contain(e)
    }
  }

  "Connecting analog types" should "throw an exception" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input y: Analog<1>
        |    output x: Analog<1>
        |    x <= y""".stripMargin
    intercept[CheckTypes.InvalidConnect] {
      passes.foldLeft(parse(input)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
    }
  }

  "Declaring register with analog types" should "throw an exception" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input clk: Clock
        |    reg r: Analog<2>, clk""".stripMargin
    intercept[CheckTypes.IllegalAnalogDeclaration] {
      passes.foldLeft(parse(input)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
    }
  }

  "Declaring memory with analog types" should "throw an exception" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input clk: Clock
        |    mem m: 
        |      data-type => Analog<2>
        |      depth => 4
        |      read-latency => 0
        |      write-latency => 1
        |      read-under-write => undefined""".stripMargin
    intercept[CheckTypes.IllegalAnalogDeclaration] {
      passes.foldLeft(parse(input)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
    }
  }

  "Declaring node with analog types" should "throw an exception" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input in: Analog<2>
        |    node n = in """.stripMargin
    intercept[CheckTypes.IllegalAnalogDeclaration] {
      passes.foldLeft(parse(input)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
    }
  }

  "Attaching a non-analog source" should "not be ok" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input source: UInt<2>
        |    inst a of A
        |    inst b of B
        |    attach source to (a.o, b.o)
        |  extmodule A :
        |    output o: Analog<2>
        |  extmodule B:
        |    input o: Analog<2>""".stripMargin
    intercept[CheckTypes.IllegalAttachSource] {
      passes.foldLeft(parse(input)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
    }
  }

  "Attach instance analog male source" should "not be ok." in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes)
    val input =
      """circuit Unit :
        |  module Unit :
        |    inst a of A
        |    inst b of B
        |    attach a.o to (b.o)
        |  extmodule A :
        |    output o: Analog<2>
        |  extmodule B:
        |    input o: Analog<2>""".stripMargin
    intercept[CheckTypes.IllegalAttachSource] {
      passes.foldLeft(parse(input)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
    }
  }

  "Attach instance analog female source" should "not be ok." in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes)
    val input =
      """circuit Unit :
        |  module Unit :
        |    inst a of A
        |    inst b of B
        |    attach b.o to (a.o)
        |  extmodule A :
        |    output o: Analog<2>
        |  extmodule B:
        |    input o: Analog<2>""".stripMargin
    intercept[CheckTypes.IllegalAttachSource] {
      passes.foldLeft(parse(input)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
    }
  }

  "Attach port analog expr" should "throw an exception" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input i: Analog<2>
        |    input j: Analog<2>
        |    attach j to (i) """.stripMargin
    intercept[CheckTypes.IllegalAttachExp] {
      passes.foldLeft(parse(input)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
    }
  }

  "Inequal attach widths" should "throw an exception" in {
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
        |    input i: Analog<3>
        |    inst a of A
        |    attach i to (a.o) 
        |  extmodule A :
        |    output o: Analog<2> """.stripMargin
    intercept[CheckWidths.AttachWidthsNotEqual] {
      passes.foldLeft(parse(input)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
    }
  }

  //"Simple compound expressions" should "be split" in {
  //  val passes = Seq(
  //    ToWorkingIR,
  //    ResolveKinds,
  //    InferTypes,
  //    ResolveGenders,
  //    InferWidths,
  //    SplitExpressions
  //  )
  //  val input =
  //    """circuit Top :
  //       |  module Top :
  //       |    input a : UInt<32>
  //       |    input b : UInt<32>
  //       |    input d : UInt<32>
  //       |    output c : UInt<1>
  //       |    c <= geq(add(a, b),d)""".stripMargin
  //  val check = Seq(
  //    "node GEN_0 = add(a, b)",
  //    "c <= geq(GEN_0, d)"
  //  )
  //  executeTest(input, check, passes)
  //}
}
