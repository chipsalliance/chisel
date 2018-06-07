// See LICENSE for license details.

package firrtlTests

import java.io._
import org.scalatest._
import org.scalatest.prop._
import firrtl._
import firrtl.annotations._
import firrtl.ir.Circuit
import firrtl.passes._
import firrtl.Parser.IgnoreInfo

class InoutVerilogSpec extends FirrtlFlatSpec {

  behavior of "Analog"

  it should "attach a module input source directly" in {
    val compiler = new VerilogCompiler
    val input =
     """circuit Attaching :
        |  module Attaching :
        |    input an: Analog<3>
        |    inst a of A
        |    inst b of B
        |    attach (an, a.an1, b.an2)
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
    executeTest(input, check, compiler, Seq(dontDedup("A"), dontDedup("B")))
  }

  it should "attach two instances" in {
    val compiler = new VerilogCompiler
    val input =
     """circuit Attaching :
        |  module Attaching :
        |    inst a of A
        |    inst b of B
        |    attach (a.an, b.an)
        |  module A:
        |    input an: Analog<3>
        |  module B:
        |    input an: Analog<3>""".stripMargin
    val check =
     """module Attaching(
       |);
       |  wire [2:0] _GEN_0;
       |  A a (
       |    .an(_GEN_0)
       |  );
       |  B b (
       |    .an(_GEN_0)
       |  );
       |endmodule
       |module A(
       |  inout  [2:0] an
       |);
       |module B(
       |  inout  [2:0] an
       |);
       |endmodule
       |""".stripMargin.split("\n") map normalized
    executeTest(input, check, compiler, Seq(dontTouch("A.an"), dontDedup("A")))
  }

  it should "attach a wire source" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit Attaching :
         |  module Attaching :
         |    wire x: Analog
         |    inst a of A
         |    attach (x, a.an)
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
    executeTest(input, check, compiler, Seq(dontTouch("Attaching.x")))
  }

  it should "attach port to submodule port through a wire" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit Attaching :
         |  module Attaching :
         |    input an: Analog<3>
         |    wire x: Analog
         |    inst a of A
         |    attach (x, a.an)
         |    attach (x, an)
         |  module A:
         |    input an: Analog<3> """.stripMargin
    val check =
      """module Attaching(
        |  inout [2:0] an
        |);
        |  A a (
        |    .an(an)
        |  );
        |endmodule
        |""".stripMargin.split("\n") map normalized
    executeTest(input, check, compiler, Seq(dontTouch("Attaching.x")))
  }


  it should "attach multiple sources" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit Attaching :
         |  module Attaching :
         |    input a1 : Analog<3>
         |    input a2 : Analog<3>
         |    wire x: Analog<3>
         |    attach (x, a1, a2)""".stripMargin
    val check =
      """module Attaching(
        |  inout  [2:0] a1,
        |  inout  [2:0] a2
        |);
        |  `ifdef SYNTHESIS
        |    assign a1 = a2;
        |    assign a2 = a1;
        |  `elsif verilator
        |    `error "Verilator does not support alias and thus cannot arbirarily connect bidirectional wires and ports"
        |  `else
        |    alias a1 = a2;
        |  `endif
        |endmodule
        |""".stripMargin.split("\n") map normalized
    executeTest(input, check, compiler)
  }

  it should "work in partial connect" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit Attaching :
         |  module Attaching :
         |    input foo : { b : UInt<3>, a : Analog<3> }
         |    output bar : { b : UInt<3>, a : Analog<3> }
         |    bar <- foo""".stripMargin
    // Omitting `ifdef SYNTHESIS and `elsif verilator since it's tested above
    val check =
      """module Attaching(
        |  input  [2:0] foo_b,
        |  inout  [2:0] foo_a,
        |  output  [2:0] bar_b,
        |  inout  [2:0] bar_a
        |);
        |  assign bar_b = foo_b;
        |  alias bar_a = foo_a;
        |endmodule
        |""".stripMargin.split("\n") map normalized
    executeTest(input, check, compiler)
  }

  it should "preserve attach order" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit Attaching :
         |  module Attaching :
         |    input a : Analog<32>
         |    input b : Analog<32>
         |    input c : Analog<32>
         |    input d : Analog<32>
         |    attach (a, b)
         |    attach (c, b)
         |    attach (a, d)""".stripMargin
    val check =
      """module Attaching(
        |  inout  [31:0] a,
        |  inout  [31:0] b,
        |  inout  [31:0] c,
        |  inout  [31:0] d
        |);
        |    alias a = b = c = d;
        |endmodule
        |""".stripMargin.split("\n") map normalized
    executeTest(input, check, compiler)

    val input2 =
      """circuit Attaching :
         |  module Attaching :
         |    input a : Analog<32>
         |    input b : Analog<32>
         |    input c : Analog<32>
         |    input d : Analog<32>
         |    attach (a, b)
         |    attach (c, d)
         |    attach (d, a)""".stripMargin
    val check2 =
      """module Attaching(
        |  inout  [31:0] a,
        |  inout  [31:0] b,
        |  inout  [31:0] c,
        |  inout  [31:0] d
        |);
        |    alias a = b = c = d;
        |endmodule
        |""".stripMargin.split("\n") map normalized
    executeTest(input2, check2, compiler)
  }

  it should "infer widths" in {
    val compiler = new VerilogCompiler
    val input =
     """circuit Attaching :
        |  module Attaching :
        |    input an: Analog
        |    inst a of A
        |    attach (an, a.an1)
        |  module A:
        |    input an1: Analog<3>""".stripMargin
    val check =
     """module Attaching(
       |  inout  [2:0] an
       |);
       |  A a (
       |    .an1(an)
       |  );
       |endmodule
       |module A(
       |  inout  [2:0] an1
       |);
       |endmodule""".stripMargin.split("\n") map normalized
    executeTest(input, check, compiler)
  }

  it should "not error if not isinvalid" in {
    val compiler = new VerilogCompiler
    val input =
     """circuit Attaching :
        |  module Attaching :
        |    output an: Analog<3>
        |""".stripMargin
    val check =
     """module Attaching(
       |  inout  [2:0] an
       |);
       |endmodule""".stripMargin.split("\n") map normalized
    executeTest(input, check, compiler)
  }
  it should "not error if isinvalid" in {
    val compiler = new VerilogCompiler
    val input =
     """circuit Attaching :
        |  module Attaching :
        |    output an: Analog<3>
        |    an is invalid
        |""".stripMargin
    val check =
     """module Attaching(
       |  inout  [2:0] an
       |);
       |endmodule""".stripMargin.split("\n") map normalized
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
        |    input clock: Clock
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

  "Attaching a non-analog expression" should "not be ok" in {
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
        |    attach (source, a.o, b.o)
        |  extmodule A :
        |    output o: Analog<2>
        |  extmodule B:
        |    input o: Analog<2>""".stripMargin
    intercept[CheckTypes.OpNotAnalog] {
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
        |    attach (i, a.o)
        |  extmodule A :
        |    output o: Analog<2> """.stripMargin
    intercept[CheckWidths.AttachWidthsNotEqual] {
      passes.foldLeft(parse(input)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
    }
  }
}
