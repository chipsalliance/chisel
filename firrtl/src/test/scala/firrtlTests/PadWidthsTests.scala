// See LICENSE for license details.

package firrtlTests

import firrtl.CircuitState
import firrtl.options.Dependency
import firrtl.stage.{Forms, TransformManager}
import firrtl.testutils.LeanTransformSpec

class PadWidthsTests extends LeanTransformSpec(Seq(Dependency(firrtl.passes.PadWidths))) {
  behavior.of("PadWidths pass")

  it should "pad widths inside a mux" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input a : UInt<32>
        |    input b : UInt<20>
        |    input pred : UInt<1>
        |    output c : UInt<32>
        |    c <= mux(pred,a,b)""".stripMargin
    val check = Seq("c <= mux(pred, a, pad(b, 32))")
    executeTest(input, check)
  }

  it should "pad widths of connects" in {
    val input =
      """circuit Top :
        |  module Top :
        |    output a : UInt<32>
        |    input b : UInt<20>
        |    a <= b
        |    """.stripMargin
    val check = Seq("a <= pad(b, 32)")
    executeTest(input, check)
  }

  it should "pad widths of register init expressions" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input clock: Clock
        |    input reset: AsyncReset
        |
        |    reg r: UInt<8>, clock with:
        |      reset => (reset, UInt<1>("h1"))
        |    """.stripMargin
    // PadWidths will call into constant prop directly, thus the literal is widened instead of adding a pad
    val check = Seq("reset => (reset, UInt<8>(\"h1\"))")
    executeTest(input, check)
  }

  private def testOp(op: String, width: Int, resultWidth: Int): Unit = {
    assert(width > 0)
    val input =
      s"""circuit Top :
         |  module Top :
         |    input a : UInt<32>
         |    input b : UInt<$width>
         |    output c : UInt<$resultWidth>
         |    c <= $op(a,b)""".stripMargin
    val check = if (width < 32) {
      Seq(s"c <= $op(a, pad(b, 32))")
    } else if (width == 32) {
      Seq(s"c <= $op(a, b)")
    } else {
      Seq(s"c <= $op(pad(a, $width), b)")
    }
    executeTest(input, check)
  }

  it should "pad widths of the arguments to add and sub" in {
    // add and sub have the same width inference rule: max(w_1, w_2) + 1
    testOp("add", 2, 33)
    testOp("add", 32, 33)
    testOp("add", 35, 36)

    testOp("sub", 2, 33)
    testOp("sub", 32, 33)
    testOp("sub", 35, 36)
  }

  it should "pad widths of the arguments to and, or and xor" in {
    // and, or and xor have the same width inference rule: max(w_1, w_2)
    testOp("and", 2, 32)
    testOp("and", 32, 32)
    testOp("and", 35, 35)

    testOp("or", 2, 32)
    testOp("or", 32, 32)
    testOp("or", 35, 35)

    testOp("xor", 2, 32)
    testOp("xor", 32, 32)
    testOp("xor", 35, 35)
  }

  it should "pad widths of the arguments to lt, leq, gt, geq, eq and neq" in {
    // lt, leq, gt, geq, eq and ne have the same width inference rule: 1
    testOp("lt", 2, 1)
    testOp("lt", 32, 1)
    testOp("lt", 35, 1)

    testOp("leq", 2, 1)
    testOp("leq", 32, 1)
    testOp("leq", 35, 1)

    testOp("gt", 2, 1)
    testOp("gt", 32, 1)
    testOp("gt", 35, 1)

    testOp("geq", 2, 1)
    testOp("geq", 32, 1)
    testOp("geq", 35, 1)

    testOp("eq", 2, 1)
    testOp("eq", 32, 1)
    testOp("eq", 35, 1)

    testOp("neq", 2, 1)
    testOp("neq", 32, 1)
    testOp("neq", 35, 1)
  }

  private val resolvedCompiler = new TransformManager(Forms.Resolved)
  private def checkWidthsAfterPadWidths(input: String, op: String): Unit = {
    val result = compile(input)

    // we serialize the result in order to rerun width inference
    val resultFir = firrtl.Parser.parse(result.circuit.serialize)
    val newWidths = resolvedCompiler.runTransform(CircuitState(resultFir, Seq()))

    // the newly loaded circuit should look the same in serialized form (if this fails, the test has a bug)
    assert(newWidths.circuit.serialize == result.circuit.serialize)

    // we compare the widths produced by PadWidths with the widths that would normally be inferred
    assert(newWidths.circuit.modules.head == result.circuit.modules.head, s"failed with op `$op`")
  }

  it should "always generate valid firrtl" in {
    // an older version of PadWidths would generate ill types firrtl for mul, div, rem and dshl

    def input(op: String): String =
      s"""circuit Top:
         |  module Top:
         |    input a: UInt<3>
         |    input b: UInt<1>
         |    output c: UInt
         |    c <= $op(a, b)
         |""".stripMargin

    def test(op: String): Unit = checkWidthsAfterPadWidths(input(op), op)

    // This was never broken, but we want to make sure that the test works.
    test("add")

    test("mul")
    test("div")
    test("rem")
    test("dshl")
  }

  private def executeTest(input: String, expected: Seq[String]): Unit = {
    val result = compile(input)
    val lines = result.circuit.serialize.split("\n").map(normalized)
    expected.map(normalized).foreach { e =>
      assert(lines.contains(e), f"Failed to find $e in ${lines.mkString("\n")}")
    }
  }
}
