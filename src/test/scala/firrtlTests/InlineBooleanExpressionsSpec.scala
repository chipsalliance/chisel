
// See LICENSE for license details.

package firrtlTests

import firrtl._
import firrtl.annotations.Annotation
import firrtl.options.Dependency
import firrtl.passes._
import firrtl.transforms._
import firrtl.testutils._
import firrtl.stage.TransformManager

class InlineBooleanExpressionsSpec extends FirrtlFlatSpec {
  val transform = new InlineBooleanExpressions
  val transforms: Seq[Transform] = new TransformManager(
      transform.prerequisites
    ).flattenedTransformOrder :+ transform

  protected def exec(input: String, annos: Seq[Annotation] = Nil) = {
    transforms.foldLeft(CircuitState(parse(input), UnknownForm, AnnotationSeq(annos))) {
      (c: CircuitState, t: Transform) => t.runTransform(c)
    }.circuit.serialize
  }

  it should "inline mux operands" in {
    val input =
      """circuit Top :
        |  module Top :
        |    output out : UInt<1>
        |    node x1 = UInt<1>(0)
        |    node x2 = UInt<1>(1)
        |    node _t = head(x1, 1)
        |    node _f = head(x2, 1)
        |    node _c = lt(x1, x2)
        |    node _y = mux(_c, _t, _f)
        |    out <= _y""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    output out : UInt<1>
        |    node x1 = UInt<1>(0)
        |    node x2 = UInt<1>(1)
        |    node _t = head(x1, 1)
        |    node _f = head(x2, 1)
        |    node _c = lt(x1, x2)
        |    node _y = mux(lt(x1, x2), head(x1, 1), head(x2, 1))
        |    out <= mux(lt(x1, x2), head(x1, 1), head(x2, 1))""".stripMargin
    val result = exec(input)
    (result) should be (parse(check).serialize)
    firrtlEquivalenceTest(input, Seq(new InlineBooleanExpressions))
  }

  it should "only inline expressions with the same file and line number" in {
    val input =
      """circuit Top :
        |  module Top :
        |    output outA1 : UInt<1>
        |    output outA2 : UInt<1>
        |    output outB : UInt<1>
        |    node x1 = UInt<1>(0)
        |    node x2 = UInt<1>(1)
        |
        |    node _t = head(x1, 1) @[A 1:1]
        |    node _f = head(x2, 1) @[A 1:2]
        |    node _y = mux(lt(x1, x2), _t, _f) @[A 1:3]
        |    outA1 <= _y @[A 1:3]
        |
        |    outA2 <= _y @[A 2:3]
        |
        |    outB <= _y @[B]""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    output outA1 : UInt<1>
        |    output outA2 : UInt<1>
        |    output outB : UInt<1>
        |    node x1 = UInt<1>(0)
        |    node x2 = UInt<1>(1)
        |
        |    node _t = head(x1, 1) @[A 1:1]
        |    node _f = head(x2, 1) @[A 1:2]
        |    node _y = mux(lt(x1, x2), head(x1, 1), head(x2, 1)) @[A 1:3]
        |    outA1 <= mux(lt(x1, x2), head(x1, 1), head(x2, 1)) @[A 1:3]
        |
        |    outA2 <= _y @[A 2:3]
        |
        |    outB <= _y @[B]""".stripMargin
    val result = exec(input)
    (result) should be (parse(check).serialize)
    firrtlEquivalenceTest(input, Seq(new InlineBooleanExpressions))
  }

  it should "inline boolean DoPrims" in {
    val input =
      """circuit Top :
        |  module Top :
        |    output outA : UInt<1>
        |    output outB : UInt<1>
        |    node x1 = UInt<3>(0)
        |    node x2 = UInt<3>(1)
        |
        |    node _a = lt(x1, x2)
        |    node _b = eq(_a, x2)
        |    node _c = and(_b, x2)
        |    outA <= _c
        |
        |    node _d = head(_c, 1)
        |    node _e = andr(_d)
        |    node _f = lt(_e, x2)
        |    outB <= _f""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    output outA : UInt<1>
        |    output outB : UInt<1>
        |    node x1 = UInt<3>(0)
        |    node x2 = UInt<3>(1)
        |
        |    node _a = lt(x1, x2)
        |    node _b = eq(lt(x1, x2), x2)
        |    node _c = and(eq(lt(x1, x2), x2), x2)
        |    outA <= and(eq(lt(x1, x2), x2), x2)
        |
        |    node _d = head(_c, 1)
        |    node _e = andr(head(_c, 1))
        |    node _f = lt(andr(head(_c, 1)), x2)
        |
        |    outB <= lt(andr(head(_c, 1)), x2)""".stripMargin
    val result = exec(input)
    (result) should be (parse(check).serialize)
    firrtlEquivalenceTest(input, Seq(new InlineBooleanExpressions))
  }

  it should "inline more boolean DoPrims" in {
    val input =
      """circuit Top :
        |  module Top :
        |    output outA : UInt<1>
        |    output outB : UInt<1>
        |    node x1 = UInt<3>(0)
        |    node x2 = UInt<3>(1)
        |
        |    node _a = lt(x1, x2)
        |    node _b = leq(_a, x2)
        |    node _c = gt(_b, x2)
        |    node _d = geq(_c, x2)
        |    outA <= _d
        |
        |    node _e = lt(x1, x2)
        |    node _f = leq(x1, _e)
        |    node _g = gt(x1, _f)
        |    node _h = geq(x1, _g)
        |    outB <= _h""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    output outA : UInt<1>
        |    output outB : UInt<1>
        |    node x1 = UInt<3>(0)
        |    node x2 = UInt<3>(1)
        |
        |    node _a = lt(x1, x2)
        |    node _b = leq(lt(x1, x2), x2)
        |    node _c = gt(leq(lt(x1, x2), x2), x2)
        |    node _d = geq(gt(leq(lt(x1, x2), x2), x2), x2)
        |    outA <= geq(gt(leq(lt(x1, x2), x2), x2), x2)
        |
        |    node _e = lt(x1, x2)
        |    node _f = leq(x1, lt(x1, x2))
        |    node _g = gt(x1, leq(x1, lt(x1, x2)))
        |    node _h = geq(x1, gt(x1, leq(x1, lt(x1, x2))))
        |
        |    outB <= geq(x1, gt(x1, leq(x1, lt(x1, x2))))""".stripMargin
    val result = exec(input)
    (result) should be (parse(check).serialize)
    firrtlEquivalenceTest(input, Seq(new InlineBooleanExpressions))
  }

  it should "limit the number of inlines" in {
    val input =
      s"""circuit Top :
         |  module Top :
         |    input c_0: UInt<1>
         |    input c_1: UInt<1>
         |    input c_2: UInt<1>
         |    input c_3: UInt<1>
         |    input c_4: UInt<1>
         |    input c_5: UInt<1>
         |    input c_6: UInt<1>
         |    output out : UInt<1>
         |
         |    node _1 = or(c_0, c_1)
         |    node _2 = or(_1, c_2)
         |    node _3 = or(_2, c_3)
         |    node _4 = or(_3, c_4)
         |    node _5 = or(_4, c_5)
         |    node _6 = or(_5, c_6)
         |
         |    out <= _6""".stripMargin
    val check =
      s"""circuit Top :
         |  module Top :
         |    input c_0: UInt<1>
         |    input c_1: UInt<1>
         |    input c_2: UInt<1>
         |    input c_3: UInt<1>
         |    input c_4: UInt<1>
         |    input c_5: UInt<1>
         |    input c_6: UInt<1>
         |    output out : UInt<1>
         |
         |    node _1 = or(c_0, c_1)
         |    node _2 = or(or(c_0, c_1), c_2)
         |    node _3 = or(or(or(c_0, c_1), c_2), c_3)
         |    node _4 = or(_3, c_4)
         |    node _5 = or(or(_3, c_4), c_5)
         |    node _6 = or(or(or(_3, c_4), c_5), c_6)
         |
         |    out <= or(or(or(_3, c_4), c_5), c_6)""".stripMargin
    val result = exec(input, Seq(InlineBooleanExpressionsMax(3)))
    (result) should be (parse(check).serialize)
    firrtlEquivalenceTest(input, Seq(new InlineBooleanExpressions))
  }

  it should "be equivalent" in {
    val input =
      """circuit InlineBooleanExpressionsEquivalenceTest :
        |  module InlineBooleanExpressionsEquivalenceTest :
        |    input in : UInt<1>[6]
        |    output out : UInt<1>
        |
        |    node _a = or(in[0], in[1])
        |    node _b = and(in[2], _a)
        |    node _c = eq(in[3], _b)
        |    node _d = lt(in[4], _c)
        |    node _e = eq(in[5], _d)
        |    node _f = head(_e, 1)
        |    out <= _f""".stripMargin
    firrtlEquivalenceTest(input, Seq(new InlineBooleanExpressions))
  }
}
