// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl._
import firrtl.ir.{Circuit, Connect, FileInfo, MultiInfo, Statement}
import firrtl.annotations.{Annotation, ReferenceTarget}
import firrtl.options.Dependency
import firrtl.passes._
import firrtl.transforms._
import firrtl.testutils._
import firrtl.stage.{PrettyNoExprInlining, TransformManager}

class InlineBooleanExpressionsSpec extends FirrtlFlatSpec {
  val transform = new InlineBooleanExpressions
  val transforms: Seq[Transform] = new TransformManager(
    transform.prerequisites
  ).flattenedTransformOrder :+ transform

  protected def exec(input: Circuit, annos: Seq[Annotation] = Nil) = {
    transforms
      .foldLeft(CircuitState(input, UnknownForm, AnnotationSeq(annos))) { (c: CircuitState, t: Transform) =>
        t.runTransform(c)
      }
      .circuit
      .serialize
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
    val result = exec(parse(input))
    (result) should be(parse(check).serialize)
    firrtlEquivalenceTest(input, Seq(new InlineBooleanExpressions))
  }

  it should "not inline dontTouched signals" in {
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
        |    node _y = mux(lt(x1, x2), _t, _f)
        |    out <= mux(lt(x1, x2), _t, _f)""".stripMargin
    val result = exec(
      parse(input),
      Seq(
        DontTouchAnnotation(ReferenceTarget("Top", "Top", Seq.empty, "_t", Seq.empty)),
        DontTouchAnnotation(ReferenceTarget("Top", "Top", Seq.empty, "_f", Seq.empty))
      )
    )
    (result) should be(parse(check).serialize)
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
    val result = exec(parse(input))
    (result) should be(parse(check).serialize)
    firrtlEquivalenceTest(input, Seq(new InlineBooleanExpressions))
  }

  it should "inline if subexpression info is a subset of parent info" in {
    val input =
      parse("""circuit test :
              |  module test :
              |    input in_1 : UInt<1>
              |    input in_2 : UInt<1>
              |    input in_3 : UInt<1>
              |    output out : UInt<1>
              |    node _c = in_1 @[A 1:1]
              |    node _t = in_2 @[A 1:1]
              |    node _f = in_3 @[A 1:1]
              |    out <= mux(_c, _t, _f)""".stripMargin).mapModule { m =>
        // workaround to insert MultiInfo
        def onStmt(stmt: Statement): Statement = stmt match {
          case c: Connect =>
            c.mapInfo { _ =>
              MultiInfo(
                Seq(
                  FileInfo("A 1:1"),
                  FileInfo("A 2:2"),
                  FileInfo("A 3:3")
                )
              )
            }
          case other => other.mapStmt(onStmt)
        }
        m.mapStmt(onStmt)
      }
    val check =
      """circuit test :
        |  module test :
        |    input in_1 : UInt<1>
        |    input in_2 : UInt<1>
        |    input in_3 : UInt<1>
        |    output out : UInt<1>
        |    node _c = in_1 @[A 1:1]
        |    node _t = in_2 @[A 1:1]
        |    node _f = in_3 @[A 1:1]
        |    out <= mux(in_1, in_2, in_3) @[A 1:1 A 2:2 A 3:3]""".stripMargin
    val result = exec(input)
    (result) should be(parse(check).serialize)
  }

  it should "inline mux condition and dshl/dhslr shamt args" in {
    val input =
      """circuit inline_mux_dshl_dshlr_args :
        |  module inline_mux_dshl_dshlr_args :
        |    input in_1 : UInt<3>
        |    input in_2 : UInt<3>
        |    input in_3 : UInt<3>
        |    output out_1 : UInt<3>
        |    output out_2 : UInt<3>
        |    output out_3 : UInt<4>
        |    node _c = head(in_1, 1)
        |    node _t = in_2
        |    node _f = in_3
        |    out_1 <= mux(_c, _t, _f)
        |    out_2 <= dshr(in_1, _c)
        |    out_3 <= dshl(in_1, _c)""".stripMargin
    val check =
      """circuit inline_mux_dshl_dshlr_args :
        |  module inline_mux_dshl_dshlr_args :
        |    input in_1 : UInt<3>
        |    input in_2 : UInt<3>
        |    input in_3 : UInt<3>
        |    output out_1 : UInt<3>
        |    output out_2 : UInt<3>
        |    output out_3 : UInt<4>
        |    node _c = head(in_1, 1)
        |    node _t = in_2
        |    node _f = in_3
        |    out_1 <= mux(head(in_1, 1), _t, _f)
        |    out_2 <= dshr(in_1, head(in_1, 1))
        |    out_3 <= dshl(in_1, head(in_1, 1))""".stripMargin
    val result = exec(parse(input))
    (result) should be(parse(check).serialize)
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
    val result = exec(parse(input))
    (result) should be(parse(check).serialize)
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
    val result = exec(parse(input))
    (result) should be(parse(check).serialize)
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
    val result = exec(parse(input), Seq(InlineBooleanExpressionsMax(3)))
    (result) should be(parse(check).serialize)
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

  it should "emit parentheses in the correct places" in {
    // should fail if any of these sub-expressions does not have parentheses
    val input =
      """
        |circuit TestParentheses :
        |  module TestParentheses :
        |    input in : UInt<1>[3]
        |    output out : UInt<1>[13]
        |
        |    out[0] <= mul(and(in[0], in[1]), in[2])
        |    out[1] <= div(and(in[0], in[1]), in[2])
        |    out[2] <= rem(and(in[0], in[1]), in[2])
        |    out[3] <= add(and(in[0], in[1]), in[2])
        |    out[4] <= sub(and(in[0], in[1]), in[2])
        |    out[5] <= dshl(in[0], and(in[1], in[2]))
        |    out[6] <= dshr(in[0], and(in[1], in[2]))
        |    out[7] <= lt(and(in[0], in[1]), in[2])
        |    out[8] <= gt(in[0], or(in[1], in[2]))
        |    out[9] <= eq(in[0], or(in[1], in[2]))
        |    out[10] <= neq(in[0], or(in[1], in[2]))
        |    out[11] <= and(in[0], xor(in[1], in[2]))
        |    out[12] <= xor(in[0], or(in[1], in[2]))
    """.stripMargin
    firrtlEquivalenceTest(input, Seq(new InlineBooleanExpressions))
  }

  it should "avoid inlining when it would create context-sensitivity bugs" in {
    val input =
      """circuit AddNot:
        |  module AddNot:
        |    input a: UInt<1>
        |    input b: UInt<1>
        |    output o: UInt<2>
        |    o <= add(a, not(b))""".stripMargin
    firrtlEquivalenceTest(input, Seq(new InlineBooleanExpressions))
  }

  // https://github.com/chipsalliance/firrtl/issues/2035
  // This is interesting because other ways of trying to express this get split out by
  // SplitExpressions and don't get inlined again
  // If we were to inline more expressions (ie. not just boolean ones) the issue this represents
  // would come up more often
  it should "handle cvt nested inside of a dshl" in {
    val input =
      """circuit DshlCvt:
        |  module DshlCvt:
        |    input a: UInt<4>
        |    input b: SInt<1>
        |    output o: UInt
        |    o <= dshl(a, asUInt(cvt(b)))""".stripMargin
    firrtlEquivalenceTest(input, Seq(new InlineBooleanExpressions))
  }

  it should s"respect --${PrettyNoExprInlining.longOption}" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input a : UInt<1>
        |    input b : UInt<1>
        |    input c : UInt<1>
        |    output out : UInt<1>
        |
        |    node _T_1 = and(a, b)
        |    out <= and(_T_1, c)""".stripMargin
    val result = exec(parse(input), PrettyNoExprInlining :: Nil)
    (result) should be(parse(input).serialize)
  }
}
