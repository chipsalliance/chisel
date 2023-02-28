// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl._
import firrtl.passes._
import firrtl.transforms._
import firrtl.testutils._
import firrtl.annotations.Annotation
import firrtl.stage.DisableFold

class ConstantPropagationSpec extends FirrtlFlatSpec {
  val transforms: Seq[Transform] =
    Seq(ToWorkingIR, ResolveFlows, new InferWidths, new ConstantPropagation)
  protected def exec(input: String, annos: Seq[Annotation] = Nil) = {
    transforms
      .foldLeft(CircuitState(parse(input), UnknownForm, AnnotationSeq(annos))) { (c: CircuitState, t: Transform) =>
        t.runTransform(c)
      }
      .circuit
      .serialize
  }
}

class ConstantPropagationMultiModule extends ConstantPropagationSpec {
  "ConstProp" should "propagate constant inputs" in {
    val input =
      """circuit Top :
  module Child :
    input in0 : UInt<1>
    input in1 : UInt<1>
    output out : UInt<1>
    out <= and(in0, in1)
  module Top :
    input x : UInt<1>
    output z : UInt<1>
    inst c of Child
    c.in0 <= x
    c.in1 <= UInt<1>(1)
    z <= c.out
"""
    val check =
      """circuit Top :
  module Child :
    input in0 : UInt<1>
    input in1 : UInt<1>
    output out : UInt<1>
    out <= in0
  module Top :
    input x : UInt<1>
    output z : UInt<1>
    inst c of Child
    c.in0 <= x
    c.in1 <= UInt<1>(1)
    z <= c.out
"""
    (parse(exec(input))) should be(parse(check))
  }

  "ConstProp" should "propagate constant inputs ONLY if ALL instance inputs get the same value" in {
    def circuit(allSame: Boolean) =
      s"""circuit Top :
  module Bottom :
    input in : UInt<1>
    output out : UInt<1>
    out <= in
  module Child :
    output out : UInt<1>
    inst b0 of Bottom
    b0.in <= UInt(1)
    out <= b0.out
  module Top :
    input x : UInt<1>
    output z : UInt<1>

    inst c of Child

    inst b0 of Bottom
    b0.in <= ${if (allSame) "UInt(1)" else "x"}
    inst b1 of Bottom
    b1.in <= UInt(1)

    z <= and(and(b0.out, b1.out), c.out)
"""
    val resultFromAllSame =
      """circuit Top :
  module Bottom :
    input in : UInt<1>
    output out : UInt<1>
    out <= UInt(1)
  module Child :
    output out : UInt<1>
    inst b0 of Bottom
    b0.in <= UInt(1)
    out <= UInt(1)
  module Top :
    input x : UInt<1>
    output z : UInt<1>
    inst c of Child
    inst b0 of Bottom
    b0.in <= UInt(1)
    inst b1 of Bottom
    b1.in <= UInt(1)
    z <= UInt(1)
"""
    (parse(exec(circuit(false)))) should be(parse(circuit(false)))
    (parse(exec(circuit(true)))) should be(parse(resultFromAllSame))
  }

  // =============================
  "ConstProp" should "do nothing on unrelated modules" in {
    val input =
      """circuit foo :
  module foo :
    input dummy : UInt<1>
    skip

  module bar :
    input dummy : UInt<1>
    skip
"""
    val check = input
    (parse(exec(input))) should be(parse(check))
  }

  // =============================
  "ConstProp" should "propagate module chains not connected to the top" in {
    val input =
      """circuit foo :
  module foo :
    input dummy : UInt<1>
    skip

  module bar1 :
    output out : UInt<1>
    inst one of baz1
    inst zero of baz0
    out <= or(one.test, zero.test)

  module bar0 :
    output out : UInt<1>
    inst one of baz1
    inst zero of baz0
    out <= and(one.test, zero.test)

  module baz1 :
    output test : UInt<1>
    test <= UInt<1>(1)
  module baz0 :
    output test : UInt<1>
    test <= UInt<1>(0)
"""
    val check =
      """circuit foo :
  module foo :
    input dummy : UInt<1>
    skip

  module bar1 :
    output out : UInt<1>
    inst one of baz1
    inst zero of baz0
    out <= UInt<1>(1)

  module bar0 :
    output out : UInt<1>
    inst one of baz1
    inst zero of baz0
    out <= UInt<1>(0)

  module baz1 :
    output test : UInt<1>
    test <= UInt<1>(1)
  module baz0 :
    output test : UInt<1>
    test <= UInt<1>(0)
"""
    (parse(exec(input))) should be(parse(check))
  }
}

// Tests the following cases for constant propagation:
//   1) Unsigned integers are always greater than or
//        equal to zero
//   2) Values are always smaller than a number greater
//        than their maximum value
//   3) Values are always greater than a number smaller
//        than their minimum value
class ConstantPropagationSingleModule extends ConstantPropagationSpec {
  // =============================
  "The rule x >= 0 " should " always be true if x is a UInt" in {
    val input =
      """circuit Top :
  module Top :
    input x : UInt<5>
    output y : UInt<1>
    y <= geq(x, UInt(0))
"""
    val check =
      """circuit Top :
  module Top :
    input x : UInt<5>
    output y : UInt<1>
    y <= UInt<1>("h1")
"""
    (parse(exec(input))) should be(parse(check))
  }

  // =============================
  "The rule x < 0 " should " never be true if x is a UInt" in {
    val input =
      """circuit Top :
  module Top :
    input x : UInt<5>
    output y : UInt<1>
    y <= lt(x, UInt(0))
"""
    val check =
      """circuit Top :
  module Top :
    input x : UInt<5>
    output y : UInt<1>
    y <= UInt<1>(0)
"""
    (parse(exec(input))) should be(parse(check))
  }

  // =============================
  "The rule 0 <= x " should " always be true if x is a UInt" in {
    val input =
      """circuit Top :
  module Top :
    input x : UInt<5>
    output y : UInt<1>
    y <= leq(UInt(0),x)
"""
    val check =
      """circuit Top :
  module Top :
    input x : UInt<5>
    output y : UInt<1>
    y <= UInt<1>(1)
"""
    (parse(exec(input))) should be(parse(check))
  }

  // =============================
  "The rule 0 > x " should " never be true if x is a UInt" in {
    val input =
      """circuit Top :
  module Top :
    input x : UInt<5>
    output y : UInt<1>
    y <= gt(UInt(0),x)
"""
    val check =
      """circuit Top :
  module Top :
    input x : UInt<5>
    output y : UInt<1>
    y <= UInt<1>(0)
"""
    (parse(exec(input))) should be(parse(check))
  }

  // =============================
  "The rule 1 < 3 " should " always be true" in {
    val input =
      """circuit Top :
  module Top :
    input x : UInt<5>
    output y : UInt<1>
    y <= lt(UInt(0),UInt(3))
"""
    val check =
      """circuit Top :
  module Top :
    input x : UInt<5>
    output y : UInt<1>
    y <= UInt<1>(1)
"""
    (parse(exec(input))) should be(parse(check))
  }

  // =============================
  "The rule x < 8 " should " always be true if x only has 3 bits" in {
    val input =
      """circuit Top :
  module Top :
    input x : UInt<3>
    output y : UInt<1>
    y <= lt(x,UInt(8))
"""
    val check =
      """circuit Top :
  module Top :
    input x : UInt<3>
    output y : UInt<1>
    y <= UInt<1>(1)
"""
    (parse(exec(input))) should be(parse(check))
  }

  // =============================
  "The rule x <= 7 " should " always be true if x only has 3 bits" in {
    val input =
      """circuit Top :
  module Top :
    input x : UInt<3>
    output y : UInt<1>
    y <= leq(x,UInt(7))
"""
    val check =
      """circuit Top :
  module Top :
    input x : UInt<3>
    output y : UInt<1>
    y <= UInt<1>(1)
"""
    (parse(exec(input))) should be(parse(check))
  }

  // =============================
  "The rule 8 > x" should " always be true if x only has 3 bits" in {
    val input =
      """circuit Top :
  module Top :
    input x : UInt<3>
    output y : UInt<1>
    y <= gt(UInt(8),x)
"""
    val check =
      """circuit Top :
  module Top :
    input x : UInt<3>
    output y : UInt<1>
    y <= UInt<1>(1)
"""
    (parse(exec(input))) should be(parse(check))
  }

  // =============================
  "The rule 7 >= x" should " always be true if x only has 3 bits" in {
    val input =
      """circuit Top :
  module Top :
    input x : UInt<3>
    output y : UInt<1>
    y <= geq(UInt(7),x)
"""
    val check =
      """circuit Top :
  module Top :
    input x : UInt<3>
    output y : UInt<1>
    y <= UInt<1>(1)
"""
    (parse(exec(input))) should be(parse(check))
  }

  // =============================
  "The rule 10 == 10" should " always be true" in {
    val input =
      """circuit Top :
  module Top :
    input x : UInt<3>
    output y : UInt<1>
    y <= eq(UInt(10),UInt(10))
"""
    val check =
      """circuit Top :
  module Top :
    input x : UInt<3>
    output y : UInt<1>
    y <= UInt<1>(1)
"""
    (parse(exec(input))) should be(parse(check))
  }

  // =============================
  "The rule x == z " should " not be true even if they have the same number of bits" in {
    val input =
      """circuit Top :
  module Top :
    input x : UInt<3>
    input z : UInt<3>
    output y : UInt<1>
    y <= eq(x,z)
"""
    val check =
      """circuit Top :
  module Top :
    input x : UInt<3>
    input z : UInt<3>
    output y : UInt<1>
    y <= eq(x,z)
"""
    (parse(exec(input))) should be(parse(check))
  }

  // =============================
  "The rule 10 != 10 " should " always be false" in {
    val input =
      """circuit Top :
  module Top :
    output y : UInt<1>
    y <= neq(UInt(10),UInt(10))
"""
    val check =
      """circuit Top :
  module Top :
    output y : UInt<1>
    y <= UInt(0)
"""
    (parse(exec(input))) should be(parse(check))
  }
  // =============================
  "The rule 1 >= 3 " should " always be false" in {
    val input =
      """circuit Top :
  module Top :
    input x : UInt<5>
    output y : UInt<1>
    y <= geq(UInt(1),UInt(3))
"""
    val check =
      """circuit Top :
  module Top :
    input x : UInt<5>
    output y : UInt<1>
    y <= UInt<1>(0)
"""
    (parse(exec(input))) should be(parse(check))
  }

  // =============================
  "The rule x >= 8 " should " never be true if x only has 3 bits" in {
    val input =
      """circuit Top :
  module Top :
    input x : UInt<3>
    output y : UInt<1>
    y <= geq(x,UInt(8))
"""
    val check =
      """circuit Top :
  module Top :
    input x : UInt<3>
    output y : UInt<1>
    y <= UInt<1>(0)
"""
    (parse(exec(input))) should be(parse(check))
  }

  // =============================
  "The rule x > 7 " should " never be true if x only has 3 bits" in {
    val input =
      """circuit Top :
  module Top :
    input x : UInt<3>
    output y : UInt<1>
    y <= gt(x,UInt(7))
"""
    val check =
      """circuit Top :
  module Top :
    input x : UInt<3>
    output y : UInt<1>
    y <= UInt<1>(0)
"""
    (parse(exec(input))) should be(parse(check))
  }

  // =============================
  "The rule 8 <= x" should " never be true if x only has 3 bits" in {
    val input =
      """circuit Top :
  module Top :
    input x : UInt<3>
    output y : UInt<1>
    y <= leq(UInt(8),x)
"""
    val check =
      """circuit Top :
  module Top :
    input x : UInt<3>
    output y : UInt<1>
    y <= UInt<1>(0)
"""
    (parse(exec(input))) should be(parse(check))
  }

  // =============================
  "The rule 7 < x" should " never be true if x only has 3 bits" in {
    val input =
      """circuit Top :
  module Top :
    input x : UInt<3>
    output y : UInt<1>
    y <= lt(UInt(7),x)
"""
    val check =
      """circuit Top :
  module Top :
    input x : UInt<3>
    output y : UInt<1>
    y <= UInt<1>(0)
"""
    (parse(exec(input))) should be(parse(check))
  }

  // =============================
  "ConstProp" should "work across wires" in {
    val input =
      """circuit Top :
  module Top :
    input x : UInt<1>
    output y : UInt<1>
    wire z : UInt<1>
    y <= z
    z <= mux(x, UInt<1>(0), UInt<1>(0))
"""
    val check =
      """circuit Top :
  module Top :
    input x : UInt<1>
    output y : UInt<1>
    wire z : UInt<1>
    y <= UInt<1>(0)
    z <= UInt<1>(0)
"""
    (parse(exec(input))) should be(parse(check))
  }

  // =============================
  "ConstProp" should "swap named nodes with temporary nodes that drive them" in {
    val input =
      """circuit Top :
  module Top :
    input x : UInt<1>
    input y : UInt<1>
    output z : UInt<1>
    node _T_1 = and(x, y)
    node n = _T_1
    z <= and(n, x)
"""
    val check =
      """circuit Top :
  module Top :
    input x : UInt<1>
    input y : UInt<1>
    output z : UInt<1>
    node n = and(x, y)
    node _T_1 = n
    z <= and(n, x)
"""
    (parse(exec(input))) should be(parse(check))
  }

  // =============================
  "ConstProp" should "swap named nodes with temporary wires that drive them" in {
    val input =
      """circuit Top :
  module Top :
    input x : UInt<1>
    input y : UInt<1>
    output z : UInt<1>
    wire _T_1 : UInt<1>
    node n = _T_1
    z <= n
    _T_1 <= and(x, y)
"""
    val check =
      """circuit Top :
  module Top :
    input x : UInt<1>
    input y : UInt<1>
    output z : UInt<1>
    wire n : UInt<1>
    node _T_1 = n
    z <= n
    n <= and(x, y)
"""
    (parse(exec(input))) should be(parse(check))
  }

  // =============================
  "ConstProp" should "swap named nodes with temporary registers that drive them" in {
    val input =
      """circuit Top :
  module Top :
    input clock : Clock
    input x : UInt<1>
    output z : UInt<1>
    reg _T_1 : UInt<1>, clock with : (reset => (UInt<1>(0), _T_1))
    node n = _T_1
    z <= n
    _T_1 <= x
"""
    val check =
      """circuit Top :
  module Top :
    input clock : Clock
    input x : UInt<1>
    output z : UInt<1>
    reg n : UInt<1>, clock with : (reset => (UInt<1>(0), n))
    node _T_1 = n
    z <= n
    n <= x
"""
    (parse(exec(input))) should be(parse(check))
  }

  // =============================
  "ConstProp" should "only swap a given name with one other name" in {
    val input =
      """circuit Top :
  module Top :
    input x : UInt<1>
    input y : UInt<1>
    output z : UInt<3>
    node _T_1 = add(x, y)
    node n = _T_1
    node m = _T_1
    z <= add(n, m)
"""
    val check =
      """circuit Top :
  module Top :
    input x : UInt<1>
    input y : UInt<1>
    output z : UInt<3>
    node n = add(x, y)
    node _T_1 = n
    node m = n
    z <= add(n, n)
"""
    (parse(exec(input))) should be(parse(check))
  }

  "ConstProp" should "NOT swap wire names with node names" in {
    val input =
      """circuit Top :
  module Top :
    input clock : Clock
    input x : UInt<1>
    input y : UInt<1>
    output z : UInt<1>
    wire hit : UInt<1>
    node _T_1 = or(x, y)
    node _T_2 = eq(_T_1, UInt<1>(1))
    hit <= _T_2
    z <= hit
"""
    val check =
      """circuit Top :
  module Top :
    input clock : Clock
    input x : UInt<1>
    input y : UInt<1>
    output z : UInt<1>
    wire hit : UInt<1>
    node _T_1 = or(x, y)
    node _T_2 = _T_1
    hit <= or(x, y)
    z <= hit
"""
    (parse(exec(input))) should be(parse(check))
  }

  "ConstProp" should "propagate constant outputs" in {
    val input =
      """circuit Top :
  module Child :
    output out : UInt<1>
    out <= UInt<1>(0)
  module Top :
    input x : UInt<1>
    output z : UInt<1>
    inst c of Child
    z <= and(x, c.out)
"""
    val check =
      """circuit Top :
  module Child :
    output out : UInt<1>
    out <= UInt<1>(0)
  module Top :
    input x : UInt<1>
    output z : UInt<1>
    inst c of Child
    z <= UInt<1>(0)
"""
    (parse(exec(input))) should be(parse(check))
  }

  "ConstProp" should "propagate constant addition" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input x : UInt<5>
        |    output z : UInt<5>
        |    node _T_1 = add(UInt<5>("h0"), UInt<5>("h1"))
        |    node _T_2 = add(_T_1, UInt<5>("h2"))
        |    z <= add(x, _T_2)
      """.stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input x : UInt<5>
        |    output z : UInt<5>
        |    node _T_1 = UInt<6>("h1")
        |    node _T_2 = UInt<7>("h3")
        |    z <= add(x, UInt<7>("h3"))
      """.stripMargin
    (parse(exec(input))) should be(parse(check))
  }

  "ConstProp" should "propagate addition with zero" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input x : UInt<5>
        |    output z : UInt<5>
        |    z <= add(x, UInt<5>("h0"))
      """.stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input x : UInt<5>
        |    output z : UInt<5>
        |    z <= pad(x, 6)
      """.stripMargin
    (parse(exec(input))) should be(parse(check))
  }

  // Optimizing this mux gives: z <= pad(UInt<2>(0), 4)
  // Thus this checks that we then optimize that pad
  "ConstProp" should "optimize nested Expressions" in {
    val input =
      """circuit Top :
        |  module Top :
        |    output z : UInt<4>
        |    z <= mux(UInt(1), UInt<2>(0), UInt<4>(0))
      """.stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    output z : UInt<4>
        |    z <= UInt<4>("h0")
      """.stripMargin
    (parse(exec(input))) should be(parse(check))
  }

  "ConstProp" should "NOT touch self-inits" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input clk : Clock
        |    input rst : UInt<1>
        |    output z : UInt<4>
        |    reg selfinit : UInt<1>, clk with : (reset => (UInt<1>(0), selfinit))
        |    selfinit <= UInt<1>(0)
        |    z <= mux(UInt(1), UInt<2>(0), UInt<4>(0))
     """.stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clk : Clock
        |    input rst : UInt<1>
        |    output z : UInt<4>
        |    reg selfinit : UInt<1>, clk with : (reset => (UInt<1>(0), selfinit))
        |    selfinit <= UInt<1>(0)
        |    z <= UInt<4>(0)
     """.stripMargin
    (parse(exec(input, Seq(NoDCEAnnotation)))) should be(parse(check))
  }

  def castCheck(tpe: String, cast: String): Unit = {
    val input =
      s"""circuit Top :
         |  module Top :
         |    input  x : $tpe
         |    output z : $tpe
         |    z <= $cast(x)
      """.stripMargin
    val check =
      s"""circuit Top :
         |  module Top :
         |    input  x : $tpe
         |    output z : $tpe
         |    z <= x
      """.stripMargin
    (parse(exec(input)).serialize) should be(parse(check).serialize)
  }
  it should "optimize unnecessary casts" in {
    castCheck("UInt<4>", "asUInt")
    castCheck("SInt<4>", "asSInt")
    castCheck("Clock", "asClock")
    castCheck("AsyncReset", "asAsyncReset")
  }

  /* */
  "The rule a / a -> 1" should "be ignored if division folds are disabled" in {
    val input =
      """circuit foo:
        |  module foo:
        |    input a: UInt<8>
        |    output b: UInt<8>
        |    b <= div(a, a)""".stripMargin
    (parse(exec(input, Seq(DisableFold(PrimOps.Div))))) should be(parse(input))
  }
}

// More sophisticated tests of the full compiler
class ConstantPropagationIntegrationSpec extends LowTransformSpec {
  def transform = new LowFirrtlOptimization

  "ConstProp" should "NOT optimize across dontTouch on nodes" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input x : UInt<1>
        |    output y : UInt<1>
        |    node z = x
        |    y <= z""".stripMargin
    val check = input
    execute(input, check, Seq(dontTouch("Top.z")))
  }

  it should "NOT optimize across nodes marked dontTouch by other annotations" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input x : UInt<1>
        |    output y : UInt<1>
        |    node z = x
        |    y <= z""".stripMargin
    val check = input
    val dontTouchRT = annotations.ModuleTarget("Top", "Top").ref("z")
    execute(input, check, Seq(AnnotationWithDontTouches(dontTouchRT)))
  }

  it should "NOT optimize across dontTouch on registers" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input clk : Clock
        |    input reset : UInt<1>
        |    output y : UInt<1>
        |    reg z : UInt<1>, clk
        |    y <= z
        |    z <= mux(reset, UInt<1>("h0"), z)""".stripMargin
    val check = input
    execute(input, check, Seq(dontTouch("Top.z")))
  }

  it should "NOT optimize across dontTouch on wires" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input x : UInt<1>
        |    output y : UInt<1>
        |    wire z : UInt<1>
        |    y <= z
        |    z <= x""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input x : UInt<1>
        |    output y : UInt<1>
        |    node z = x
        |    y <= z""".stripMargin
    execute(input, check, Seq(dontTouch("Top.z")))
  }

  it should "NOT optimize across dontTouch on output ports" in {
    val input =
      """circuit Top :
        |  module Child :
        |    output out : UInt<1>
        |    out <= UInt<1>(0)
        |  module Top :
        |    input x : UInt<1>
        |    output z : UInt<1>
        |    inst c of Child
        |    z <= and(x, c.out)""".stripMargin
    val check = input
    execute(input, check, Seq(dontTouch("Child.out")))
  }

  it should "NOT optimize across dontTouch on input ports" in {
    val input =
      """circuit Top :
        |  module Child :
        |    input in0 : UInt<1>
        |    input in1 : UInt<1>
        |    output out : UInt<1>
        |    out <= and(in0, in1)
        |  module Top :
        |    input x : UInt<1>
        |    output z : UInt<1>
        |    inst c of Child
        |    z <= c.out
        |    c.in0 <= x
        |    c.in1 <= UInt<1>(1)""".stripMargin
    val check = input
    execute(input, check, Seq(dontTouch("Child.in1")))
  }

  it should "NOT optimize if no-constant-propagation is enabled" in {
    val input =
      """circuit Foo:
        |  module Foo:
        |    input a: UInt<1>
        |    output b: UInt<1>
        |    b <= and(UInt<1>(0), a)""".stripMargin
    val check = parse(input).serialize
    execute(input, check, Seq(NoConstantPropagationAnnotation))
  }

  it should "still propagate constants even when there is name swapping" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input x : UInt<1>
        |    input y : UInt<1>
        |    output z : UInt<1>
        |    node _T_1 = and(and(x, y), UInt<1>(0))
        |    node n = _T_1
        |    z <= n""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input x : UInt<1>
        |    input y : UInt<1>
        |    output z : UInt<1>
        |    z <= UInt<1>(0)""".stripMargin
    execute(input, check, Seq.empty)
  }

  it should "pad constant connections to wires when propagating" in {
    val input =
      """circuit Top :
        |  module Top :
        |    output z : UInt<16>
        |    wire w : { a : UInt<8>, b : UInt<8> }
        |    w.a <= UInt<2>("h3")
        |    w.b <= UInt<2>("h3")
        |    z <= cat(w.a, w.b)""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    output z : UInt<16>
        |    z <= UInt<16>("h303")""".stripMargin
    execute(input, check, Seq.empty)
  }

  it should "pad constant connections to registers when propagating" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    output z : UInt<16>
        |    reg r : { a : UInt<8>, b : UInt<8> }, clock
        |    r.a <= UInt<2>("h3")
        |    r.b <= UInt<2>("h3")
        |    z <= cat(r.a, r.b)""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    output z : UInt<16>
        |    z <= UInt<16>("h303")""".stripMargin
    execute(input, check, Seq.empty)
  }

  it should "pad zero when constant propping a register replaced with zero" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    output z : UInt<16>
        |    reg r : UInt<8>, clock
        |    r <= or(r, UInt(0))
        |    node n = UInt("hab")
        |    z <= cat(n, r)""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    output z : UInt<16>
        |    z <= UInt<16>("hab00")""".stripMargin
    execute(input, check, Seq.empty)
  }

  it should "pad constant connections to outputs when propagating" in {
    val input =
      """circuit Top :
        |  module Child :
        |    output x : UInt<8>
        |    x <= UInt<2>("h3")
        |  module Top :
        |    output z : UInt<16>
        |    inst c of Child
        |    z <= cat(UInt<2>("h3"), c.x)""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    output z : UInt<16>
        |    z <= UInt<16>("h303")""".stripMargin
    execute(input, check, Seq.empty)
  }

  it should "pad constant connections to submodule inputs when propagating" in {
    val input =
      """circuit Top :
        |  module Child :
        |    input x : UInt<8>
        |    output y : UInt<16>
        |    y <= cat(UInt<2>("h3"), x)
        |  module Top :
        |    output z : UInt<16>
        |    inst c of Child
        |    c.x <= UInt<2>("h3")
        |    z <= c.y""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    output z : UInt<16>
        |    z <= UInt<16>("h303")""".stripMargin
    execute(input, check, Seq.empty)
  }

  it should "remove pads if the width is <= the width of the argument" in {
    def input(w: Int) =
      s"""circuit Top :
         |  module Top :
         |    input x : UInt<8>
         |    output z : UInt<8>
         |    z <= pad(x, $w)""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input x : UInt<8>
        |    output z : UInt<8>
        |    z <= x""".stripMargin
    execute(input(6), check, Seq.empty)
    execute(input(8), check, Seq.empty)
  }

  "Registers with no reset or connections" should "be replaced with constant zero" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    output z : UInt<8>
        |    reg r : UInt<8>, clock
        |    z <= r""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    output z : UInt<8>
        |    z <= UInt<8>(0)""".stripMargin
    execute(input, check, Seq.empty)
  }

  "Registers with ONLY constant reset" should "be replaced with that constant" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    input reset : UInt<1>
        |    output z : UInt<8>
        |    reg r : UInt<8>, clock with : (reset => (reset, UInt<4>("hb")))
        |    z <= r""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    input reset : UInt<1>
        |    output z : UInt<8>
        |    z <= UInt<8>("hb")""".stripMargin
    execute(input, check, Seq.empty)
  }

  "Registers async reset and a constant connection" should "NOT be removed" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    input reset : AsyncReset
        |    input en : UInt<1>
        |    output z : UInt<8>
        |    reg r : UInt<8>, clock with : (reset => (reset, UInt<4>("hb")))
        |    when en :
        |      r <= UInt<4>("h0")
        |    z <= r""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    input reset : AsyncReset
        |    input en : UInt<1>
        |    output z : UInt<8>
        |    reg r : UInt<8>, clock with :
        |      reset => (reset, UInt<8>("hb"))
        |    z <= r
        |    r <= mux(en, UInt<8>("h0"), r)""".stripMargin
    execute(input, check, Seq.empty)
  }

  "Registers with constant reset and connection to the same constant" should "be replaced with that constant" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    input reset : UInt<1>
        |    input cond : UInt<1>
        |    output z : UInt<8>
        |    reg r : UInt<8>, clock with : (reset => (reset, UInt<4>("hb")))
        |    when cond :
        |      r <= UInt<4>("hb")
        |    z <= r""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    input reset : UInt<1>
        |    input cond : UInt<1>
        |    output z : UInt<8>
        |    z <= UInt<8>("hb")""".stripMargin
    execute(input, check, Seq.empty)
  }

  "Const prop of registers" should "do limited speculative expansion of optimized muxes to absorb bigger cones" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    input en : UInt<1>
        |    output out : UInt<1>
        |    reg r1 : UInt<1>, clock
        |    reg r2 : UInt<1>, clock
        |    when en :
        |      r1 <= UInt<1>(1)
        |    r2 <= UInt<1>(0)
        |    when en :
        |      r2 <= r2
        |    out <= xor(r1, r2)""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    input en : UInt<1>
        |    output out : UInt<1>
        |    out <= UInt<1>("h1")""".stripMargin
    execute(input, check, Seq.empty)
  }

  "A register with constant reset and all connection to either itself or the same constant" should "be replaced with that constant" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    input reset : UInt<1>
        |    input cmd : UInt<3>
        |    output z : UInt<8>
        |    reg r : UInt<8>, clock with : (reset => (reset, UInt<4>("h7")))
        |    r <= r
        |    when eq(cmd, UInt<3>("h0")) :
        |      r <= UInt<3>("h7")
        |    else :
        |      when eq(cmd, UInt<3>("h1")) :
        |        r <= r
        |      else :
        |        when eq(cmd, UInt<3>("h2")) :
        |          r <= UInt<4>("h7")
        |        else :
        |          r <= r
        |    z <= r""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    input reset : UInt<1>
        |    input cmd : UInt<3>
        |    output z : UInt<8>
        |    z <= UInt<8>("h7")""".stripMargin
    execute(input, check, Seq.empty)
  }

  "Registers with ONLY constant connection" should "be replaced with that constant" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    input reset : UInt<1>
        |    output z : SInt<8>
        |    reg r : SInt<8>, clock
        |    r <= SInt<4>(-5)
        |    z <= r""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    input reset : UInt<1>
        |    output z : SInt<8>
        |    z <= SInt<8>(-5)""".stripMargin
    execute(input, check, Seq.empty)
  }

  "Registers with identical constant reset and connection" should "be replaced with that constant" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    input reset : UInt<1>
        |    output z : UInt<8>
        |    reg r : UInt<8>, clock with : (reset => (reset, UInt<4>("hb")))
        |    r <= UInt<4>("hb")
        |    z <= r""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    input reset : UInt<1>
        |    output z : UInt<8>
        |    z <= UInt<8>("hb")""".stripMargin
    execute(input, check, Seq.empty)
  }

  "Connections to a node reference" should "be replaced with the rhs of that node" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input a : UInt<8>
        |    input b : UInt<8>
        |    input c : UInt<1>
        |    output z : UInt<8>
        |    node x = mux(c, a, b)
        |    z <= x""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input a : UInt<8>
        |    input b : UInt<8>
        |    input c : UInt<1>
        |    output z : UInt<8>
        |    z <= mux(c, a, b)""".stripMargin
    execute(input, check, Seq.empty)
  }

  "Registers connected only to themselves" should "be replaced with zero" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    output a : UInt<8>
        |    reg ra : UInt<8>, clock
        |    ra <= ra
        |    a <= ra
        |""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    output a : UInt<8>
        |    a <= UInt<8>(0)
        |""".stripMargin
    execute(input, check, Seq.empty)
  }

  "Registers connected only to themselves from constant propagation" should "be replaced with zero" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    output a : UInt<8>
        |    reg ra : UInt<8>, clock
        |    ra <= or(ra, UInt(0))
        |    a <= ra
        |""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input clock : Clock
        |    output a : UInt<8>
        |    a <= UInt<8>(0)
        |""".stripMargin
    execute(input, check, Seq.empty)
  }

  "Temporary named port" should "not be declared as a node" in {
    val input =
      """circuit Top :
        |  module Top :
        |    input _T_61 : UInt<1>
        |    output z : UInt<1>
        |    node a = _T_61
        |    z <= a""".stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    input _T_61 : UInt<1>
        |    output z : UInt<1>
        |    z <= _T_61""".stripMargin
    execute(input, check, Seq.empty)
  }

  behavior.of("ConstProp")

  it should "optimize shl of constants" in {
    val input =
      """circuit Top :
        |  module Top :
        |    output z : UInt<7>
        |    z <= shl(UInt(5), 4)
      """.stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    output z : UInt<7>
        |    z <= UInt<7>("h50")
      """.stripMargin
    execute(input, check, Seq.empty)
  }

  it should "optimize shr of constants" in {
    val input =
      """circuit Top :
        |  module Top :
        |    output z : UInt<1>
        |    z <= shr(UInt(5), 2)
      """.stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    output z : UInt<1>
        |    z <= UInt<1>("h1")
      """.stripMargin
    execute(input, check, Seq.empty)
  }

  // Due to #866, we need dshl optimized away or it'll become a dshlw and error in parsing
  // Include cat to verify width is correct
  it should "optimize dshl of constant" in {
    val input =
      """circuit Top :
        |  module Top :
        |    output z : UInt<8>
        |    node n = dshl(UInt<1>(0), UInt<2>(0))
        |    z <= cat(UInt<4>("hf"), n)
      """.stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    output z : UInt<8>
        |    z <= UInt<8>("hf0")
      """.stripMargin
    execute(input, check, Seq.empty)
  }

  // Include cat and constants to verify width is correct
  it should "optimize dshr of constant" in {
    val input =
      """circuit Top :
        |  module Top :
        |    output z : UInt<8>
        |    node n = dshr(UInt<4>(0), UInt<2>(2))
        |    z <= cat(UInt<4>("hf"), n)
      """.stripMargin
    val check =
      """circuit Top :
        |  module Top :
        |    output z : UInt<8>
        |    z <= UInt<8>("hf0")
      """.stripMargin
    execute(input, check, Seq.empty)
  }

  private def matchingArgs(op: String, iType: String, oType: String, result: String): Unit = {
    val input =
      s"""circuit Top :
         |  module Top :
         |    input i : ${iType}
         |    output o : ${oType}
         |    o <= ${op}(i, i)
      """.stripMargin
    val check =
      s"""circuit Top :
         |  module Top :
         |    input i : ${iType}
         |    output o : ${oType}
         |    o <= ${result}
      """.stripMargin
    execute(input, check, Seq.empty)
  }

  it should "optimize some binary operations when arguments match" in {
    // Signedness matters
    matchingArgs("sub", "UInt<8>", "UInt<8>", """ UInt<8>("h0")  """)
    matchingArgs("sub", "SInt<8>", "SInt<8>", """ SInt<8>("h0")  """)
    matchingArgs("div", "UInt<8>", "UInt<8>", """ UInt<8>("h1")  """)
    matchingArgs("div", "SInt<8>", "SInt<8>", """ SInt<8>("h1")  """)
    matchingArgs("rem", "UInt<8>", "UInt<8>", """ UInt<8>("h0")  """)
    matchingArgs("rem", "SInt<8>", "SInt<8>", """ SInt<8>("h0")  """)
    matchingArgs("and", "UInt<8>", "UInt<8>", """ i              """)
    matchingArgs("and", "SInt<8>", "UInt<8>", """ asUInt(i)      """)
    // Signedness doesn't matter
    matchingArgs("or", "UInt<8>", "UInt<8>", """ i """)
    matchingArgs("or", "SInt<8>", "UInt<8>", """ asUInt(i) """)
    matchingArgs("xor", "UInt<8>", "UInt<8>", """ UInt<8>("h0")  """)
    matchingArgs("xor", "SInt<8>", "UInt<8>", """ UInt<8>("h0")  """)
    // Always true
    matchingArgs("eq", "UInt<8>", "UInt<1>", """ UInt<1>("h1")  """)
    matchingArgs("leq", "UInt<8>", "UInt<1>", """ UInt<1>("h1")  """)
    matchingArgs("geq", "UInt<8>", "UInt<1>", """ UInt<1>("h1")  """)
    // Never true
    matchingArgs("neq", "UInt<8>", "UInt<1>", """ UInt<1>("h0")  """)
    matchingArgs("lt", "UInt<8>", "UInt<1>", """ UInt<1>("h0")  """)
    matchingArgs("gt", "UInt<8>", "UInt<1>", """ UInt<1>("h0")  """)
  }

  behavior.of("Reduction operators")

  it should "optimize andr of a literal" in {
    val input =
      s"""|circuit Foo:
          |  module Foo:
          |    output _4b0: UInt<1>
          |    output _4b15: UInt<1>
          |    output _4b7: UInt<1>
          |    output _4b1: UInt<1>
          |    output _0b0: UInt<1>
          |    _4b0 <= andr(UInt<4>(0))
          |    _4b15 <= andr(UInt<4>(15))
          |    _4b7 <= andr(UInt<4>(7))
          |    _4b1 <= andr(UInt<4>(1))
          |    wire _0bI: UInt<0>
          |    _0bI is invalid
          |    _0b0 <= andr(_0bI)
          |""".stripMargin
    val check =
      s"""|circuit Foo:
          |  module Foo:
          |    output _4b0: UInt<1>
          |    output _4b15: UInt<1>
          |    output _4b7: UInt<1>
          |    output _4b1: UInt<1>
          |    output _0b0: UInt<1>
          |    _4b0 <= UInt<1>(0)
          |    _4b15 <= UInt<1>(1)
          |    _4b7 <= UInt<1>(0)
          |    _4b1 <= UInt<1>(0)
          |    _0b0 <= UInt<1>(1)
          |""".stripMargin
    execute(input, check, Seq.empty)
  }

  it should "optimize orr of a literal" in {
    val input =
      s"""|circuit Foo:
          |  module Foo:
          |    output _4b0: UInt<1>
          |    output _4b15: UInt<1>
          |    output _4b7: UInt<1>
          |    output _4b1: UInt<1>
          |    output _0b0: UInt<1>
          |    _4b0 <= orr(UInt<4>(0))
          |    _4b15 <= orr(UInt<4>(15))
          |    _4b7 <= orr(UInt<4>(7))
          |    _4b1 <= orr(UInt<4>(1))
          |    wire _0bI: UInt<0>
          |    _0bI is invalid
          |    _0b0 <= orr(_0bI)
          |""".stripMargin
    val check =
      s"""|circuit Foo:
          |  module Foo:
          |    output _4b0: UInt<1>
          |    output _4b15: UInt<1>
          |    output _4b7: UInt<1>
          |    output _4b1: UInt<1>
          |    output _0b0: UInt<1>
          |    _4b0 <= UInt<1>(0)
          |    _4b15 <= UInt<1>(1)
          |    _4b7 <= UInt<1>(1)
          |    _4b1 <= UInt<1>(1)
          |    _0b0 <= UInt<1>(0)
          |""".stripMargin
    execute(input, check, Seq.empty)
  }

  it should "optimize xorr of a literal" in {
    val input =
      s"""|circuit Foo:
          |  module Foo:
          |    output _4b0: UInt<1>
          |    output _4b15: UInt<1>
          |    output _4b7: UInt<1>
          |    output _4b1: UInt<1>
          |    output _0b0: UInt<1>
          |    _4b0 <= xorr(UInt<4>(0))
          |    _4b15 <= xorr(UInt<4>(15))
          |    _4b7 <= xorr(UInt<4>(7))
          |    _4b1 <= xorr(UInt<4>(1))
          |    wire _0bI: UInt<0>
          |    _0bI is invalid
          |    _0b0 <= xorr(_0bI)
          |""".stripMargin
    val check =
      s"""|circuit Foo:
          |  module Foo:
          |    output _4b0: UInt<1>
          |    output _4b15: UInt<1>
          |    output _4b7: UInt<1>
          |    output _4b1: UInt<1>
          |    output _0b0: UInt<1>
          |    _4b0 <= UInt<1>(0)
          |    _4b15 <= UInt<1>(0)
          |    _4b7 <= UInt<1>(1)
          |    _4b1 <= UInt<1>(1)
          |    _0b0 <= UInt<1>(0)
          |""".stripMargin
    execute(input, check, Seq.empty)
  }

  it should "optimize bitwise operations of signed literals" in {
    val input =
      s"""|circuit Foo:
          |  module Foo:
          |    input in1: SInt<3>
          |    output out1: UInt<2>
          |    output out2: UInt<2>
          |    output out3: UInt<2>
          |    output out4: UInt<4>
          |    out1 <= xor(SInt<2>(-1), SInt<2>(1))
          |    out2 <= or(SInt<2>(-1), SInt<2>(1))
          |    out3 <= and(SInt<2>(-1), SInt<2>(-2))
          |    out4 <= xor(in1, SInt<4>(0))
          |""".stripMargin
    val check =
      s"""|circuit Foo:
          |  module Foo:
          |    input in1: SInt<3>
          |    output out1: UInt<2>
          |    output out2: UInt<2>
          |    output out3: UInt<2>
          |    output out4: UInt<4>
          |    out1 <= UInt<2>(2)
          |    out2 <= UInt<2>(3)
          |    out3 <= UInt<2>(2)
          |    node _GEN_0 = pad(in1, 4)
          |    out4 <= asUInt(_GEN_0)
          |""".stripMargin
    execute(input, check, Seq.empty)
  }

  "ConstProp" should "compose with Dedup and not duplicate modules " in {
    val input =
      """circuit Top :
        |  module child :
        |    input x : UInt<1>
        |    output z : UInt<1>
        |    z <= not(x)
        |  module child_1 :
        |    input x : UInt<1>
        |    output z : UInt<1>
        |    z <= not(x)
        |  module Top :
        |    input x : UInt<1>
        |    output z : UInt<1>
        |    inst c of child
        |    inst c_1 of child_1
        |    c.x <= x
        |    c_1.x <= x
        |    z <= and(c.z, c_1.z)""".stripMargin
    val check =
      """circuit Top :
        |  module child :
        |    input x : UInt<1>
        |    output z : UInt<1>
        |    z <= not(x)
        |  module Top :
        |    input x : UInt<1>
        |    output z : UInt<1>
        |    inst c of child
        |    inst c_1 of child
        |    z <= and(c.z, c_1.z)
        |    c.x <= x
        |    c_1.x <= x""".stripMargin
    execute(input, check, Seq(dontTouch("child.z"), dontTouch("child_1.z")))
  }
}

class ConstantPropagationEquivalenceSpec extends FirrtlFlatSpec {
  private val srcDir = "/constant_propagation_tests"
  private val transforms = Seq(new ConstantPropagation)

  "anything added to zero" should "be equal to itself" in {
    val input =
      s"""circuit AddZero :
         |  module AddZero :
         |    input in : UInt<5>
         |    output out1 : UInt<6>
         |    output out2 : UInt<6>
         |    out1 <= add(in, UInt<5>("h0"))
         |    out2 <= add(UInt<5>("h0"), in)""".stripMargin
    firrtlEquivalenceTest(input, transforms)
  }

  "constants added together" should "be propagated" in {
    val input =
      s"""circuit AddLiterals :
         |  module AddLiterals :
         |    input uin : UInt<5>
         |    input sin : SInt<5>
         |    output uout : UInt<6>
         |    output sout : SInt<6>
         |    node uconst = add(UInt<5>("h1"), UInt<5>("h2"))
         |    uout <= add(uconst, uin)
         |    node sconst = add(SInt<5>("h1"), SInt<5>("h-1"))
         |    sout <= add(sconst, sin)""".stripMargin
    firrtlEquivalenceTest(input, transforms)
  }

  "UInt addition" should "have the correct widths" in {
    val input =
      s"""circuit WidthsAddUInt :
         |  module WidthsAddUInt :
         |    input in : UInt<3>
         |    output out1 : UInt<10>
         |    output out2 : UInt<10>
         |    wire temp : UInt<5>
         |    temp <= add(in, UInt<1>("h0"))
         |    out1 <= cat(temp, temp)
         |    node const = add(UInt<4>("h1"), UInt<3>("h2"))
         |    out2 <= cat(const, const)""".stripMargin
    firrtlEquivalenceTest(input, transforms)
  }

  "SInt addition" should "have the correct widths" in {
    val input =
      s"""circuit WidthsAddSInt :
         |  module WidthsAddSInt :
         |    input in : SInt<3>
         |    output out1 : UInt<10>
         |    output out2 : UInt<10>
         |    wire temp : SInt<5>
         |    temp <= add(in, SInt<7>("h0"))
         |    out1 <= cat(temp, temp)
         |    node const = add(SInt<4>("h1"), SInt<3>("h-2"))
         |    out2 <= cat(const, const)""".stripMargin
    firrtlEquivalenceTest(input, transforms)
  }

  // https://github.com/chipsalliance/firrtl/issues/2034
  "SInt OR with constant zero" should "have the correct widths" in {
    val input =
      s"""circuit WidthsOrSInt :
         |  module WidthsOrSInt :
         |    input in : SInt<1>
         |    input in2 : SInt<4>
         |    output out : UInt<8>
         |    output out2 : UInt<8>
         |    out <= or(in, SInt<8>(0))
         |    out2 <= or(in2, SInt<8>(0))""".stripMargin
    firrtlEquivalenceTest(input, transforms)
  }

  "addition by zero width wires" should "have the correct widths" in {
    val input =
      s"""circuit ZeroWidthAdd:
         |  module ZeroWidthAdd:
         |    input x: UInt<0>
         |    output y: UInt<7>
         |    node temp = add(x, UInt<9>("h0"))
         |    y <= cat(temp, temp)""".stripMargin
    firrtlEquivalenceTest(input, transforms)
  }

  "tail of constants" should "be propagated" in {
    val input =
      s"""circuit TailTester :
         |  module TailTester :
         |    output out : UInt<1>
         |    node temp = add(UInt<1>("h00"), UInt<5>("h017"))
         |    node tail_temp = tail(temp, 1)
         |    out <= tail_temp""".stripMargin
    firrtlEquivalenceTest(input, transforms)
  }

  "head of constants" should "be propagated" in {
    val input =
      s"""circuit TailTester :
         |  module TailTester :
         |    output out : UInt<1>
         |    node temp = add(UInt<1>("h00"), UInt<5>("h017"))
         |    node head_temp = head(temp, 3)
         |    out <= head_temp""".stripMargin
    firrtlEquivalenceTest(input, transforms)
  }

  "reduction of literals" should "be propagated" in {
    val input =
      s"""circuit ConstPropReductionTester :
         |  module ConstPropReductionTester :
         |    output out1 : UInt<1>
         |    output out2 : UInt<1>
         |    output out3 : UInt<1>
         |    out1 <= xorr(SInt<2>(-1))
         |    out2 <= andr(SInt<2>(-1))
         |    out3 <= orr(SInt<2>(-1))""".stripMargin
    firrtlEquivalenceTest(input, transforms)
  }

  "addition of negative literals" should "be propagated" in {
    val input =
      s"""circuit AddTester :
         |  module AddTester :
         |    output ref : SInt<2>
         |    ref <= add(SInt<1>("h-1"), SInt<1>("h-1"))
         |""".stripMargin
    firrtlEquivalenceTest(input, transforms)
  }

  "propagation of signed expressions" should "have the correct signs" in {
    val input =
      s"""circuit SignTester :
         |  module SignTester :
         |    output ref : SInt<3>
         |    ref <= mux(UInt<1>("h0"), SInt<3>("h0"), neg(UInt<2>("h3")))
         |""".stripMargin
    firrtlEquivalenceTest(input, transforms)
  }
}
