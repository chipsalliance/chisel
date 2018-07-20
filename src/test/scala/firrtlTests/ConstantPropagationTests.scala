// See LICENSE for license details.

package firrtlTests

import firrtl._
import firrtl.ir.Circuit
import firrtl.Parser.IgnoreInfo
import firrtl.passes._
import firrtl.transforms._

class ConstantPropagationSpec extends FirrtlFlatSpec {
  val transforms = Seq(
      ToWorkingIR,
      ResolveKinds,
      InferTypes,
      ResolveGenders,
      InferWidths,
      new ConstantPropagation)
  protected def exec(input: String) = {
    transforms.foldLeft(CircuitState(parse(input), UnknownForm)) {
      (c: CircuitState, t: Transform) => t.runTransform(c)
    }.circuit.serialize
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(circuit(false)))) should be (parse(circuit(false)))
      (parse(exec(circuit(true)))) should be (parse(resultFromAllSame))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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
      (parse(exec(input))) should be (parse(check))
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

  "ConstProp" should "NOT optimize across dontTouch on registers" in {
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
}
