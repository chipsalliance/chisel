// See LICENSE for license details.

package firrtlTests

import firrtl._
import firrtl.ir.Circuit
import firrtl.Parser.IgnoreInfo
import firrtl.passes._
import firrtl.transforms._

// Tests the following cases for constant propagation:
//   1) Unsigned integers are always greater than or
//        equal to zero
//   2) Values are always smaller than a number greater
//        than their maximum value
//   3) Values are always greater than a number smaller
//        than their minimum value
class ConstantPropagationSpec extends FirrtlFlatSpec {
  val transforms = Seq(
      ToWorkingIR,
      ResolveKinds,
      InferTypes,
      ResolveGenders,
      InferWidths,
      new ConstantPropagation)
  private def exec(input: String) = {
    transforms.foldLeft(CircuitState(parse(input), UnknownForm)) {
      (c: CircuitState, t: Transform) => t.runTransform(c)
    }.circuit.serialize
  }
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
    z <= n
"""
      val check =
"""circuit Top :
  module Top :
    input x : UInt<1>
    input y : UInt<1>
    output z : UInt<1>
    node n = and(x, y)
    node _T_1 = n
    z <= n
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
    hit <= _T_1
    z <= hit
"""
      (parse(exec(input))) should be (parse(check))
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
      val check =
        """circuit Top :
          |  module Top :
          |    input x : UInt<1>
          |    output y : UInt<1>
          |    node z = x
          |    y <= z""".stripMargin
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
          |    wire z : UInt<1>
          |    y <= z
          |    z <= x""".stripMargin
    execute(input, check, Seq(dontTouch("Top.z")))
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
}
