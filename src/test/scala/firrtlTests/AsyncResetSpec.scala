// See LICENSE for license details.

package firrtlTests

import firrtl._
import FirrtlCheckers._

class AsyncResetSpec extends FirrtlFlatSpec {
  def compile(input: String): CircuitState =
    (new VerilogCompiler).compileAndEmit(CircuitState(parse(input), ChirrtlForm), List.empty)
  def compileBody(body: String) = {
    val str = """
      |circuit Test :
      |  module Test :
      |""".stripMargin + body.split("\n").mkString("    ", "\n    ", "")
    compile(str)
  }

  "AsyncReset" should "generate async-reset always blocks" in {
    val result = compileBody(s"""
      |input clock : Clock
      |input reset : AsyncReset
      |input x : UInt<8>
      |output z : UInt<8>
      |reg r : UInt<8>, clock with : (reset => (reset, UInt(123)))
      |r <= x
      |z <= r""".stripMargin
    )
    result should containLine ("always @(posedge clock or posedge reset) begin")
  }

  it should "work in nested and flipped aggregates with regular and partial connect" in {
    val result = compileBody(s"""
      |output fizz : { flip foo : { a : AsyncReset, flip b: AsyncReset }[2], bar : { a : AsyncReset, flip b: AsyncReset }[2] }
      |output buzz : { flip foo : { a : AsyncReset, flip b: AsyncReset }[2], bar : { a : AsyncReset, flip b: AsyncReset }[2] }
      |fizz.bar <= fizz.foo
      |buzz.bar <- buzz.foo
      |""".stripMargin
    )

    result should containLine ("assign fizz_foo_0_b = fizz_bar_0_b;")
    result should containLine ("assign fizz_foo_1_b = fizz_bar_1_b;")
    result should containLine ("assign fizz_bar_0_a = fizz_foo_0_a;")
    result should containLine ("assign fizz_bar_1_a = fizz_foo_1_a;")
    result should containLine ("assign buzz_foo_0_b = buzz_bar_0_b;")
    result should containLine ("assign buzz_foo_1_b = buzz_bar_1_b;")
    result should containLine ("assign buzz_bar_0_a = buzz_foo_0_a;")
    result should containLine ("assign buzz_bar_1_a = buzz_foo_1_a;")
  }

  it should "support casting to other types" in {
    val result = compileBody(s"""
      |input a : AsyncReset
      |output u : Interval[0, 1].0
      |output v : UInt<1>
      |output w : SInt<1>
      |output x : Clock
      |output y : Fixed<1><<0>>
      |output z : AsyncReset
      |u <= asInterval(a, 0, 1, 0)
      |v <= asUInt(a)
      |w <= asSInt(a)
      |x <= asClock(a)
      |y <= asFixedPoint(a, 0)
      |z <= asAsyncReset(a)
      |""".stripMargin
    )
    result should containLine ("assign v = a;")
    result should containLine ("assign w = a;")
    result should containLine ("assign x = a;")
    result should containLine ("assign y = a;")
    result should containLine ("assign z = a;")
  }

  "Other types" should "support casting to AsyncReset" in {
    val result = compileBody(s"""
      |input a : UInt<1>
      |input b : SInt<1>
      |input c : Clock
      |input d : Fixed<1><<0>>
      |input e : AsyncReset
      |input f : Interval[0, 1].0
      |output u : AsyncReset
      |output v : AsyncReset
      |output w : AsyncReset
      |output x : AsyncReset
      |output y : AsyncReset
      |output z : AsyncReset
      |u <= asAsyncReset(a)
      |v <= asAsyncReset(b)
      |w <= asAsyncReset(c)
      |x <= asAsyncReset(d)
      |y <= asAsyncReset(e)
      |z <= asAsyncReset(f)""".stripMargin
    )
    result should containLine ("assign u = a;")
    result should containLine ("assign v = b;")
    result should containLine ("assign w = c;")
    result should containLine ("assign x = d;")
    result should containLine ("assign y = e;")
    result should containLine ("assign z = f;")
  }

  "Non-literals" should "NOT be allowed as reset values for AsyncReset" in {
    an [checks.CheckResets.NonLiteralAsyncResetValueException] shouldBe thrownBy {
      compileBody(s"""
        |input clock : Clock
        |input reset : AsyncReset
        |input x : UInt<8>
        |input y : UInt<8>
        |output z : UInt<8>
        |reg r : UInt<8>, clock with : (reset => (reset, y))
        |r <= x
        |z <= r""".stripMargin
      )
    }
  }

  "Late non-literals connections" should "NOT be allowed as reset values for AsyncReset" in {
    an [checks.CheckResets.NonLiteralAsyncResetValueException] shouldBe thrownBy {
      compileBody(s"""
        |input clock : Clock
        |input reset : AsyncReset
        |input x : UInt<8>
        |input y : UInt<8>
        |output z : UInt<8>
        |wire a : UInt<8>
        |reg r : UInt<8>, clock with : (reset => (reset, a))
        |a <= y
        |r <= x
        |z <= r""".stripMargin
      )
    }
  }

  "Hidden Non-literals" should "NOT be allowed as reset values for AsyncReset" in {
    an [checks.CheckResets.NonLiteralAsyncResetValueException] shouldBe thrownBy {
      compileBody(s"""
        |input clock : Clock
        |input reset : AsyncReset
        |input x : UInt<1>[4]
        |input y : UInt<1>
        |output z : UInt<1>[4]
        |wire literal : UInt<1>[4]
        |literal[0] <= UInt<1>("h00")
        |literal[1] <= y
        |literal[2] <= UInt<1>("h00")
        |literal[3] <= UInt<1>("h00")
        |reg r : UInt<1>[4], clock with : (reset => (reset, literal))
        |r <= x
        |z <= r""".stripMargin
      )
    }
  }
  "Wire connected to non-literal" should "NOT be allowed as reset values for AsyncReset" in {
    an [checks.CheckResets.NonLiteralAsyncResetValueException] shouldBe thrownBy {
      compileBody(s"""
        |input clock : Clock
        |input reset : AsyncReset
        |input x : UInt<1>
        |input y : UInt<1>
        |input cond : UInt<1>
        |output z : UInt<1>
        |wire w : UInt<1>
        |w <= UInt(1)
        |when cond :
        |  w <= y
        |reg r : UInt<1>, clock with : (reset => (reset, w))
        |r <= x
        |z <= r""".stripMargin
      )
    }
  }

  "Complex literals" should "be allowed as reset values for AsyncReset" in {
    val result = compileBody(s"""
        |input clock : Clock
        |input reset : AsyncReset
        |input x : UInt<1>[4]
        |output z : UInt<1>[4]
        |wire literal : UInt<1>[4]
        |literal[0] <= UInt<1>("h00")
        |literal[1] <= UInt<1>("h00")
        |literal[2] <= UInt<1>("h00")
        |literal[3] <= UInt<1>("h00")
        |reg r : UInt<1>[4], clock with : (reset => (reset, literal))
        |r <= x
        |z <= r""".stripMargin
      )
    result should containLine ("always @(posedge clock or posedge reset) begin")
  }

  "Complex literals of complex literals" should "be allowed as reset values for AsyncReset" in {
    val result = compileBody(s"""
        |input clock : Clock
        |input reset : AsyncReset
        |input x : UInt<1>[4]
        |output z : UInt<1>[4]
        |wire literal : UInt<1>[2]
        |literal[0] <= UInt<1>("h01")
        |literal[1] <= UInt<1>("h01")
        |wire complex_literal : UInt<1>[4]
        |complex_literal[0] <= literal[0]
        |complex_literal[1] <= literal[1]
        |complex_literal[2] <= UInt<1>("h00")
        |complex_literal[3] <= UInt<1>("h00")
        |reg r : UInt<1>[4], clock with : (reset => (reset, complex_literal))
        |r <= x
        |z <= r""".stripMargin
      )
    result should containLine ("always @(posedge clock or posedge reset) begin")
  }
  "Literals of bundle literals" should "be allowed as reset values for AsyncReset" in {
    val result = compileBody(s"""
        |input clock : Clock
        |input reset : AsyncReset
        |input x : UInt<1>[4]
        |output z : UInt<1>[4]
        |wire bundle : {a: UInt<1>, b: UInt<1>}
        |bundle.a <= UInt<1>("h01")
        |bundle.b <= UInt<1>("h01")
        |wire complex_literal : UInt<1>[4]
        |complex_literal[0] <= bundle.a
        |complex_literal[1] <= bundle.b
        |complex_literal[2] <= UInt<1>("h00")
        |complex_literal[3] <= UInt<1>("h00")
        |reg r : UInt<1>[4], clock with : (reset => (reset, complex_literal))
        |r <= x
        |z <= r""".stripMargin
      )
    result should containLine ("always @(posedge clock or posedge reset) begin")
  }

  "CheckResets" should "NOT raise StackOverflow Exception on Combinational Loops (should be caught by firrtl.transforms.CheckCombLoops)" in {
    an [firrtl.transforms.CheckCombLoops.CombLoopException] shouldBe thrownBy {
      compileBody(s"""
        |input clock : Clock
        |input reset : AsyncReset
        |wire x : UInt<1>
        |wire y : UInt<2>
        |x <= UInt<1>("h01")
        |node ad = add(x, y)
        |node adt = tail(ad, 1)
        |y <= adt
        |reg r : UInt, clock with : (reset => (reset, y))
        |""".stripMargin
      )
    }
  }

  "Every async reset reg" should "generate its own always block" in {
    val result = compileBody(s"""
      |input clock0 : Clock
      |input clock1 : Clock
      |input syncReset : UInt<1>
      |input asyncReset : AsyncReset
      |input x : UInt<8>[5]
      |output z : UInt<8>[5]
      |reg r0 : UInt<8>, clock0 with : (reset => (syncReset, UInt(123)))
      |reg r1 : UInt<8>, clock1 with : (reset => (syncReset, UInt(123)))
      |reg r2 : UInt<8>, clock0 with : (reset => (asyncReset, UInt(123)))
      |reg r3 : UInt<8>, clock0 with : (reset => (asyncReset, UInt(123)))
      |reg r4 : UInt<8>, clock1 with : (reset => (asyncReset, UInt(123)))
      |r0 <= x[0]
      |r1 <= x[1]
      |r2 <= x[2]
      |r3 <= x[3]
      |r4 <= x[4]
      |z[0] <= r0
      |z[1] <= r1
      |z[2] <= r2
      |z[3] <= r3
      |z[4] <= r4""".stripMargin
    )
    result should containLines (
      "always @(posedge clock0) begin",
      "if (syncReset) begin",
      "r0 <= 8'h7b;",
      "end else begin",
      "r0 <= x_0;",
      "end",
      "end"
    )
    result should containLines (
      "always @(posedge clock1) begin",
      "if (syncReset) begin",
      "r1 <= 8'h7b;",
      "end else begin",
      "r1 <= x_1;",
      "end",
      "end"
    )
    result should containLines (
      "always @(posedge clock0 or posedge asyncReset) begin",
      "if (asyncReset) begin",
      "r2 <= 8'h7b;",
      "end else begin",
      "r2 <= x_2;",
      "end",
      "end"
    )
    result should containLines (
      "always @(posedge clock0 or posedge asyncReset) begin",
      "if (asyncReset) begin",
      "r3 <= 8'h7b;",
      "end else begin",
      "r3 <= x_3;",
      "end",
      "end"
    )
    result should containLines (
      "always @(posedge clock1 or posedge asyncReset) begin",
      "if (asyncReset) begin",
      "r4 <= 8'h7b;",
      "end else begin",
      "r4 <= x_4;",
      "end",
      "end"
    )
  }

}

class AsyncResetExecutionTest extends ExecutionTest("AsyncResetTester", "/features")

