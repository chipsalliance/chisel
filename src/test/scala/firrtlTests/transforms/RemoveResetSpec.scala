// SPDX-License-Identifier: Apache-2.0

package firrtlTests.transforms

import org.scalatest.GivenWhenThen

import firrtl.testutils.FirrtlFlatSpec
import firrtl.testutils.FirrtlCheckers._

import firrtl.{CircuitState, WRef}
import firrtl.ir.{Connect, DefRegister, Mux}
import firrtl.stage.{FirrtlCircuitAnnotation, FirrtlSourceAnnotation, FirrtlStage}

class RemoveResetSpec extends FirrtlFlatSpec with GivenWhenThen {

  private def toLowFirrtl(string: String): CircuitState = {
    When("the circuit is compiled to low FIRRTL")
    (new FirrtlStage)
      .execute(Array("-X", "low"), Seq(FirrtlSourceAnnotation(string)))
      .collectFirst { case FirrtlCircuitAnnotation(a) => a }
      .map(a => firrtl.CircuitState(a, firrtl.UnknownForm))
      .get
  }

  behavior.of("RemoveReset")

  it should "not generate a reset mux for an invalid init" in {
    Given("a 1-bit register 'foo' initialized to invalid, 1-bit wire 'bar'")
    val input =
      """|circuit Example :
         |  module Example :
         |    input clock : Clock
         |    input rst : UInt<1>
         |    input in : UInt<1>
         |    output out : UInt<1>
         |
         |    wire bar : UInt<1>
         |    bar is invalid
         |
         |    reg foo : UInt<1>, clock with : (reset => (rst, bar))
         |    foo <= in
         |    out <= foo""".stripMargin

    val outputState = toLowFirrtl(input)

    Then("'foo' is NOT connected to a reset mux")
    outputState shouldNot containTree { case Connect(_, WRef("foo", _, _, _), Mux(_, _, _, _)) => true }
  }

  it should "generate a reset mux for only the portion of an invalid aggregate that is reset" in {
    Given("aggregate register 'foo' with 2-bit field 'a' and 1-bit field 'b'")
    And("aggregate, invalid wire 'bar' with the same fields")
    And("'foo' is initialized to 'bar'")
    And("'bar.a[1]' connected to zero")
    val input =
      """|circuit Example :
         |  module Example :
         |    input clock : Clock
         |    input rst : UInt<1>
         |    input in :  {a : UInt<1>[2], b : UInt<1>}
         |    output out :  {a : UInt<1>[2], b : UInt<1>}
         |
         |    wire bar : {a : UInt<1>[2], b : UInt<1>}
         |    bar is invalid
         |    bar.a[1] <= UInt<1>(0)
         |
         |    reg foo :  {a : UInt<1>[2], b : UInt<1>}, clock with : (reset => (rst, bar))
         |    foo <= in
         |    out <= foo""".stripMargin

    val outputState = toLowFirrtl(input)

    Then("foo.a[0] is NOT connected to a reset mux")
    outputState shouldNot containTree { case Connect(_, WRef("foo_a_0", _, _, _), Mux(_, _, _, _)) => true }
    And("foo.a[1] is connected to a reset mux")
    outputState should containTree { case Connect(_, WRef("foo_a_1", _, _, _), Mux(_, _, _, _)) => true }
    And("foo.b is NOT connected to a reset mux")
    outputState shouldNot containTree { case Connect(_, WRef("foo_b", _, _, _), Mux(_, _, _, _)) => true }
  }

  it should "propagate invalidations across connects" in {
    Given("aggregate register 'foo' with 1-bit field 'a' and 1-bit field 'b'")
    And("aggregate, invalid wires 'bar' and 'baz' with the same fields")
    And("'foo' is initialized to 'baz'")
    And("'bar.a' is connected to zero")
    And("'baz' is connected to 'bar'")
    val input =
      """|circuit Example :
         |  module Example :
         |    input clock : Clock
         |    input rst : UInt<1>
         |    input in : { a : UInt<1>, b : UInt<1> }
         |    output out : { a : UInt<1>, b : UInt<1> }
         |
         |    wire bar : { a : UInt<1>, b : UInt<1> }
         |    bar is invalid
         |    bar.a <= UInt<1>(0)
         |
         |    wire baz : { a : UInt<1>, b : UInt<1> }
         |    baz is invalid
         |    baz <= bar
         |
         |    reg foo : { a : UInt<1>, b : UInt<1> }, clock with : (reset => (rst, baz))
         |    foo <= in
         |    out <= foo""".stripMargin

    val outputState = toLowFirrtl(input)

    Then("'foo.a' is connected to a reset mux")
    outputState should containTree { case Connect(_, WRef("foo_a", _, _, _), Mux(_, _, _, _)) => true }
    And("'foo.b' is NOT connected to a reset mux")
    outputState shouldNot containTree { case Connect(_, WRef("foo_b", _, _, _), Mux(_, _, _, _)) => true }
  }

  it should "canvert a reset wired to UInt<0> to a canonical non-reset" in {
    Given("foo's reset is connected to zero")
    val input =
      """|circuit Example :
         |  module Example :
         |    input clock : Clock
         |    input rst : UInt<1>
         |    input in : UInt<2>
         |    output out : UInt<2>
         |    reg foo : UInt<2>, clock with : (reset => (UInt(0), UInt(3)))
         |    foo <= in
         |    out <= foo""".stripMargin

    val outputState = toLowFirrtl(input)

    Then("foo has a canonical non-reset declaration after RemoveReset")
    outputState should containTree { case DefRegister(_, "foo", _, _, firrtl.Utils.zero, WRef("foo", _, _, _)) => true }
    And("foo is NOT connected to a reset mux")
    outputState shouldNot containTree { case Connect(_, WRef("foo", _, _, _), Mux(_, _, _, _)) => true }
  }
}
