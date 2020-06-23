
package firrtlTests.formal

import firrtl.{CircuitState, Parser, Transform, UnknownForm}
import firrtl.testutils.FirrtlFlatSpec
import firrtl.transforms.formal.AssertSubmoduleAssumptions
import firrtl.stage.{Forms, TransformManager}

class AssertSubmoduleAssumptionsSpec extends FirrtlFlatSpec {
  behavior of "AssertSubmoduleAssumptions"

  val transforms = new TransformManager(Forms.HighForm, Forms.MinimalHighForm)
    .flattenedTransformOrder ++ Seq(new AssertSubmoduleAssumptions)

  def run(input: String, check: Seq[String], debug: Boolean = false): Unit = {
    val circuit = Parser.parse(input.split("\n").toIterator)
    val result = transforms.foldLeft(CircuitState(circuit, UnknownForm)) {
      (c: CircuitState, p: Transform) => p.runTransform(c)
    }
    val lines = result.circuit.serialize.split("\n") map normalized

    if (debug) {
      println(lines.mkString("\n"))
    }

    for (ch <- check) {
      lines should contain (ch)
    }
  }

  it should "convert `assume` to `assert` in a submodule" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input clock : Clock
        |    input reset : UInt<1>
        |    input in : UInt<8>
        |    output out : UInt<8>
        |    inst sub of Sub
        |    sub.clock <= clock
        |    sub.reset <= reset
        |    sub.in <= in
        |    out <= sub.out
        |    assume(clock, eq(in, UInt(0)), UInt(1), "assume0")
        |    assert(clock, eq(out, UInt(0)), UInt(1), "assert0")
        |
        |  module Sub :
        |    input clock : Clock
        |    input reset : UInt<1>
        |    input in : UInt<8>
        |    output out : UInt<8>
        |    out <= in
        |    assume(clock, eq(in, UInt(1)), UInt(1), "assume1")
        |    assert(clock, eq(out, UInt(1)), UInt(1), "assert1")
        |""".stripMargin

    val check = Seq(
      "assert(clock, eq(in, UInt<1>(\"h1\")), UInt<1>(\"h1\"), \"assume1\")"
    )
    run(input, check)
  }

  it should "convert `assume` to `assert` in a nested submodule" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input clock : Clock
        |    input reset : UInt<1>
        |    input in : UInt<8>
        |    output out : UInt<8>
        |    inst sub of Sub
        |    sub.clock <= clock
        |    sub.reset <= reset
        |    sub.in <= in
        |    out <= sub.out
        |    assume(clock, eq(in, UInt(0)), UInt(1), "assume0")
        |    assert(clock, eq(out, UInt(0)), UInt(1), "assert0")
        |
        |  module Sub :
        |    input clock : Clock
        |    input reset : UInt<1>
        |    input in : UInt<8>
        |    output out : UInt<8>
        |    inst nestedSub of NestedSub
        |    nestedSub.clock <= clock
        |    nestedSub.reset <= reset
        |    nestedSub.in <= in
        |    out <= nestedSub.out
        |    assume(clock, eq(in, UInt(1)), UInt(1), "assume1")
        |    assert(clock, eq(out, UInt(1)), UInt(1), "assert1")
        |
        |  module NestedSub :
        |    input clock : Clock
        |    input reset : UInt<1>
        |    input in : UInt<8>
        |    output out : UInt<8>
        |    out <= in
        |    assume(clock, eq(in, UInt(2)), UInt(1), "assume2")
        |    assert(clock, eq(out, UInt(2)), UInt(1), "assert2")
        |""".stripMargin

    val check = Seq(
      "assert(clock, eq(in, UInt<1>(\"h1\")), UInt<1>(\"h1\"), \"assume1\")",
      "assert(clock, eq(in, UInt<2>(\"h2\")), UInt<1>(\"h1\"), \"assume2\")"
    )
    run(input, check)
  }
}
