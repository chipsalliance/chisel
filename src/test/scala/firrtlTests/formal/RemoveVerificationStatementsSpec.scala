// SPDX-License-Identifier: Apache-2.0

package firrtlTests.formal

import firrtl.{CircuitState, Parser, Transform, UnknownForm}
import firrtl.stage.{Forms, TransformManager}
import firrtl.testutils.FirrtlFlatSpec
import firrtl.transforms.formal.RemoveVerificationStatements

class RemoveVerificationStatementsSpec extends FirrtlFlatSpec {
  behavior.of("RemoveVerificationStatements")

  val transforms = new TransformManager(Forms.HighForm, Forms.MinimalHighForm).flattenedTransformOrder ++ Seq(
    new RemoveVerificationStatements
  )

  def run(input: String, antiCheck: Seq[String], debug: Boolean = false): Unit = {
    val circuit = Parser.parse(input.split("\n").toIterator)
    val result = transforms.foldLeft(CircuitState(circuit, UnknownForm)) { (c: CircuitState, p: Transform) =>
      p.runTransform(c)
    }
    val lines = result.circuit.serialize.split("\n").map(normalized)

    if (debug) {
      println(lines.mkString("\n"))
    }

    for (ch <- antiCheck) {
      lines should not contain (ch)
    }
  }

  it should "remove all verification statements" in {
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
        |    cover(clock, eq(out, UInt(0)), UInt(1), "cover0")
        |
        |  module Sub :
        |    input clock : Clock
        |    input reset : UInt<1>
        |    input in : UInt<8>
        |    output out : UInt<8>
        |    out <= in
        |    assume(clock, eq(in, UInt(1)), UInt(1), "assume1")
        |    assert(clock, eq(out, UInt(1)), UInt(1), "assert1")
        |    cover(clock, eq(out, UInt(1)), UInt(1), "cover1")
        |""".stripMargin

    val antiCheck = Seq(
      "assume(clock, eq(in, UInt<1>(\"h0\")), UInt<1>(\"h1\"), \"assume0\")",
      "assert(clock, eq(out, UInt<1>(\"h0\")), UInt<1>(\"h1\"), \"assert0\")",
      "cover(clock, eq(out, UInt<1>(\"h0\")), UInt<1>(\"h1\"), \"cover0\")",
      "assume(clock, eq(in, UInt<1>(\"h1\")), UInt<1>(\"h1\"), \"assume1\")",
      "assert(clock, eq(out, UInt<1>(\"h1\")), UInt<1>(\"h1\"), \"assert1\")",
      "cover(clock, eq(out, UInt<1>(\"h1\")), UInt<1>(\"h1\"), \"cover1\")"
    )
    run(input, antiCheck)
  }

}
