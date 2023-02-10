// SPDX-License-Identifier: Apache-2.0

package firrtlTests.transforms

import firrtl.testutils.FirrtlFlatSpec
import firrtl._
import firrtl.passes._
import firrtl.annotations._
import firrtl.annotations.TargetToken.{Field, Index}

class InferWidthsWithAnnosSpec extends FirrtlFlatSpec {
  private def executeTest(input: String, check: String, transforms: Seq[Transform], annotations: Seq[Annotation]) = {
    val start = CircuitState(parse(input), ChirrtlForm, annotations)
    val end = transforms.foldLeft(start) { (c: CircuitState, t: Transform) =>
      t.runTransform(c)
    }
    val resLines = end.circuit.serialize.split("\n").map(normalized)
    val checkLines = parse(check).serialize.split("\n").map(normalized)

    resLines should be(checkLines)
  }

  "InferWidthsWithAnnos" should "infer widths using WidthGeqConstraintAnnotation" in {
    val transforms =
      Seq(ToWorkingIR, ResolveFlows, new InferWidths)

    val annos = Seq(
      WidthGeqConstraintAnnotation(
        ReferenceTarget("Top", "A", Nil, "y", Nil),
        ReferenceTarget("Top", "B", Nil, "x", Nil)
      )
    )

    val input =
      """circuit Top :
        |  module Top :
        |    inst b of B
        |    inst a of A
        |
        |  module B :
        |    wire x: UInt<3>
        |
        |  module A :
        |    wire y: UInt""".stripMargin

    val output =
      """circuit Top :
        |  module Top :
        |    inst b of B
        |    inst a of A
        |
        |  module B :
        |    wire x: UInt<3>
        |
        |  module A :
        |    wire y: UInt<3>""".stripMargin

    // A.y should have same width as B.x
    executeTest(input, output, transforms, annos)
  }

  "InferWidthsWithAnnos" should "work with token paths" in {
    val transforms =
      Seq(ToWorkingIR, ResolveFlows, new InferWidths)

    val tokenLists = Seq(
      Seq(Field("x")),
      Seq(Field("y"), Index(0), Field("yy")),
      Seq(Field("y"), Index(1), Field("yy"))
    )

    val annos = tokenLists.map { tokens =>
      WidthGeqConstraintAnnotation(
        ReferenceTarget("Top", "A", Nil, "bundle", tokens),
        ReferenceTarget("Top", "B", Nil, "bundle", tokens)
      )
    }

    val input =
      """circuit Top :
        |  module Top :
        |    inst b of B
        |    inst a of A
        |
        |  module B :
        |    wire bundle : {x : UInt<1>, y: {yy : UInt<3>}[2] }
        |
        |  module A :
        |    wire bundle : {x : UInt, y: {yy : UInt}[2] }""".stripMargin

    val output =
      """circuit Top :
        |  module Top :
        |    inst b of B
        |    inst a of A
        |
        |  module B :
        |    wire bundle : {x : UInt<1>, y: {yy : UInt<3>}[2] }
        |
        |  module A :
        |    wire bundle : {x : UInt<1>, y: {yy : UInt<3>}[2] }""".stripMargin

    // elements of A.bundle should have same width as B.bundle
    executeTest(input, output, transforms, annos)
  }
}
