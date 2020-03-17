// See LICENSE for license details.

package firrtlTests.transforms

import firrtl.testutils.FirrtlFlatSpec
import firrtl._
import firrtl.passes._
import firrtl.passes.wiring.{WiringTransform, SourceAnnotation, SinkAnnotation}
import firrtl.annotations._
import firrtl.annotations.TargetToken.{Field, Index}


class InferWidthsWithAnnosSpec extends FirrtlFlatSpec {
  private def executeTest(input: String,
    check: String,
    transforms: Seq[Transform],
    annotations: Seq[Annotation]) = {
    val start = CircuitState(parse(input), ChirrtlForm, annotations)
    val end = transforms.foldLeft(start) {
      (c: CircuitState, t: Transform) => t.runTransform(c)
    }
    val resLines = end.circuit.serialize.split("\n") map normalized
    val checkLines = parse(check).serialize.split("\n") map normalized

    resLines should be (checkLines)
  }

  "CheckWidths on wires with unknown widths" should "result in an error" in {
    val transforms = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveFlows,
      new InferWidths,
      CheckWidths)

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

    // A.y should have uninferred width
    intercept[CheckWidths.UninferredWidth] {
      executeTest(input, "", transforms, Seq.empty)
    }
  }

  "InferWidthsWithAnnos" should "infer widths using WidthGeqConstraintAnnotation" in {
    val transforms = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveFlows,
      new InferWidths,
      CheckWidths)

    val annos = Seq(WidthGeqConstraintAnnotation(
      ReferenceTarget("Top", "A", Nil, "y", Nil),
      ReferenceTarget("Top", "B", Nil, "x", Nil)))

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
    val transforms = Seq(
      ToWorkingIR,
      CheckHighForm,
      ResolveKinds,
      InferTypes,
      CheckTypes,
      ResolveFlows,
      new InferWidths,
      CheckWidths)

    val tokenLists = Seq(
      Seq(Field("x")),
      Seq(Field("y"), Index(0), Field("yy")),
      Seq(Field("y"), Index(1), Field("yy"))
    )

    val annos = tokenLists.map { tokens =>
      WidthGeqConstraintAnnotation(
        ReferenceTarget("Top", "A", Nil, "bundle", tokens),
        ReferenceTarget("Top", "B", Nil, "bundle", tokens))
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

  "InferWidthsWithAnnos" should "work with WiringTransform" in {
    def transforms() = Seq(
      ToWorkingIR,
      ResolveKinds,
      InferTypes,
      ResolveFlows,
      new InferWidths,
      CheckWidths,
      new WiringTransform,
      new ResolveAndCheck
    )
    val sourceTarget = ComponentName("bundle", ModuleName("A", CircuitName("Top")))
    val source = SourceAnnotation(sourceTarget, "pin")

    val sinkTarget = ComponentName("bundle", ModuleName("B", CircuitName("Top")))
    val sink = SinkAnnotation(sinkTarget, "pin")

    val tokenLists = Seq(
      Seq(Field("x")),
      Seq(Field("y"), Index(0), Field("yy")),
      Seq(Field("y"), Index(1), Field("yy"))
    )

    val wgeqAnnos = tokenLists.map { tokens =>
      WidthGeqConstraintAnnotation(
        ReferenceTarget("Top", "A", Nil, "bundle", tokens),
        ReferenceTarget("Top", "B", Nil, "bundle", tokens))
    }

    val failAnnos = Seq(source, sink)
    val successAnnos = wgeqAnnos ++: failAnnos

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
        |    wire bundle : {x : UInt<1>, y: {yy : UInt<3>}[2] }
        |    inst b of B
        |    inst a of A
        |    b.pin <= bundle
        |    bundle <= a.bundle_0
        |
        |  module B :
        |    input pin : {x : UInt<1>, y: {yy : UInt<3>}[2] }
        |    wire bundle : {x : UInt<1>, y: {yy : UInt<3>}[2] }
        |    bundle <= pin
        |
        |  module A :
        |    output bundle_0 : {x : UInt<1>, y: {yy : UInt<3>}[2] }
        |    wire bundle : {x : UInt<1>, y: {yy : UInt<3>}[2] }
        |    bundle_0 <= bundle"""
        .stripMargin

    // should fail without extra constraint annos due to UninferredWidths
    val exceptions = intercept[PassExceptions] {
      executeTest(input, "", transforms, failAnnos)
    }.exceptions.reverse

    val msg = exceptions.head.toString
    assert(msg.contains(s"2 errors detected!"))
    assert(exceptions.tail.forall(_.isInstanceOf[CheckWidths.UninferredWidth]))

    // should pass with extra constraints
    executeTest(input, output, transforms, successAnnos)
  }
}
