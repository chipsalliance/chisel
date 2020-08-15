// See LICENSE for license details.

package firrtlTests.annotationTests

import firrtl._
import firrtl.annotations.{Annotation, CircuitTarget, CompleteTarget, DeletedAnnotation}
import firrtl.annotations.transforms.{DupedResult, ResolvePaths}
import firrtl.transforms.DedupedResult
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should._

class MorphismSpec extends AnyFlatSpec with Matchers {

  object AnAnnotation {
    def apply(target: CompleteTarget) = new AnAnnotation(Some(target))
  }

  case class AnAnnotation(
    target: Option[CompleteTarget],
    from:   Option[AnAnnotation] = None,
    cause:  Option[String] = None)
      extends Annotation {
    override def update(renames: RenameMap): Seq[AnAnnotation] = {
      if (target.isDefined) {
        renames.get(target.get) match {
          case None          => Seq(this)
          case Some(Seq())   => Seq(AnAnnotation(None, Some(this)))
          case Some(targets) =>
            //TODO: Add cause of renaming, requires FIRRTL change to RenameMap
            targets.map { t => AnAnnotation(Some(t), Some(this)) }
        }
      } else Seq(this)
    }

    private def expand(stringBuilder: StringBuilder): StringBuilder = {
      if (target.isDefined) {
        stringBuilder.append(s"${target.get.serialize}")
      } else {
        stringBuilder.append(s"<DELETED>")
      }
      if (from.isDefined) {
        val arrow = cause.map("(" + _ + ")").getOrElse("")
        stringBuilder.append(s" <-$arrow- ")
        from.get.expand(stringBuilder)
      }
      stringBuilder
    }

    override def serialize: String = expand(new StringBuilder()).toString
  }

  object StripDeleted extends Transform {

    override def inputForm = UnknownForm

    override def outputForm = UnknownForm

    override def execute(a: CircuitState): CircuitState = {

      val annotationsx = a.annotations.filter {
        case a: DeletedAnnotation => false
        case AnAnnotation(None, _, _) => false
        case _: DupedResult   => false
        case _: DedupedResult => false
        case _ => true
      }

      a.copy(annotations = annotationsx)

    }

  }

  trait CircuitFixture {

    /** An input FIRRTL string */
    val input: String

    lazy val output: String = input

    /** Input annotations */
    val annotations: AnnotationSeq = Seq.empty

    val finalAnnotations: Option[AnnotationSeq] = None

    lazy val state = CircuitState(Parser.parse(input), UnknownForm, annotations)
  }

  trait RightInverseFixture extends CircuitFixture {

    /** An endomorphism i.e. a function mapping from CircuitState to CircuitState */
    val f: Seq[Transform]

    /** The right inverse of f */
    val g: Seq[Transform]

    val setup: Seq[Transform] = Seq(
      firrtl.passes.ToWorkingIR,
      new firrtl.ResolveAndCheck
    )

    val cleanup: Seq[Transform] = Seq(
      StripDeleted
    )

    def apply(a: CircuitState): CircuitState = {
      val ax = (setup ++ f ++ g).foldLeft(a) {
        case (state, transform) => transform.runTransform(state)
      }

      cleanup.foldLeft(ax) {
        case (state, transform) => transform.transform(state)
      }
    }

    lazy val outputState = apply(state)

    def test(): Unit = {

      /* The output circuit should be the same as the input circuit */
      outputState.circuit.serialize should be(Parser.parse(output).serialize)
      info("the circuits are the same")
      info(state.circuit.serialize)

      /* The output annotations should match the input annotations */
      info(s"Input annotations:\n\t${state.annotations.toList.mkString("\n\t")}")
      info(s"Output annotations:\n\t${outputState.annotations.toList.mkString("\n\t")}")
      if (finalAnnotations.nonEmpty) {
        info(s"Final annotations:\n\t${finalAnnotations.get.toList.mkString("\n\t")}")
      }

      info(s"Output Annotation History:\n")
      outputState.annotations.collect {
        case a: AnAnnotation => info(a.serialize)
      }

      val inputAnnotations = state.annotations.filter {
        case r: ResolvePaths => false
        case other => true
      }

      if (finalAnnotations.isEmpty) {
        outputState.annotations.size should be(inputAnnotations.size)
        info("the number of annotations is the same")

        outputState.annotations.zip(inputAnnotations).collect {
          case (a: AnAnnotation, b: AnAnnotation) => a.target should be(b.target)
        }
        info("each annotation is the same")
      } else {
        outputState.annotations.zip(finalAnnotations.get).collect {
          case (a: AnAnnotation, b: AnAnnotation) => a.target should be(b.target)
        }

        outputState.annotations.size should be(finalAnnotations.get.size)
        info("the number of annotations is the same")

        info("each annotation is the same as the final annotations")
      }
    }

  }

  trait IdempotencyFixture extends CircuitFixture {

    /** An endomorphism */
    val f: Seq[Transform]

    val setup: Seq[Transform] = Seq(
      firrtl.passes.ToWorkingIR,
      new firrtl.ResolveAndCheck
    )

    val cleanup: Seq[Transform] = Seq(
      StripDeleted
    )

    def apply(a: CircuitState): (CircuitState, CircuitState) = {

      val once = (setup ++ f).foldLeft(a) {
        case (state, transform) => transform.runTransform(state)
      }

      val twice = f.foldLeft(once) {
        case (state, transform) => transform.runTransform(state)
      }

      val onceClean = cleanup.foldLeft(once) {
        case (state, transform) => transform.transform(state)
      }

      val twiceClean = cleanup.foldLeft(twice) {
        case (state, transform) => transform.transform(state)
      }

      (onceClean, twiceClean)

    }

    lazy val (oneApplication, twoApplications) = apply(state)

    def test(): Unit = {

      info("a second application does not change the circuit")
      twoApplications.circuit.serialize should be(oneApplication.circuit.serialize)

      info("each annotation is the same after a second application")
      twoApplications.annotations.zip(oneApplication.annotations).foreach {
        case (a, b) => a should be(b)
      }

      info("the number of annotations after a second application is the same")
      twoApplications.annotations.size should be(oneApplication.annotations.size)

    }

  }

  trait DefaultExample extends CircuitFixture {
    override val input =
      """|circuit Top:
         |  module Foo:
         |    node a = UInt<1>(0)
         |  module Bop:
         |    node a = UInt<1>(0)
         |  module Fub:
         |    node a = UInt<1>(0)
         |  module Bar:
         |    node a = UInt<1>(0)
         |  module Baz:
         |    input x: UInt<1>
         |    inst foo of Foo
         |    inst bar of Bar
         |  module Qux:
         |    input x: UInt<1>
         |    inst foo of Fub
         |    inst bar of Bop
         |  module Top:
         |    inst baz of Baz
         |    inst qux of Qux""".stripMargin

    def deduped =
      """|circuit Top:
         |  module Foo:
         |    node a = UInt<1>(0)
         |  module Baz:
         |    input x: UInt<1>
         |    inst foo of Foo
         |    inst bar of Foo
         |  module Top:
         |    inst baz of Baz
         |    inst qux of Baz""".stripMargin

    def allModuleInstances =
      IndexedSeq(
        CircuitTarget("Top").module("Foo"),
        CircuitTarget("Top").module("Bar"),
        CircuitTarget("Top").module("Fub"),
        CircuitTarget("Top").module("Bop"),
        CircuitTarget("Top").module("Baz"),
        CircuitTarget("Top").module("Qux"),
        CircuitTarget("Top").module("Top")
      )

    def allAbsoluteInstances =
      IndexedSeq(
        CircuitTarget("Top").module("Top").instOf("baz", "Baz").instOf("foo", "Foo"),
        CircuitTarget("Top").module("Top").instOf("baz", "Baz").instOf("bar", "Bar"),
        CircuitTarget("Top").module("Top").instOf("qux", "Qux").instOf("foo", "Fub"),
        CircuitTarget("Top").module("Top").instOf("qux", "Qux").instOf("bar", "Bop"),
        CircuitTarget("Top").module("Top").instOf("baz", "Baz"),
        CircuitTarget("Top").module("Top").instOf("qux", "Qux"),
        CircuitTarget("Top").module("Top")
      )

    def allRelative2LevelInstances =
      IndexedSeq(
        CircuitTarget("Top").module("Baz").instOf("foo", "Foo"),
        CircuitTarget("Top").module("Baz").instOf("bar", "Bar"),
        CircuitTarget("Top").module("Qux").instOf("foo", "Fub"),
        CircuitTarget("Top").module("Qux").instOf("bar", "Bop"),
        CircuitTarget("Top").module("Top").instOf("baz", "Baz"),
        CircuitTarget("Top").module("Top").instOf("qux", "Qux"),
        CircuitTarget("Top").module("Top")
      )

    def allDedupedAbsoluteInstances =
      IndexedSeq(
        CircuitTarget("Top").module("Top").instOf("baz", "Baz").instOf("foo", "Foo"),
        CircuitTarget("Top").module("Top").instOf("baz", "Baz").instOf("bar", "Foo"),
        CircuitTarget("Top").module("Top").instOf("qux", "Baz").instOf("foo", "Foo"),
        CircuitTarget("Top").module("Top").instOf("qux", "Baz").instOf("bar", "Foo"),
        CircuitTarget("Top").module("Top").instOf("baz", "Baz"),
        CircuitTarget("Top").module("Top").instOf("qux", "Baz"),
        CircuitTarget("Top").module("Top")
      )
  }

  behavior.of("EliminateTargetPaths")

  // NOTE: equivalience is defined structurally in this case
  trait RightInverseEliminateTargetsFixture extends RightInverseFixture with DefaultExample {
    override val f: Seq[Transform] = Seq(new firrtl.transforms.DedupModules)
    override val g: Seq[Transform] = Seq(new firrtl.annotations.transforms.EliminateTargetPaths)
  }
  trait IdempotencyEliminateTargetsFixture extends IdempotencyFixture with DefaultExample {
    override val f: Seq[Transform] = Seq(new firrtl.annotations.transforms.EliminateTargetPaths)
  }

  it should "invert DedupModules with no annotations" in new RightInverseEliminateTargetsFixture {
    override val annotations: AnnotationSeq = Seq(
      ResolvePaths(allAbsoluteInstances)
    )
    test()
  }

  it should "invert DedupModules with absolute InstanceTarget annotations" in new RightInverseEliminateTargetsFixture {
    override val annotations: AnnotationSeq =
      allAbsoluteInstances.map(AnAnnotation(_)) :+ ResolvePaths(allAbsoluteInstances)

    override val finalAnnotations: Option[AnnotationSeq] = Some(
      allModuleInstances.map(AnAnnotation.apply)
    )
    test()
  }

  it should "invert DedupModules with all ModuleTarget annotations" in new RightInverseEliminateTargetsFixture {
    override val annotations: AnnotationSeq =
      allModuleInstances.map(AnAnnotation.apply) :+ ResolvePaths(allAbsoluteInstances)
    test()
  }

  it should "invert DedupModules with relative InstanceTarget annotations" in new RightInverseEliminateTargetsFixture {
    override val annotations: AnnotationSeq =
      allRelative2LevelInstances.map(AnAnnotation.apply) :+ ResolvePaths(allAbsoluteInstances)

    override val finalAnnotations: Option[AnnotationSeq] = Some(
      allModuleInstances.map(AnAnnotation.apply)
    )
    test()
  }

  it should "invert DedupModules with a ReferenceTarget annotation" in new RightInverseEliminateTargetsFixture {
    override val annotations: AnnotationSeq = Seq(
      AnAnnotation(CircuitTarget("Top").module("Top").ref("x")),
      ResolvePaths(allAbsoluteInstances)
    )
    test()
  }

  it should "invert DedupModules with partially duplicated modules" in new RightInverseEliminateTargetsFixture {
    override val input =
      """|circuit Top:
         |  module Foo:
         |    node a = UInt<1>(0)
         |  module Bar:
         |    node a = UInt<1>(0)
         |  module Baz:
         |    input x: UInt<1>
         |    inst foo of Foo
         |    inst foox of Foo
         |    inst bar of Bar
         |  module Top:
         |    inst baz of Baz
         |    inst qux of Baz""".stripMargin
    override lazy val output =
      """|circuit Top :
         |  module Foo___Top_baz_bar :
         |    node a = UInt<1>("h0")
         |  module Foo___Top_qux_foox :
         |    node a = UInt<1>("h0")
         |  module Foo___Top_qux_bar :
         |    node a = UInt<1>("h0")
         |  module Foo___Top_baz_foox :
         |    node a = UInt<1>("h0")
         |  module Foo___Top_baz_foo :
         |    node a = UInt<1>("h0")
         |  module Foo___Top_qux_foo :
         |    node a = UInt<1>("h0")
         |  module Baz___Top_baz :
         |    input x : UInt<1>
         |    inst foo of Foo___Top_baz_foo
         |    inst foox of Foo___Top_baz_foox
         |    inst bar of Foo___Top_baz_bar
         |  module Baz___Top_qux :
         |    input x : UInt<1>
         |    inst foo of Foo___Top_qux_foo
         |    inst foox of Foo___Top_qux_foox
         |    inst bar of Foo___Top_qux_bar
         |  module Top :
         |    inst baz of Baz___Top_baz
         |    inst qux of Baz___Top_qux""".stripMargin
    override val annotations: AnnotationSeq = Seq(
      AnAnnotation(CircuitTarget("Top").module("Baz").instOf("foo", "Foo")),
      ResolvePaths(
        Seq(
          CircuitTarget("Top").module("Top").instOf("baz", "Baz").instOf("foo", "Foo"),
          CircuitTarget("Top").module("Top").instOf("baz", "Baz").instOf("foox", "Foo"),
          CircuitTarget("Top").module("Top").instOf("baz", "Baz").instOf("bar", "Bar"),
          CircuitTarget("Top").module("Top").instOf("qux", "Baz").instOf("foo", "Foo"),
          CircuitTarget("Top").module("Top").instOf("qux", "Baz").instOf("foox", "Foo"),
          CircuitTarget("Top").module("Top").instOf("qux", "Baz").instOf("bar", "Bar")
        )
      )
    )

    override val finalAnnotations: Option[AnnotationSeq] = Some(
      Seq(
        AnAnnotation(CircuitTarget("Top").module("Foo___Top_qux_foo")),
        AnAnnotation(CircuitTarget("Top").module("Foo___Top_baz_foo"))
      )
    )
    test()
  }

  it should "be idempotent with per-module annotations" in new IdempotencyEliminateTargetsFixture {

    /** An endomorphism */
    override val annotations: AnnotationSeq =
      allModuleInstances.map(AnAnnotation.apply) :+ ResolvePaths(allAbsoluteInstances)
    test()
  }

  it should "be idempotent with per-instance annotations" in new IdempotencyEliminateTargetsFixture {

    /** An endomorphism */
    override val annotations: AnnotationSeq =
      allAbsoluteInstances.map(AnAnnotation.apply) :+ ResolvePaths(allAbsoluteInstances)
    test()
  }

  it should "be idempotent with relative module annotations" in new IdempotencyEliminateTargetsFixture {

    /** An endomorphism */
    override val annotations: AnnotationSeq =
      allRelative2LevelInstances.map(AnAnnotation.apply) :+ ResolvePaths(allAbsoluteInstances)
    test()
  }

  behavior.of("DedupModules")

  trait RightInverseDedupModulesFixture extends RightInverseFixture with DefaultExample {
    override val f: Seq[Transform] = Seq(new firrtl.annotations.transforms.EliminateTargetPaths)
    override val g: Seq[Transform] = Seq(new firrtl.transforms.DedupModules)
  }

  trait IdempotencyDedupModulesFixture extends IdempotencyFixture with DefaultExample {
    override val f: Seq[Transform] = Seq(new firrtl.transforms.DedupModules)
  }

  it should "invert EliminateTargetPaths with no annotations" in new RightInverseDedupModulesFixture {
    override val annotations: AnnotationSeq = Seq(
      ResolvePaths(allAbsoluteInstances)
    )
    override lazy val output = deduped

    test()
  }

  it should "invert EliminateTargetPaths with absolute InstanceTarget annotations" in new RightInverseDedupModulesFixture {
    override val annotations: AnnotationSeq =
      allAbsoluteInstances.map(AnAnnotation(_)) :+ ResolvePaths(allAbsoluteInstances)
    override val finalAnnotations: Option[AnnotationSeq] = Some(allDedupedAbsoluteInstances.map(AnAnnotation.apply))
    override lazy val output = deduped
    test()
  }

  it should "invert EliminateTargetPaths with all ModuleTarget annotations" in new RightInverseDedupModulesFixture {
    override val annotations: AnnotationSeq =
      allModuleInstances.map(AnAnnotation.apply) :+ ResolvePaths(allAbsoluteInstances)
    override val finalAnnotations: Option[AnnotationSeq] = Some(
      allDedupedAbsoluteInstances.map(AnAnnotation.apply)
    )
    override lazy val output = deduped
    test()
  }

  it should "invert EliminateTargetPaths with partially duplicated modules" in new RightInverseDedupModulesFixture {
    override val input =
      """|circuit Top:
         |  module Foo:
         |    node a = UInt<1>(0)
         |  module Bar:
         |    node a = UInt<1>(0)
         |  module Baz:
         |    input x: UInt<1>
         |    inst foo of Foo
         |    inst foox of Foo
         |    inst bar of Bar
         |  module Top:
         |    inst baz of Baz
         |    inst qux of Baz""".stripMargin
    override lazy val output =
      """|circuit Top :
         |  module Foo :
         |    node a = UInt<1>("h0")
         |  module Baz :
         |    input x : UInt<1>
         |    inst foo of Foo
         |    inst foox of Foo
         |    inst bar of Foo
         |  module Top :
         |    inst baz of Baz
         |    inst qux of Baz""".stripMargin
    override val annotations: AnnotationSeq = Seq(
      AnAnnotation(CircuitTarget("Top").module("Baz").instOf("foo", "Foo")),
      ResolvePaths(
        Seq(
          CircuitTarget("Top").module("Top").instOf("baz", "Baz").instOf("foo", "Foo"),
          CircuitTarget("Top").module("Top").instOf("baz", "Baz").instOf("foox", "Foo"),
          CircuitTarget("Top").module("Top").instOf("baz", "Baz").instOf("bar", "Bar"),
          CircuitTarget("Top").module("Top").instOf("qux", "Baz").instOf("foo", "Foo"),
          CircuitTarget("Top").module("Top").instOf("qux", "Baz").instOf("foox", "Foo"),
          CircuitTarget("Top").module("Top").instOf("qux", "Baz").instOf("bar", "Bar")
        )
      )
    )

    override val finalAnnotations: Option[AnnotationSeq] = Some(
      Seq(
        AnAnnotation(CircuitTarget("Top").module("Top").instOf("baz", "Baz").instOf("foo", "Foo")),
        AnAnnotation(CircuitTarget("Top").module("Top").instOf("qux", "Baz").instOf("foo", "Foo"))
      )
    )
    test()
  }

  it should "be idempotent with per-module annotations" in new IdempotencyDedupModulesFixture {

    /** An endomorphism */
    override val annotations: AnnotationSeq =
      allModuleInstances.map(AnAnnotation.apply) :+ ResolvePaths(allAbsoluteInstances)
    test()
  }

  it should "be idempotent with per-instance annotations" in new IdempotencyDedupModulesFixture {

    /** An endomorphism */
    override val annotations: AnnotationSeq =
      allAbsoluteInstances.map(AnAnnotation.apply) :+ ResolvePaths(allAbsoluteInstances)
    test()
  }

  it should "be idempotent with relative module annotations" in new IdempotencyDedupModulesFixture {

    /** An endomorphism */
    override val annotations: AnnotationSeq =
      allRelative2LevelInstances.map(AnAnnotation.apply) :+ ResolvePaths(allAbsoluteInstances)
    test()
  }

  /*
  //TODO: Future tests of GroupComponents + InlineInstances renaming
  behavior of "GroupComponents"
  it should "invert InlineInstances with not annotations" in (pending)
  it should "invert InlineInstances with InstanceTarget annotations" in (pending)
  it should "invert InlineInstances with a ModuleTarget annotation" in (pending)
  it should "invert InlineInstances with a ReferenceTarget annotation" in (pending)
  it should "be idempotent" in (pending)

  behavior of "InlineInstances"
  it should "invert GroupComponents with not annotations" in (pending)
  it should "invert GroupComponents with InstanceTarget annotations" in (pending)
  it should "invert GroupComponents with a ModuleTarget annotation" in (pending)
  it should "invert GroupComponents with a ReferenceTarget annotation" in (pending)
  it should "be idempotent" in (pending)
   */

}
