// See LICENSE for license details.

package firrtlTests.annotationTests

import firrtl.{AnnotationSeq, CircuitState, Parser, UnknownForm}
import firrtl.annotations.{
  CircuitName,
  CircuitTarget,
  ComponentName,
  ModuleName,
  MultiTargetAnnotation,
  ReferenceTarget,
  SingleTargetAnnotation,
  Target}
import firrtl.annotations.transforms.CleanupNamedTargets

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

object CleanupNamedTargetsSpec {

  case class SingleReferenceAnnotation(target: ReferenceTarget) extends SingleTargetAnnotation[ReferenceTarget] {
    override def duplicate(a: ReferenceTarget) = this.copy(target = a)
  }

  case class MultiReferenceAnnotation(targets: Seq[Seq[Target]]) extends MultiTargetAnnotation {
    override def duplicate(a: Seq[Seq[Target]]) = this.copy(targets = a)
  }

}

class CleanupNamedTargetsSpec extends AnyFlatSpec with Matchers {

  import CleanupNamedTargetsSpec._

  class F {

    val transform = new CleanupNamedTargets

    val circuit =
      """|circuit Foo:
         |  module Bar:
         |    skip
         |  module Foo:
         |    inst bar of Bar
         |    wire baz: UInt<1>
         |""".stripMargin

    lazy val foo = CircuitTarget("Foo").module("Foo")

    lazy val barTarget = ComponentName("bar", ModuleName("Foo", CircuitName("Foo"))).toTarget

    lazy val bazTarget = ComponentName("baz", ModuleName("Foo", CircuitName("Foo"))).toTarget

    def circuitState(a: AnnotationSeq): CircuitState = CircuitState(Parser.parse(circuit), UnknownForm, a, None)

  }

  behavior of "CleanupNamedTargets"

  it should "convert a SingleTargetAnnotation[ReferenceTarget] of an instance to an InstanceTarget" in new F {
    val annotations: AnnotationSeq = Seq(SingleReferenceAnnotation(barTarget))

    transform.transform(circuitState(annotations)).renames.get.get(barTarget) should be {
      Some(Seq(foo.instOf("bar", "Bar")))
    }
  }

  it should "convert a MultiTargetAnnotation with a ReferenceTarget of an instance to an InstanceTarget" in new F {
    val annotations: AnnotationSeq = Seq(MultiReferenceAnnotation(Seq(Seq(barTarget), Seq(bazTarget))))

    val renames = transform.transform(circuitState(annotations)).renames.get

    renames.get(barTarget) should be (Some(Seq(foo.instOf("bar", "Bar"))))

    info("and not touch a true ReferenceAnnotation")
    renames.get(bazTarget) should be (None)

  }

}
