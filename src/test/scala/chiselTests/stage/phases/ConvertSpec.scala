// See LICENSE for license details.

package chiselTests.stage.phases


import chisel3._
import chisel3.experimental.{ChiselAnnotation, RunFirrtlTransform}
import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.stage.phases.{Convert, Elaborate}

import firrtl.{AnnotationSeq, CircuitForm, CircuitState, DependencyAPIMigration, Transform, UnknownForm}
import firrtl.annotations.{Annotation, NoTargetAnnotation}
import firrtl.options.Phase
import firrtl.stage.{FirrtlCircuitAnnotation, RunFirrtlTransformAnnotation}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ConvertSpecFirrtlTransform extends Transform with DependencyAPIMigration {
  override def prerequisites = Seq.empty
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Transform) = false
  def execute(state: CircuitState): CircuitState = state
}

case class ConvertSpecFirrtlAnnotation(name: String) extends NoTargetAnnotation

case class ConvertSpecChiselAnnotation(name: String) extends ChiselAnnotation with RunFirrtlTransform {
  def toFirrtl: Annotation = ConvertSpecFirrtlAnnotation(name)
  def transformClass: Class[_ <: Transform] = classOf[ConvertSpecFirrtlTransform]
}

class ConvertSpecFoo extends RawModule {
  override val desiredName: String = "foo"

  val in = IO(Input(Bool()))
  val out = IO(Output(Bool()))

  experimental.annotate(ConvertSpecChiselAnnotation("bar"))
}

class ConvertSpec extends AnyFlatSpec with Matchers {

  class Fixture { val phase: Phase = new Convert }

  behavior of classOf[Convert].toString

  it should "convert a Chisel Circuit to a FIRRTL Circuit" in new Fixture {
    val annos: AnnotationSeq = Seq(ChiselGeneratorAnnotation(() => new ConvertSpecFoo))

    val annosx = Seq(new Elaborate, phase)
      .foldLeft(annos)( (a, p) => p.transform(a) )

    info("FIRRTL circuit generated")
    annosx.collect{ case a: FirrtlCircuitAnnotation => a.circuit.main }.toSeq should be (Seq("foo"))

    info("FIRRTL annotations generated")
    annosx.collect{ case a: ConvertSpecFirrtlAnnotation => a.name }.toSeq should be (Seq("bar"))

    info("FIRRTL transform annotations generated")
    annosx.collect{ case a: RunFirrtlTransformAnnotation => a.transform.getClass}
      .toSeq should be (Seq(classOf[ConvertSpecFirrtlTransform]))
  }

}
