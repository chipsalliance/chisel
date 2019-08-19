// See LICENSE for license details.

package chiselTests.stage.phases

import org.scalatest.{FlatSpec, Matchers}

import chisel3._
import chisel3.experimental.{ChiselAnnotation, RunFirrtlTransform}
import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.stage.phases.{Convert, Elaborate}

import firrtl.{AnnotationSeq, CircuitForm, CircuitState, Transform, UnknownForm}
import firrtl.annotations.{Annotation, NoTargetAnnotation}
import firrtl.options.Phase
import firrtl.stage.{FirrtlCircuitAnnotation, RunFirrtlTransformAnnotation}

class ConvertSpecFirrtlTransform extends Transform {
  def inputForm: CircuitForm = UnknownForm
  def outputForm: CircuitForm = UnknownForm
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

class ConvertSpec extends FlatSpec with Matchers {

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
