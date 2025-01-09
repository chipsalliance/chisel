// SPDX-License-Identifier: Apache-2.0

package chiselTests.stage.phases

import chisel3._
import chisel3.experimental.ChiselAnnotation
import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.stage.phases.{Convert, Elaborate}

import firrtl.AnnotationSeq
import firrtl.annotations.{Annotation, NoTargetAnnotation}
import firrtl.options.Phase
import firrtl.stage.FirrtlCircuitAnnotation
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

case class ConvertSpecFirrtlAnnotation(name: String) extends NoTargetAnnotation

case class ConvertSpecChiselAnnotation(name: String) extends ChiselAnnotation {
  def toFirrtl: Annotation = ConvertSpecFirrtlAnnotation(name)
}

class ConvertSpecFoo extends RawModule {
  override val desiredName: String = "foo"

  val in = IO(Input(Bool()))
  val out = IO(Output(Bool()))

  experimental.annotate(ConvertSpecChiselAnnotation("bar"))
}

class ConvertSpec extends AnyFlatSpec with Matchers {

  class Fixture { val phase: Phase = new Convert }

  behavior.of(classOf[Convert].toString)

  it should "convert a Chisel Circuit to a FIRRTL Circuit" in new Fixture {
    val annos: AnnotationSeq = Seq(ChiselGeneratorAnnotation(() => new ConvertSpecFoo))

    val annosx = Seq(new Elaborate, phase)
      .foldLeft(annos)((a, p) => p.transform(a))

    info("FIRRTL circuit generated")
    val circuit = annosx.collectFirst { case a: FirrtlCircuitAnnotation => a.circuit }.get
    circuit.main should be("foo")

    info("FIRRTL annotations generated")
    annosx.collect { case a: ConvertSpecFirrtlAnnotation => a.name }.toSeq should be(Seq("bar"))

  }

}
