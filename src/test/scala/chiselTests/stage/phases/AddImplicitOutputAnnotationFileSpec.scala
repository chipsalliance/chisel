// See LICENSE for license details.

package chiselTests.stage.phases

import org.scalatest.{FlatSpec, Matchers}

import chisel3.RawModule
import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.stage.phases.{AddImplicitOutputAnnotationFile, Elaborate}

import firrtl.AnnotationSeq
import firrtl.options.{OutputAnnotationFileAnnotation, Phase}

class AddImplicitOutputAnnotationFileSpec extends FlatSpec with Matchers {

  class Foo extends RawModule { override val desiredName = "Foo" }

  class Fixture { val phase: Phase = new AddImplicitOutputAnnotationFile }

  behavior of classOf[AddImplicitOutputAnnotationFile].toString

  it should "not override an existing OutputAnnotationFileAnnotation" in new Fixture {
    val annotations: AnnotationSeq = Seq(
      ChiselGeneratorAnnotation(() => new Foo),
      OutputAnnotationFileAnnotation("Bar") )

    Seq( new Elaborate, phase )
      .foldLeft(annotations)((a, p) => p.transform(a))
      .collect{ case a: OutputAnnotationFileAnnotation => a.file }
      .toSeq should be (Seq("Bar"))
  }

  it should "generate an OutputAnnotationFileAnnotation from a ChiselCircuitAnnotation" in new Fixture {
    val annotations: AnnotationSeq = Seq( ChiselGeneratorAnnotation(() => new Foo) )

    Seq( new Elaborate, phase )
      .foldLeft(annotations)((a, p) => p.transform(a))
      .collect{ case a: OutputAnnotationFileAnnotation => a.file }
      .toSeq should be (Seq("Foo"))
  }

}
