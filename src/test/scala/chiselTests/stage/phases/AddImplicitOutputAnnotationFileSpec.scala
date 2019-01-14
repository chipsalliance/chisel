// See LICENSE for license details.

package chiselTests.stage.phases

import org.scalatest.{FlatSpec, Matchers}

import chisel3.experimental.RawModule
import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.stage.phases.{AddImplicitOutputAnnotationFile, Elaborate}

import firrtl.AnnotationSeq
import firrtl.options.OutputAnnotationFileAnnotation

class AddImplicitOutputAnnotationFileSpec extends FlatSpec with Matchers {

  class Foo extends RawModule { override val desiredName = "Foo" }

  behavior of AddImplicitOutputAnnotationFile.name

  it should "not override an existing OutputAnnotationFileAnnotation" in {
    val annotations: AnnotationSeq = Seq(
      ChiselGeneratorAnnotation(() => new Foo),
      OutputAnnotationFileAnnotation("Bar") )

    Seq( Elaborate, AddImplicitOutputAnnotationFile )
      .foldLeft(annotations)((a, p) => p.transform(a))
      .collect{ case a: OutputAnnotationFileAnnotation => a.file }
      .toSeq should be (Seq("Bar"))
  }

  it should "generate an OutputAnnotationFileAnnotation from a ChiselCircuitAnnotation" in {
    val annotations: AnnotationSeq = Seq( ChiselGeneratorAnnotation(() => new Foo) )

    Seq( Elaborate, AddImplicitOutputAnnotationFile )
      .foldLeft(annotations)((a, p) => p.transform(a))
      .collect{ case a: OutputAnnotationFileAnnotation => a.file }
      .toSeq should be (Seq("Foo"))
  }

}
