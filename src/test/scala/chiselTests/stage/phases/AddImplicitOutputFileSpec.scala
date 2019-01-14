// See LICENSE for license details.

package chiselTests.stage.phases

import org.scalatest.{FlatSpec, Matchers}

import chisel3.experimental.RawModule
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselOutputFileAnnotation}
import chisel3.stage.phases.{AddImplicitOutputFile, Elaborate}

import firrtl.AnnotationSeq
import firrtl.options.{StageOptions, TargetDirAnnotation}
import firrtl.options.Viewer.view

class AddImplicitOutputFileSpec extends FlatSpec with Matchers {

  class Foo extends RawModule { override val desiredName = "Foo" }

  behavior of AddImplicitOutputFile.name

  it should "not override an existing ChiselOutputFileAnnotation" in {
    val annotations: AnnotationSeq = Seq(
      ChiselGeneratorAnnotation(() => new Foo),
      ChiselOutputFileAnnotation("Bar") )

    Seq( Elaborate, AddImplicitOutputFile )
      .foldLeft(annotations)((a, p) => p.transform(a))
      .collect{ case a: ChiselOutputFileAnnotation => a.file }
      .toSeq should be (Seq("Bar"))
  }

  it should "generate a ChiselOutputFileAnnotation from a ChiselCircuitAnnotation" in {
    val annotations: AnnotationSeq = Seq(
      ChiselGeneratorAnnotation(() => new Foo),
      TargetDirAnnotation("test_run_dir") )

    Seq( Elaborate, AddImplicitOutputFile )
      .foldLeft(annotations)((a, p) => p.transform(a))
      .collect{ case a: ChiselOutputFileAnnotation => a.file }
      .toSeq should be (Seq("Foo"))
  }

  it should "do nothing to an empty annotation sequence" in {
    AddImplicitOutputFile.transform(AnnotationSeq(Seq.empty)).toSeq should be (empty)
  }

}
