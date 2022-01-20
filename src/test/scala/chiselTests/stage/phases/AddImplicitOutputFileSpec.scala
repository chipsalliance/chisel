// SPDX-License-Identifier: Apache-2.0

package chiselTests.stage.phases

import chisel3.RawModule
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselOutputFileAnnotation}
import chisel3.stage.phases.{AddImplicitOutputFile, Elaborate}

import firrtl.AnnotationSeq
import firrtl.options.{Phase, StageOptions, TargetDirAnnotation}
import firrtl.options.Viewer.view
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AddImplicitOutputFileSpec extends AnyFlatSpec with Matchers {

  class Foo extends RawModule { override val desiredName = "Foo" }

  class Fixture { val phase: Phase = new AddImplicitOutputFile }

  behavior.of(classOf[AddImplicitOutputFile].toString)

  it should "not override an existing ChiselOutputFileAnnotation" in new Fixture {
    val annotations: AnnotationSeq = Seq(ChiselGeneratorAnnotation(() => new Foo), ChiselOutputFileAnnotation("Bar"))

    Seq(new Elaborate, phase)
      .foldLeft(annotations)((a, p) => p.transform(a))
      .collect { case a: ChiselOutputFileAnnotation => a.file }
      .toSeq should be(Seq("Bar"))
  }

  it should "generate a ChiselOutputFileAnnotation from a ChiselCircuitAnnotation" in new Fixture {
    val annotations: AnnotationSeq = Seq(ChiselGeneratorAnnotation(() => new Foo), TargetDirAnnotation("test_run_dir"))

    Seq(new Elaborate, phase)
      .foldLeft(annotations)((a, p) => p.transform(a))
      .collect { case a: ChiselOutputFileAnnotation => a.file }
      .toSeq should be(Seq("Foo"))
  }

  it should "do nothing to an empty annotation sequence" in new Fixture {
    phase.transform(AnnotationSeq(Seq.empty)).toSeq should be(empty)
  }

}
