// SPDX-License-Identifier: Apache-2.0

package chiselTests.stage.phases

import chisel3.RawModule
import chisel3.stage.{ChiselCircuitAnnotation, ChiselGeneratorAnnotation, ChiselOutputFileAnnotation}
import chisel3.stage.phases.{Convert, Elaborate, Emitter}

import firrtl.{AnnotationSeq, EmittedFirrtlCircuitAnnotation}
import firrtl.options.{Phase, TargetDirAnnotation}

import java.io.File
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class EmitterSpec extends AnyFlatSpec with Matchers {

  class FooModule extends RawModule { override val desiredName = "Foo" }
  class BarModule extends RawModule { override val desiredName = "Bar" }

  class Fixture { val phase: Phase = new Emitter }

  behavior.of(classOf[Emitter].toString)

  it should "do nothing if no ChiselOutputFileAnnotations are present" in new Fixture {
    val dir = new File("test_run_dir/EmitterSpec")
    val annotations =
      (new Elaborate).transform(Seq(TargetDirAnnotation(dir.toString), ChiselGeneratorAnnotation(() => new FooModule)))
    val annotationsx = phase.transform(annotations)

    val Seq(fooFile, barFile) = Seq("Foo.fir", "Bar.fir").map(f => new File(dir.toString + "/" + f))

    info(s"$fooFile does not exist")
    fooFile should not(exist)

    info("annotations are unmodified")
    annotationsx.toSeq should be(annotations.toSeq)
  }

  it should "emit a ChiselCircuitAnnotation to a specific file" in new Fixture {
    val dir = new File("test_run_dir/EmitterSpec")
    val circuit = (new Elaborate)
      .transform(Seq(ChiselGeneratorAnnotation(() => new BarModule)))
      .collectFirst { case a: ChiselCircuitAnnotation => a }
      .get
    val annotations =
      phase.transform(Seq(TargetDirAnnotation(dir.toString), circuit, ChiselOutputFileAnnotation("Baz")))

    val bazFile = new File(dir.toString + "/Baz.fir")

    info(s"$bazFile exists")
    bazFile should (exist)
  }

}
