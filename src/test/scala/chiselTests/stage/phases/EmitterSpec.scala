// See LICENSE for license details.

package chiselTests.stage.phases

import org.scalatest.{FlatSpec, Matchers}

import chisel3.experimental.RawModule
import chisel3.stage.{ChiselCircuitAnnotation, ChiselGeneratorAnnotation, ChiselOutputFileAnnotation}
import chisel3.stage.phases.{Convert, Elaborate, Emitter}

import firrtl.{AnnotationSeq, EmittedFirrtlCircuitAnnotation}
import firrtl.annotations.DeletedAnnotation
import firrtl.options.TargetDirAnnotation

import java.io.File

class EmitterSpec extends FlatSpec with Matchers {

  class FooModule extends RawModule { override val desiredName = "Foo" }
  class BarModule extends RawModule { override val desiredName = "Bar" }

  behavior of Emitter.getClass.getName

  it should "do nothing if no ChiselOutputFileAnnotations are present" in {
    val dir = new File("test_run_dir/EmitterSpec")
    val annotations = Elaborate.transform(Seq( TargetDirAnnotation(dir.toString),
                                               ChiselGeneratorAnnotation(() => new FooModule) ))
    val annotationsx = Emitter.transform(annotations)

    val Seq(fooFile, barFile) = Seq("Foo.fir", "Bar.fir").map(f => new File(dir + "/" + f))

    info(s"$fooFile does not exist")
    fooFile should not (exist)

    info("annotations are unmodified")
    annotationsx.toSeq should be (annotations.toSeq)
  }

  it should "emit a ChiselCircuitAnnotation to a specific file" in {
    val dir = new File("test_run_dir/EmitterSpec")
    val circuit = Elaborate
      .transform(Seq(ChiselGeneratorAnnotation(() => new BarModule)))
      .collectFirst{ case a: ChiselCircuitAnnotation => a}
      .get
    val annotations = Emitter.transform(Seq( TargetDirAnnotation(dir.toString),
                                             circuit,
                                             ChiselOutputFileAnnotation("Baz") ))

    val bazFile = new File(dir + "/Baz.fir")

    info(s"$bazFile exists")
    bazFile should (exist)

    info("a deleted EmittedFirrtlCircuitAnnotation should be generated")
    annotations.collect{ case a @ DeletedAnnotation(_, _: EmittedFirrtlCircuitAnnotation) => a }.size should be (1)
  }

}
