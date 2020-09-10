// See LICENSE for license details.

package chiselTests.stage.phases


import chisel3.RawModule
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselOutputFileAnnotation, CircuitSerializationAnnotation}
import chisel3.stage.CircuitSerializationAnnotation._
import chisel3.stage.phases.{AddSerializationAnnotations, AddImplicitOutputFile, Elaborate}

import firrtl.AnnotationSeq
import firrtl.options.{Phase, PhaseManager, Dependency, TargetDirAnnotation}
import firrtl.options.Viewer.view
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AddSerializationAnnotationsSpec extends AnyFlatSpec with Matchers {

  class Foo extends RawModule { override val desiredName = "Foo" }

  class Fixture {
    val phase: Phase = new AddSerializationAnnotations
    val manager = new PhaseManager(Dependency[AddSerializationAnnotations] :: Nil)
  }

  behavior of classOf[AddSerializationAnnotations].toString

  it should "default to FirrtlFileFormat" in new Fixture {
    val annotations: AnnotationSeq = Seq(
      ChiselGeneratorAnnotation(() => new Foo),
      ChiselOutputFileAnnotation("Bar") )

    manager
      .transform(annotations)
      .collect { case CircuitSerializationAnnotation(_, filename, format) => (filename, format) }
      .toSeq should be (Seq(("Bar", FirrtlFileFormat)))
  }

  it should "support ProtoBufFileFormat" in new Fixture {
    val annotations: AnnotationSeq = Seq(
      ChiselGeneratorAnnotation(() => new Foo),
      ChiselOutputFileAnnotation("Bar.pb") )

    manager
      .transform(annotations)
      .collect { case CircuitSerializationAnnotation(_, filename, format) => (filename, format) }
      .toSeq should be (Seq(("Bar", ProtoBufFileFormat)))
  }

  it should "support explicitly asking for FirrtlFileFormat" in new Fixture {
    val annotations: AnnotationSeq = Seq(
      ChiselGeneratorAnnotation(() => new Foo),
      ChiselOutputFileAnnotation("Bar.pb.fir") )

    manager
      .transform(annotations)
      .collect { case CircuitSerializationAnnotation(_, filename, format) => (filename, format) }
      .toSeq should be (Seq(("Bar.pb", FirrtlFileFormat)))
  }

}
