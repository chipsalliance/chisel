// See LICENSE for license details.

package chisel3.stage.phases

import org.scalatest.{FlatSpec, Matchers}

import chisel3.stage.{NoRunFirrtlCompilerAnnotation, ChiselOutputFileAnnotation}

import firrtl.options.{OutputAnnotationFileAnnotation, StageOptions}
import firrtl.options.Viewer.view
import firrtl.stage.phases.DriverCompatibility.TopNameAnnotation

class DriverCompatibilitySpec extends FlatSpec with Matchers {

  behavior of classOf[DriverCompatibility.AddImplicitOutputFile].toString

  it should "do nothing if a ChiselOutputFileAnnotation is present" in {
    val annotations = Seq(
      ChiselOutputFileAnnotation("Foo"),
      TopNameAnnotation("Bar") )
    (new DriverCompatibility.AddImplicitOutputFile).transform(annotations).toSeq should be (annotations)
  }

  it should "add a ChiselOutputFileAnnotation derived from a TopNameAnnotation" in {
    val annotations = Seq( TopNameAnnotation("Bar") )
    val expected = ChiselOutputFileAnnotation("Bar") +: annotations
    (new DriverCompatibility.AddImplicitOutputFile).transform(annotations).toSeq should be (expected)
  }

  behavior of classOf[DriverCompatibility.AddImplicitOutputAnnotationFile].toString

  it should "do nothing if an OutputAnnotationFileAnnotation is present" in {
    val annotations = Seq(
      OutputAnnotationFileAnnotation("Foo"),
      TopNameAnnotation("Bar") )
    (new DriverCompatibility.AddImplicitOutputAnnotationFile).transform(annotations).toSeq should be (annotations)
  }

  it should "add an OutputAnnotationFileAnnotation derived from a TopNameAnnotation" in {
    val annotations = Seq( TopNameAnnotation("Bar") )
    val expected = OutputAnnotationFileAnnotation("Bar") +: annotations
    (new DriverCompatibility.AddImplicitOutputAnnotationFile).transform(annotations).toSeq should be (expected)
  }

  behavior of classOf[DriverCompatibility.DisableFirrtlStage].toString

  it should "add a NoRunFirrtlCompilerAnnotation if one does not exist" in {
    val annos = Seq(NoRunFirrtlCompilerAnnotation)
    val expected = DriverCompatibility.RunFirrtlCompilerAnnotation +: annos
    (new DriverCompatibility.DisableFirrtlStage).transform(Seq.empty).toSeq should be (expected)
  }

  it should "NOT add a NoRunFirrtlCompilerAnnotation if one already exists" in {
    val annos = Seq(NoRunFirrtlCompilerAnnotation)
    (new DriverCompatibility.DisableFirrtlStage).transform(annos).toSeq should be (annos)
  }

  behavior of classOf[DriverCompatibility.ReEnableFirrtlStage].toString

  it should "NOT strip a NoRunFirrtlCompilerAnnotation if NO RunFirrtlCompilerAnnotation is present" in {
    val annos = Seq(NoRunFirrtlCompilerAnnotation, DriverCompatibility.RunFirrtlCompilerAnnotation)
    (new DriverCompatibility.ReEnableFirrtlStage).transform(annos).toSeq should be (Seq.empty)
  }

  it should "strip a NoRunFirrtlCompilerAnnotation if a RunFirrtlCompilerAnnotation is present" in {
    val annos = Seq(NoRunFirrtlCompilerAnnotation)
    (new DriverCompatibility.ReEnableFirrtlStage).transform(annos).toSeq should be (annos)
  }

}
