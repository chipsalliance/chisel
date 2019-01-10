// See LICENSE for license details.

package chiselTests.stage.phases

import org.scalatest.{FlatSpec, Matchers}

import chisel3.stage.{ChiselOutputFileAnnotation, NoRunFirrtlCompilerAnnotation, PrintFullStackTraceAnnotation}
import chisel3.stage.phases.Checks

import firrtl.AnnotationSeq
import firrtl.annotations.NoTargetAnnotation
import firrtl.options.{OptionsException, Phase}

class ChecksSpec extends FlatSpec with Matchers {

  def checkExceptionMessage(phase: Phase, annotations: AnnotationSeq, messageStart: String): Unit =
    intercept[OptionsException]{ phase.transform(annotations) }.getMessage should startWith(messageStart)

  behavior of Checks.name

  it should "do nothing on sane annotation sequences" in {
    val a = Seq(NoRunFirrtlCompilerAnnotation, PrintFullStackTraceAnnotation)
    Checks.transform(a).toSeq should be (a)
  }

  it should "throw an OptionsException if more than one NoRunFirrtlCompilerAnnotation is specified" in {
    val a = Seq(NoRunFirrtlCompilerAnnotation, NoRunFirrtlCompilerAnnotation)
    checkExceptionMessage(Checks, a, "At most one NoRunFirrtlCompilerAnnotation")
  }

  it should "throw an OptionsException if more than one PrintFullStackTraceAnnotation is specified" in {
    val a = Seq(PrintFullStackTraceAnnotation, PrintFullStackTraceAnnotation)
    checkExceptionMessage(Checks, a, "At most one PrintFullStackTraceAnnotation")
  }

  it should "throw an OptionsException if more than one ChiselOutputFileAnnotation is specified" in {
    val a = Seq(ChiselOutputFileAnnotation("foo"), ChiselOutputFileAnnotation("bar"))
    checkExceptionMessage(Checks, a, "At most one Chisel output file")
  }

}
