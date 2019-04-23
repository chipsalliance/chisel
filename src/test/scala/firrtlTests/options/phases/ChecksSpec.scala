// See LICENSE for license details.

package firrtlTests.options.phases

import org.scalatest.{FlatSpec, Matchers}

import firrtl.AnnotationSeq
import firrtl.options.{OptionsException, OutputAnnotationFileAnnotation, Phase, TargetDirAnnotation}
import firrtl.options.phases.Checks

class ChecksSpec extends FlatSpec with Matchers {

  val targetDir = TargetDirAnnotation("foo")
  val annoOut = OutputAnnotationFileAnnotation("bar")

  class Fixture { val phase: Phase = new Checks }

  /* A minimum annotation Seq that will pass [[Checks]] */
  val min = Seq(targetDir)

  def checkExceptionMessage(phase: Phase, annotations: AnnotationSeq, messageStart: String): Unit =
    intercept[OptionsException]{ phase.transform(annotations) }.getMessage should startWith(messageStart)

  behavior of classOf[Checks].toString

  it should "enforce exactly one TargetDirAnnotation" in new Fixture {
    info("0 target directories throws an exception")
    checkExceptionMessage(phase, Seq.empty, "Exactly one target directory must be specified")

    info("2 target directories throws an exception")
    checkExceptionMessage(phase, Seq(targetDir, targetDir), "Exactly one target directory must be specified")
  }

  it should "enforce zero or one output annotation files" in new Fixture {
    info("0 output annotation files is okay")
    phase.transform(Seq(targetDir))

    info("2 output annotation files throws an exception")
    val in = Seq(targetDir, annoOut, annoOut)
    checkExceptionMessage(phase, in, "At most one output annotation file can be specified")
  }

  it should "pass if the minimum annotations are specified" in new Fixture {
    info(s"""Minimum required: ${min.map(_.getClass.getSimpleName).mkString(", ")}""")
    phase.transform(min)
  }

}
