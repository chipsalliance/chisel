// SPDX-License-Identifier: Apache-2.0

package firrtlTests.stage.phases

import firrtl.stage._

import firrtl.AnnotationSeq
import firrtl.options.{OptionsException, OutputAnnotationFileAnnotation, Phase}
import firrtl.stage.phases.Checks
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ChecksSpec extends AnyFlatSpec with Matchers {

  class Fixture { val phase: Phase = new Checks }

  val inputCircuit = FirrtlCircuitAnnotation(firrtl.ir.Circuit(firrtl.ir.NoInfo, Seq.empty, "Foo"))
  val outputFile = OutputFileAnnotation("bar")
  val outputAnnotationFile = OutputAnnotationFileAnnotation("baz")
  val infoMode = InfoModeAnnotation("ignore")

  val min = Seq(inputCircuit, infoMode)

  def checkExceptionMessage(phase: Phase, annotations: AnnotationSeq, messageStart: String): Unit =
    intercept[OptionsException] { phase.transform(annotations) }.getMessage should startWith(messageStart)

  behavior.of(classOf[Checks].toString)

  it should "require exactly one input source" in new Fixture {
    info("0 input source causes an exception")
    checkExceptionMessage(phase, Seq.empty, "Unable to determine FIRRTL source to read")

    info("2 input sources causes an exception")
    val in = min :+ inputCircuit
    checkExceptionMessage(phase, in, "Multiply defined input FIRRTL sources")
  }

  it should "enforce zero or one output files" in new Fixture {
    val in = min ++ Seq(outputFile, outputFile)
    checkExceptionMessage(phase, in, "No more than one output file can be specified")
  }

  it should "validate info mode names" in new Fixture {
    info("Good info mode names should work")
    Seq("ignore", "use", "gen", "append")
      .map(info => phase.transform(Seq(inputCircuit, InfoModeAnnotation(info))))
  }

  it should "enforce exactly one info mode" in new Fixture {
    info("0 info modes should throw an exception")
    checkExceptionMessage(
      phase,
      Seq(inputCircuit),
      "Exactly one info mode must be specified, but none found"
    )

    info("2 info modes should throw an exception")
    val i = infoMode.modeName
    checkExceptionMessage(phase, min :+ infoMode, s"Exactly one info mode must be specified, but found '$i, $i'")
  }

  it should "pass if the minimum annotations are specified" in new Fixture {
    info(s"""Minimum required: ${min.map(_.getClass.getSimpleName).mkString(", ")}""")
    phase.transform(min)
  }

}
