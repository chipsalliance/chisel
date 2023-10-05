// SPDX-License-Identifier: Apache-2.0

package firrtlTests.options

import firrtl.AnnotationSeq
import firrtl.annotations.{Annotation, NoTargetAnnotation}
import firrtl.options.{DoNotTerminateOnExit, DuplicateHandling, ExceptOnError, OptionsException}

import scopt.OptionParser

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class OptionParserSpec extends AnyFlatSpec with Matchers with firrtl.testutils.Utils {

  case class IntAnnotation(x: Int) extends NoTargetAnnotation {
    def extract: Int = x
  }

  /* An option parser that prepends to a Seq[Int] */
  class IntParser extends OptionParser[AnnotationSeq]("Int Parser") {
    opt[Int]("integer").abbr("n").unbounded().action((x, c) => IntAnnotation(x) +: c)
    help("help")
  }

  trait DuplicateShortOption { this: OptionParser[AnnotationSeq] =>
    opt[Int]("not-an-integer").abbr("n").unbounded().action((x, c) => IntAnnotation(x) +: c)
  }

  trait DuplicateLongOption { this: OptionParser[AnnotationSeq] =>
    opt[Int]("integer").abbr("m").unbounded().action((x, c) => IntAnnotation(x) +: c)
  }

  trait WithIntParser { val parser = new IntParser }

  behavior.of("A default OptionsParser")

  it should "print to stderr on an invalid option" in new WithIntParser {
    grabStdOutErr { parser.parse(Array("--foo"), Seq[Annotation]()) }._2 should include("Unknown option --foo")
  }

  behavior.of("An OptionParser with DoNotTerminateOnExit mixed in")

  behavior.of("An OptionParser with DuplicateHandling mixed in")

  it should "detect short duplicates" in {
    val parser = new IntParser with DuplicateHandling with DuplicateShortOption
    intercept[OptionsException] { parser.parse(Array[String](), Seq[Annotation]()) }.getMessage should startWith(
      "Duplicate short option"
    )
  }

  it should "detect long duplicates" in {
    val parser = new IntParser with DuplicateHandling with DuplicateLongOption
    intercept[OptionsException] { parser.parse(Array[String](), Seq[Annotation]()) }.getMessage should startWith(
      "Duplicate long option"
    )
  }

  behavior.of("An OptionParser with ExceptOnError mixed in")

  it should "cause an OptionsException on an invalid option" in {
    val parser = new IntParser with ExceptOnError
    intercept[OptionsException] { parser.parse(Array("--foo"), Seq[Annotation]()) }.getMessage should include(
      "Unknown option"
    )
  }

}
