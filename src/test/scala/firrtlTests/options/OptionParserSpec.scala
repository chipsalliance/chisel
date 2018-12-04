// See LICENSE for license details

package firrtlTests.options

import firrtl.{AnnotationSeq, FIRRTLException}
import firrtl.annotations.{Annotation, NoTargetAnnotation}
import firrtl.options.{DoNotTerminateOnExit, DuplicateHandling, ExceptOnError, OptionsException}

import scopt.OptionParser

import org.scalatest.{FlatSpec, Matchers}

class OptionParserSpec extends FlatSpec with Matchers with firrtlTests.Utils {

  case class IntAnnotation(x: Int) extends NoTargetAnnotation {
    def extract: Int = x
  }

  /* An option parser that prepends to a Seq[Int] */
  class IntParser extends OptionParser[AnnotationSeq]("Int Parser") {
    opt[Int]("integer").abbr("n").unbounded.action( (x, c) => IntAnnotation(x) +: c )
    help("help")
  }

  trait DuplicateShortOption { this: OptionParser[AnnotationSeq] =>
    opt[Int]("not-an-integer").abbr("n").unbounded.action( (x, c) => IntAnnotation(x) +: c )
  }

  trait DuplicateLongOption { this: OptionParser[AnnotationSeq] =>
      opt[Int]("integer").abbr("m").unbounded.action( (x, c) => IntAnnotation(x) +: c )
  }

  trait WithIntParser { val parser = new IntParser }

  behavior of "A default OptionsParser"

  it should "call sys.exit if terminate is called" in new WithIntParser {
    info("exit status of 1 for failure")
    catchStatus { parser.terminate(Left("some message")) } should be (Left(1))

    info("exit status of 0 for success")
    catchStatus { parser.terminate(Right(Unit)) } should be (Left(0))
  }

  it should "print to stderr on an invalid option" in new WithIntParser {
    grabStdOutErr{ parser.parse(Array("--foo"), Seq[Annotation]()) }._2 should include ("Unknown option --foo")
  }

  behavior of "An OptionParser with DoNotTerminateOnExit mixed in"

  it should "disable sys.exit for terminate method" in {
    val parser = new IntParser with DoNotTerminateOnExit

    info("no exit for failure")
    catchStatus { parser.terminate(Left("some message")) } should be (Right(()))

    info("no exit for success")
    catchStatus { parser.terminate(Right(Unit)) } should be (Right(()))
  }

  behavior of "An OptionParser with DuplicateHandling mixed in"

  it should "detect short duplicates" in {
    val parser = new IntParser with DuplicateHandling with DuplicateShortOption
    intercept[OptionsException] { parser.parse(Array[String](), Seq[Annotation]()) }
      .getMessage should startWith ("Duplicate short option")
  }

  it should "detect long duplicates" in {
    val parser = new IntParser with DuplicateHandling with DuplicateLongOption
    intercept[OptionsException] { parser.parse(Array[String](), Seq[Annotation]()) }
      .getMessage should startWith ("Duplicate long option")
  }

  behavior of "An OptionParser with ExceptOnError mixed in"

  it should "cause an OptionsException on an invalid option" in {
    val parser = new IntParser with ExceptOnError
    intercept[OptionsException] { parser.parse(Array("--foo"), Seq[Annotation]()) }
      .getMessage should include ("Unknown option")
  }

}
