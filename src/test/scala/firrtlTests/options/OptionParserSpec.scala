// See LICENSE for license details

package firrtlTests.options

import firrtl.{AnnotationSeq, FIRRTLException}
import firrtl.annotations.{Annotation, NoTargetAnnotation}
import firrtl.options.{DoNotTerminateOnExit, DuplicateHandling, OptionsException}

import scopt.OptionParser

import org.scalatest.{FlatSpec, Matchers}

import java.security.Permission

class OptionParserSpec extends FlatSpec with Matchers {

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

  case class ExitException(status: Option[Int]) extends SecurityException("Found a sys.exit")

  /* Security manager that disallows calls to sys.exit */
  class ExceptOnExit extends SecurityManager {
    override def checkPermission(perm: Permission): Unit = {}
    override def checkPermission(perm: Permission, context: Object): Unit = {}
    override def checkExit(status: Int): Unit = {
      super.checkExit(status)
      throw ExitException(Some(status))
    }
  }

  /* Tell a parser to terminate in an environment where sys.exit throws an exception */
  def catchStatus(parser: OptionParser[_], exitState: Either[String, Unit]): Option[Int] = {
    System.setSecurityManager(new ExceptOnExit())
    val status = try {
      parser.terminate(exitState)
      throw new ExitException(None)
    } catch {
      case ExitException(s) => s
    }
    System.setSecurityManager(null)
    status
  }

  behavior of "default OptionsParser"

  it should "terminate on exit" in {
    val parser = new IntParser

    info("By default, exit statuses are reported")
    catchStatus(parser, Left("some message")) should be (Some(1))
    catchStatus(parser, Right(Unit))          should be (Some(0))
  }

  behavior of "DoNotTerminateOnExit"

  it should "disable sys.exit for terminate method" in {
    val parser = new IntParser with DoNotTerminateOnExit
    catchStatus(parser, Left("some message")) should be (None)
    catchStatus(parser, Right(Unit))          should be (None)
  }

  behavior of "DuplicateHandling"

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

}
