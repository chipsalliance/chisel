// SPDX-License-Identifier: Apache-2.0

package chiselTests

import _root_.logger.{LogLevel, LogLevelAnnotation, Logger}
import chisel3._
import chisel3.stage.{ChiselGeneratorAnnotation, PrintFullStackTraceAnnotation}
import chisel3.testers._
import circt.stage.{CIRCTTarget, CIRCTTargetAnnotation, ChiselStage}
import chisel3.simulator._
import svsim._
import firrtl.annotations.Annotation
import firrtl.ir.Circuit
import firrtl.stage.FirrtlCircuitAnnotation
import firrtl.util.BackendCompilationUtilities
import firrtl.{AnnotationSeq, EmittedVerilogCircuitAnnotation}
import org.scalacheck._
import org.scalatest._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.propspec.AnyPropSpec
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks
import org.scalactic.source.Position

import java.io.{ByteArrayOutputStream, PrintStream}
import java.security.Permission
import scala.reflect.ClassTag
import java.text.SimpleDateFormat
import java.util.Calendar
import chisel3.reflect.DataMirror

/** Common utility functions for Chisel unit tests. */
sealed trait ChiselRunners extends Assertions {

  def elaborateAndGetModule[A <: RawModule](t: => A): A = {
    var res: Any = null
    ChiselStage.emitCHIRRTL {
      res = t
      res.asInstanceOf[A]
    }
    res.asInstanceOf[A]
  }

  /** Compiles a Chisel Module to FIRRTL
    * NOTE: This uses the "test_run_dir" as the default directory for generated code.
    * @param t the generator for the module
    * @return The FIRRTL Circuit and Annotations _before_ FIRRTL compilation
    */
  def getFirrtlAndAnnos(t: => RawModule, providedAnnotations: Seq[Annotation] = Nil): (Circuit, Seq[Annotation]) = {
    TestUtils.getChirrtlAndAnnotations(t, providedAnnotations)
  }
}

trait WidthHelpers extends Assertions {

  def assertKnownWidth(expected: Int, args: Iterable[String] = Nil)(gen: => Data)(implicit pos: Position): Unit = {
    class TestModule extends Module {
      val testPoint = gen
      assert(testPoint.getWidth === expected)
      val out = IO(chiselTypeOf(testPoint))
      // Sanity check that firrtl doesn't change the width
      val zero = 0.U(0.W).asTypeOf(chiselTypeOf(testPoint))
      if (DataMirror.isWire(testPoint)) {
        testPoint := zero
      }
      out := zero
      out := testPoint
    }
    val verilog = ChiselStage.emitSystemVerilog(new TestModule, args.toArray, Array("-disable-all-randomization"))
    expected match {
      case 0 => assert(!verilog.contains("out"))
      case 1 =>
        assert(verilog.contains(s"out"))
        assert(!verilog.contains(s"0] out"))
      case _ => assert(verilog.contains(s"[${expected - 1}:0] out"))
    }
  }

  def assertInferredWidth(expected: Int, args: Iterable[String] = Nil)(gen: => Data)(implicit pos: Position): Unit = {
    class TestModule extends Module {
      val testPoint = gen
      assert(!testPoint.isWidthKnown, s"Asserting that width should be inferred yet width is known to Chisel!")
      // Sanity check that firrtl doesn't change the width
      val widthcheck = Wire(chiselTypeOf(testPoint))
      dontTouch(widthcheck)
      val zero = 0.U(0.W).asTypeOf(chiselTypeOf(testPoint))
      if (DataMirror.isWire(testPoint)) {
        testPoint := zero
      }
      widthcheck := zero
      widthcheck := testPoint
    }
    val verilog =
      ChiselStage.emitSystemVerilog(new TestModule, args.toArray :+ "--dump-fir", Array("-disable-all-randomization"))
    expected match {
      case 0 => assert(!verilog.contains("widthcheck"))
      case 1 =>
        assert(verilog.contains(s"widthcheck"))
        assert(!verilog.contains(s"0] widthcheck"))
      case _ => assert(verilog.contains(s"[${expected - 1}:0] widthcheck"))
    }
  }

}

trait FileCheck extends BeforeAndAfterEachTestData { this: Suite =>
  import scala.Console.{withErr, withOut}

  private def sanitize(n: String): String = n.replaceAll(" ", "_").replaceAll("\\W+", "")

  private val testRunDir: os.Path = os.pwd / os.RelPath(BackendCompilationUtilities.TestDirectory)
  private val suiteDir:   os.Path = testRunDir / sanitize(suiteName)
  private var checkFile:  Option[os.Path] = None

  override def beforeEach(testData: TestData): Unit = {
    // TODO check that these are always available
    val nameDir = suiteDir / sanitize(testData.name)
    os.makeDir.all(nameDir)
    checkFile = Some(nameDir / s"${sanitize(testData.text)}.check")
    super.beforeEach(testData) // To be stackable, must call super.beforeEach
  }

  override def afterEach(testData: TestData): Unit = {
    checkFile = None
    super.afterEach(testData) // To be stackable, must call super.beforeEach
  }

  /** Run FileCheck on a String against some checks */
  def fileCheckString(in: String, fileCheckArgs: String*)(check: String): Unit = {
    // Filecheck needs the thing to check in a file
    os.write.over(checkFile.get, check)
    val extraArgs = os.Shellable(fileCheckArgs)
    os.proc("FileCheck", "--allow-empty", checkFile.get, extraArgs).call(stdin = in)
  }

  /** Elaborate a Module to FIRRTL and check the FIRRTL with FileCheck */
  def generateFirrtlAndFileCheck(t: => RawModule, fileCheckArgs: String*)(check: String): Unit = {
    fileCheckString(ChiselStage.emitCHIRRTL(t), fileCheckArgs: _*)(check)
  }

  /** Generate SystemVerilog and run it through FileCheck */
  def generateSystemVerilogAndFileCheck(t: => RawModule, fileCheckArgs: String*)(check: String): Unit = {
    fileCheckString(ChiselStage.emitSystemVerilog(t), fileCheckArgs: _*)(check)
  }

  /** Elaborate a Module, capture the stdout and stderr, check stdout and stderr with FileCheck */
  def elaborateAndFileCheckOutAndErr(t: => RawModule, fileCheckArgs: String*)(check: String): Unit = {
    val outStream = new ByteArrayOutputStream()
    withOut(outStream)(withErr(outStream)(ChiselStage.emitCHIRRTL(t)))
    val result = outStream.toString
    fileCheckString(outStream.toString, fileCheckArgs: _*)(check)
  }
}

/** Spec base class for BDD-style testers. */
abstract class ChiselFlatSpec extends AnyFlatSpec with ChiselRunners with Matchers

/** Spec base class for BDD-style testers. */
abstract class ChiselFreeSpec extends AnyFreeSpec with ChiselRunners with Matchers

/** Spec base class for BDD-style testers. */
abstract class ChiselFunSpec extends AnyFunSpec with ChiselRunners with Matchers

/** Utilities for writing property-based checks */
trait PropertyUtils extends ScalaCheckPropertyChecks {

  // Constrain the default number of instances generated for every use of forAll.
  implicit override val generatorDrivenConfig: PropertyCheckConfiguration =
    PropertyCheckConfiguration(minSuccessful = 8, minSize = 1, sizeRange = 3)

  // Generator for small positive integers.
  val smallPosInts = Gen.choose(1, 4)

  // Generator for positive (ascending or descending) ranges.
  def posRange: Gen[Range] = for {
    dir <- Gen.oneOf(true, false)
    step <- Gen.choose(1, 3)
    m <- Gen.choose(1, 10)
    n <- Gen.choose(1, 10)
  } yield {
    if (dir) {
      Range(m, (m + n) * step, step)
    } else {
      Range((m + n) * step, m, -step)
    }
  }

  // Generator for widths considered "safe".
  val safeUIntWidth = Gen.choose(1, 30)

  // Generators for integers that fit within "safe" widths.
  val safeUInts = Gen.choose(0, (1 << 30))

  // Generators for vector sizes.
  val vecSizes = Gen.choose(0, 4)

  // Generator for string representing an arbitrary integer.
  val binaryString = for (i <- Arbitrary.arbitrary[Int]) yield "b" + i.toBinaryString

  // Generator for a sequence of Booleans of size n.
  def enSequence(n: Int): Gen[List[Boolean]] = Gen.containerOfN[List, Boolean](n, Gen.oneOf(true, false))

  // Generator which gives a width w and a list (of size n) of numbers up to w bits.
  def safeUIntN(n: Int): Gen[(Int, List[Int])] = for {
    w <- smallPosInts
    i <- Gen.containerOfN[List, Int](n, Gen.choose(0, (1 << w) - 1))
  } yield (w, i)

  // Generator which gives a width w and a numbers up to w bits.
  val safeUInt = for {
    w <- smallPosInts
    i <- Gen.choose(0, (1 << w) - 1)
  } yield (w, i)

  // Generator which gives a width w and a list (of size n) of a pair of numbers up to w bits.
  def safeUIntPairN(n: Int): Gen[(Int, List[(Int, Int)])] = for {
    w <- smallPosInts
    i <- Gen.containerOfN[List, Int](n, Gen.choose(0, (1 << w) - 1))
    j <- Gen.containerOfN[List, Int](n, Gen.choose(0, (1 << w) - 1))
  } yield (w, i.zip(j))

  // Generator which gives a width w and a pair of numbers up to w bits.
  val safeUIntPair = for {
    w <- smallPosInts
    i <- Gen.choose(0, (1 << w) - 1)
    j <- Gen.choose(0, (1 << w) - 1)
  } yield (w, i, j)

}

/** Spec base class for property-based testers. */
abstract class ChiselPropSpec extends AnyPropSpec with ChiselRunners with PropertyUtils with Matchers

trait Utils {

  /** Run some Scala thunk and return STDOUT and STDERR as strings.
    * @param thunk some Scala code
    * @return a tuple containing STDOUT, STDERR, and what the thunk returns
    */
  def grabStdOutErr[T](thunk: => T): (String, String, T) = {
    val stdout, stderr = new ByteArrayOutputStream()
    val ret = scala.Console.withOut(stdout) { scala.Console.withErr(stderr) { thunk } }
    (stdout.toString, stderr.toString, ret)
  }

  /** Run some Scala thunk and return all logged messages as Strings
    * @param thunk some Scala code
    * @return a tuple containing LOGGED, and what the thunk returns
    */
  def grabLog[T](thunk: => T): (String, T) = grabLogLevel(LogLevel.default)(thunk)

  /** Run some Scala thunk and return all logged messages as Strings
    * @param level the log level to use
    * @param thunk some Scala code
    * @return a tuple containing LOGGED, and what the thunk returns
    */
  def grabLogLevel[T](level: LogLevel.Value)(thunk: => T): (String, T) = {
    val baos = new ByteArrayOutputStream()
    val stream = new PrintStream(baos, true, "utf-8")
    val ret = Logger.makeScope(LogLevelAnnotation(level) :: Nil) {
      Logger.setOutput(stream)
      thunk
    }
    (baos.toString, ret)
  }

  /** Encodes a System.exit exit code
    * @param status the exit code
    */
  private case class ExitException(status: Int) extends SecurityException(s"Found a sys.exit with code $status")

  /** Run some code and rethrow an exception with a specific type if an exception of that type occurs anywhere in the
    * stack trace.
    *
    * This is useful for "extracting" one exception that may be wrapped by other exceptions.
    *
    * Example usage:
    * {{{
    * a [ChiselException] should be thrownBy extractCause[ChiselException] { /* ... */ }
    * }}}
    *
    * @param thunk some code to run
    * @tparam A the type of the exception to extract
    * @return nothing
    */
  def extractCause[A <: Throwable: ClassTag](thunk: => Any): Unit = {
    def unrollCauses(a: Throwable): Seq[Throwable] = a match {
      case null => Seq.empty
      case _    => a +: unrollCauses(a.getCause)
    }

    val exceptions: Seq[_ <: Throwable] =
      try {
        thunk
        Seq.empty
      } catch {
        case a: Throwable => unrollCauses(a)
      }

    exceptions.collectFirst { case a: A => a } match {
      case Some(a) => throw a
      case None =>
        exceptions match {
          case Nil    => ()
          case h :: t => throw h
        }
    }

  }
}
