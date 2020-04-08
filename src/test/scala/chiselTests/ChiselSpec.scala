// See LICENSE for license details.

package chiselTests

import org.scalatest._
import org.scalatest.prop._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalacheck._
import chisel3._
import chisel3.testers._
import firrtl.{AnnotationSeq, CommonOptions, ExecutionOptionsManager, FirrtlExecutionFailure, FirrtlExecutionSuccess, HasFirrtlOptions}
import firrtl.util.BackendCompilationUtilities
import java.io.ByteArrayOutputStream
import java.security.Permission
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks
import scala.reflect.ClassTag

/** Common utility functions for Chisel unit tests. */
trait ChiselRunners extends Assertions with BackendCompilationUtilities {
  def runTester(t: => BasicTester,
                additionalVResources: Seq[String] = Seq(),
                annotations: AnnotationSeq = Seq()
               ): Boolean = {
    TesterDriver.execute(() => t, additionalVResources, annotations)
  }
  def assertTesterPasses(t: => BasicTester,
                         additionalVResources: Seq[String] = Seq(),
                         annotations: AnnotationSeq = Seq()
                        ): Unit = {
    assert(runTester(t, additionalVResources, annotations))
  }
  def assertTesterFails(t: => BasicTester,
                        additionalVResources: Seq[String] = Seq(),
                        annotations: Seq[chisel3.aop.Aspect[_]] = Seq()
                       ): Unit = {
    assert(!runTester(t, additionalVResources, annotations))
  }
  def elaborate(t: => RawModule): Unit = Driver.elaborate(() => t)

  def assertKnownWidth(expected: Int)(gen: => Data): Unit = {
    assertTesterPasses(new BasicTester {
      val x = gen
      assert(x.getWidth === expected)
      // Sanity check that firrtl doesn't change the width
      x := 0.U.asTypeOf(chiselTypeOf(x))
      val (_, done) = chisel3.util.Counter(true.B, 2)
      when (done) {
        chisel3.assert(~(x.asUInt) === -1.S(expected.W).asUInt)
        stop()
      }
    })
  }

  def assertInferredWidth(expected: Int)(gen: => Data): Unit = {
    assertTesterPasses(new BasicTester {
      val x = gen
      assert(!x.isWidthKnown, s"Asserting that width should be inferred yet width is known to Chisel!")
      x := 0.U.asTypeOf(chiselTypeOf(x))
      val (_, done) = chisel3.util.Counter(true.B, 2)
      when (done) {
        chisel3.assert(~(x.asUInt) === -1.S(expected.W).asUInt)
        stop()
      }
    })
  }

  /** Given a generator, return the Firrtl that it generates.
    *
    * @param t Module generator
    * @return Firrtl representation as a String
    */
  def generateFirrtl(t: => RawModule): String = Driver.emit(() => t)

  /** Compiles a Chisel Module to Verilog
    * NOTE: This uses the "test_run_dir" as the default directory for generated code.
    * @param t the generator for the module
    * @return the Verilog code as a string.
    */
  def compile(t: => RawModule): String = {
    val testDir = createTestDirectory(this.getClass.getSimpleName)
    val manager = new ExecutionOptionsManager("compile") with HasFirrtlOptions
                                                         with HasChiselExecutionOptions {
      commonOptions = CommonOptions(targetDirName = testDir.toString)
    }

    Driver.execute(manager, () => t) match {
      case ChiselExecutionSuccess(_, _, Some(firrtlExecRes)) =>
        firrtlExecRes match {
          case FirrtlExecutionSuccess(_, verilog) => verilog
          case FirrtlExecutionFailure(msg) => fail(msg)
        }
      case ChiselExecutionSuccess(_, _, None) => fail() // This shouldn't happen
      case ChiselExecutionFailure(msg) => fail(msg)
    }
  }
}

/** Spec base class for BDD-style testers. */
abstract class ChiselFlatSpec extends AnyFlatSpec with ChiselRunners with Matchers

class ChiselTestUtilitiesSpec extends ChiselFlatSpec {
  import org.scalatest.exceptions.TestFailedException
  // Who tests the testers?
  "assertKnownWidth" should "error when the expected width is wrong" in {
    val caught = intercept[ChiselException] {
      assertKnownWidth(7) {
        Wire(UInt(8.W))
      }
    }
    assert(caught.getCause.isInstanceOf[TestFailedException])
  }

  it should "error when the width is unknown" in {
    a [ChiselException] shouldBe thrownBy {
      assertKnownWidth(7) {
        Wire(UInt())
      }
    }
  }

  it should "work if the width is correct" in {
    assertKnownWidth(8) {
      Wire(UInt(8.W))
    }
  }

  "assertInferredWidth" should "error if the width is known" in {
    val caught = intercept[ChiselException] {
      assertInferredWidth(8) {
        Wire(UInt(8.W))
      }
    }
    assert(caught.getCause.isInstanceOf[TestFailedException])
  }

  it should "error if the expected width is wrong" in {
    a [TestFailedException] shouldBe thrownBy {
      assertInferredWidth(8) {
        val w = Wire(UInt())
        w := 2.U(2.W)
        w
      }
    }
  }

  it should "pass if the width is correct" in {
    assertInferredWidth(4) {
      val w = Wire(UInt())
      w := 2.U(4.W)
      w
    }
  }
}

/** Spec base class for property-based testers. */
class ChiselPropSpec extends PropSpec with ChiselRunners with ScalaCheckPropertyChecks with Matchers {

  // Constrain the default number of instances generated for every use of forAll.
  implicit override val generatorDrivenConfig: PropertyCheckConfiguration =
    PropertyCheckConfiguration(minSuccessful = 8, minSize = 1, sizeRange = 3)

  // Generator for small positive integers.
  val smallPosInts = Gen.choose(1, 4)

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
  } yield (w, i zip j)

  // Generator which gives a width w and a pair of numbers up to w bits.
  val safeUIntPair = for {
    w <- smallPosInts
    i <- Gen.choose(0, (1 << w) - 1)
    j <- Gen.choose(0, (1 << w) - 1)
  } yield (w, i, j)
}

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

  /** Encodes a System.exit exit code
    * @param status the exit code
    */
  private case class ExitException(status: Int) extends SecurityException(s"Found a sys.exit with code $status")

  /** A security manager that converts calls to System.exit into [[ExitException]]s by explicitly disabling the ability of
    * a thread to actually exit. For more information, see:
    *   - https://docs.oracle.com/javase/tutorial/essential/environment/security.html
    */
  private class ExceptOnExit extends SecurityManager {
    override def checkPermission(perm: Permission): Unit = {}
    override def checkPermission(perm: Permission, context: Object): Unit = {}
    override def checkExit(status: Int): Unit = {
      super.checkExit(status)
      throw ExitException(status)
    }
  }

  /** Encodes a file that some code tries to write to
    * @param the file name
    */
  private case class WriteException(file: String) extends SecurityException(s"Tried to write to file $file")

  /** A security manager that converts writes to any file into [[WriteException]]s.
    */
  private class ExceptOnWrite extends SecurityManager {
    override def checkPermission(perm: Permission): Unit = {}
    override def checkPermission(perm: Permission, context: Object): Unit = {}
    override def checkWrite(file: String): Unit = {
      super.checkWrite(file)
      throw WriteException(file)
    }
  }

  /** Run some Scala code (a thunk) in an environment where all System.exit are caught and returned. This avoids a
    * situation where a test results in something actually exiting and killing the entire test. This is necessary if you
    * want to test a command line program, e.g., the `main` method of [[firrtl.options.Stage Stage]].
    *
    * NOTE: THIS WILL NOT WORK IN SITUATIONS WHERE THE THUNK IS CATCHING ALL [[Exception]]s OR [[Throwable]]s, E.G.,
    * SCOPT. IF THIS IS HAPPENING THIS WILL NOT WORK. REPEAT THIS WILL NOT WORK.
    * @param thunk some Scala code
    * @return either the output of the thunk (`Right[T]`) or an exit code (`Left[Int]`)
    */
  def catchStatus[T](thunk: => T): Either[Int, T] = {
    try {
      System.setSecurityManager(new ExceptOnExit())
      Right(thunk)
    } catch {
      case ExitException(a) => Left(a)
    } finally {
      System.setSecurityManager(null)
    }
  }

  /** Run some Scala code (a thunk) in an environment where file writes are caught and the file that a program tries to
    * write to is returned. This is useful if you want to test that some thunk either tries to write to a specific file
    * or doesn't try to write at all.
    */
  def catchWrites[T](thunk: => T): Either[String, T] = {
    try {
      System.setSecurityManager(new ExceptOnWrite())
      Right(thunk)
    } catch {
      case WriteException(a) => Left(a)
    } finally {
      System.setSecurityManager(null)
    }
  }

  /** Run some code extracting an exception cause that matches a type parameter
    * @param thunk some code to run
    * @tparam A the type of the exception to expect
    * @return nothing
    * @throws the exception of type parameter A if it was found
    */
  def extractCause[A <: Throwable : ClassTag](thunk: => Any): Unit = {
    def unrollCauses(a: Throwable): Seq[Throwable] = a match {
      case null => Seq.empty
      case _    => a +: unrollCauses(a.getCause)
    }

    val exceptions: Seq[_ <: Throwable] = try {
      thunk
      Seq.empty
    } catch {
      case a: Throwable => unrollCauses(a)
    }

    exceptions.collectFirst{ case a: A => a } match {
      case Some(a) => throw a
      case None => exceptions match {
        case Nil    => Unit
        case h :: t => throw h
      }
    }

  }

}
