// See LICENSE for license details.

package chiselTests

import java.io.File
import org.scalatest._
import org.scalatest.prop._
import org.scalacheck._
import chisel3._
import chisel3.experimental.RawModule
import chisel3.testers._
import firrtl.{
  CommonOptions,
  ExecutionOptionsManager,
  HasFirrtlOptions,
  FirrtlExecutionSuccess,
  FirrtlExecutionFailure
}
import firrtl.util.BackendCompilationUtilities

/** Common utility functions for Chisel unit tests. */
trait ChiselRunners extends Assertions with BackendCompilationUtilities {
  def runTester(t: => BasicTester, additionalVResources: Seq[String] = Seq()): Boolean = {
    TesterDriver.execute(() => t, additionalVResources)
  }
  def assertTesterPasses(t: => BasicTester, additionalVResources: Seq[String] = Seq()): Unit = {
    assert(runTester(t, additionalVResources))
  }
  def assertTesterFails(t: => BasicTester, additionalVResources: Seq[String] = Seq()): Unit = {
    assert(!runTester(t, additionalVResources))
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
abstract class ChiselFlatSpec extends FlatSpec with ChiselRunners with Matchers

class ChiselTestUtilitiesSpec extends ChiselFlatSpec {
  import org.scalatest.exceptions.TestFailedException
  // Who tests the testers?
  "assertKnownWidth" should "error when the expected width is wrong" in {
    a [TestFailedException] shouldBe thrownBy {
      assertKnownWidth(7) {
        Wire(UInt(8.W))
      }
    }
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
    a [TestFailedException] shouldBe thrownBy {
      assertInferredWidth(8) {
        Wire(UInt(8.W))
      }
    }
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
class ChiselPropSpec extends PropSpec with ChiselRunners with PropertyChecks with Matchers {

  // Constrain the default number of instances generated for every use of forAll.
  implicit override val generatorDrivenConfig =
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
