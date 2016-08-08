// See LICENSE for license details.

package chisel3.iotesters

import org.scalatest._
import org.scalatest.prop._
import org.scalacheck._
import chisel3._
import chisel3.testers._

/** Common utility functions for Chisel unit tests. */
trait ChiselRunners extends Assertions {
  val backends = Seq("firrtl", "verilator", "vcs")
  def runTester(t: => BasicTester, additionalVResources: Seq[String] = Seq()): Boolean = {
    TesterDriver.execute(() => t, additionalVResources)
  }
  def assertTesterPasses(t: => BasicTester, additionalVResources: Seq[String] = Seq()): Unit = {
    assert(runTester(t, additionalVResources))
  }
  def elaborate(t: => Module): Unit = Driver.elaborate(() => t)
}

/** Spec base class for BDD-style testers. */
class ChiselFlatSpec extends FlatSpec with ChiselRunners with Matchers

/** Spec base class for property-based testers. */
class ChiselPropSpec extends PropSpec with ChiselRunners with PropertyChecks {

  // Constrain the default number of instances generated for every use of forAll.
  implicit override val generatorDrivenConfig =
    PropertyCheckConfig(minSuccessful = 8, minSize = 1, maxSize = 4)

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
