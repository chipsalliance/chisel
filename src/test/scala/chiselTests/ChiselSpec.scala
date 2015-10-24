// See LICENSE for license details.

package chiselTests

import org.scalatest._
import org.scalatest.prop._
import org.scalacheck._
import Chisel._
import Chisel.testers._

/** Common utility functions for Chisel unit tests. */
trait ChiselRunners {
  def execute(t: => BasicTester): Boolean = TesterDriver.execute(t)
  def elaborate(t: => Module): Circuit = TesterDriver.elaborate(t)
}

/** Spec base class for BDD-style testers. */
class ChiselFlatSpec extends FlatSpec with ChiselRunners with Matchers

/** Spec base class for property-based testers. */
class ChiselPropSpec extends PropSpec with ChiselRunners with PropertyChecks {
  /** Returns the number of 1s in the binary representation of the input. */
  def popCount(n: Long): Int = n.toBinaryString.count(_ == '1')

  // Generator for small positive integers.
  val smallPosInts = Gen.choose(1, 7)

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
