// SPDX-License-Identifier: Apache-2.0

package chiselTests

import org.scalacheck.{Arbitrary, Gen}
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

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
